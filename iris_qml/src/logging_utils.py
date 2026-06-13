"""
logging_utils.py — Centralised logging + a Sampler wrapper that PROVES
where the quantum circuits actually run.

Two things live here:

1. ``setup_logging()`` configures one logger ("qml") that writes both to the
   console and to ``outputs/quantum_run.log`` with timestamps. Call it once at
   start-up.

2. ``LoggingSamplerV2`` wraps ANY Qiskit V2 sampler (StatevectorSampler,
   Aer SamplerV2, or the IBM Runtime SamplerV2). Every time qiskit-machine-learning
   submits circuits, the wrapper logs:
       - how many PUBs (circuits) were sent,
       - the qubit count / depth of the first circuit,
       - the resulting job id and the backend name.

   For IBM hardware the logged job id is the *real* IBM job id — you can paste it
   into https://quantum.ibm.com/jobs to confirm the run executed on a QPU.
   This is the verification the rest of the pipeline relies on.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from qiskit.primitives.base import BaseSamplerV2

LOGGER_NAME = "qml"


# ─────────────────────────────────────────────────────────────────
# Logger configuration
# ─────────────────────────────────────────────────────────────────
def setup_logging(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    """Configure and return the shared 'qml' logger (idempotent)."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    if logger.handlers:  # already configured
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.info("Logging to file: %s", log_file.resolve())

    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


# ─────────────────────────────────────────────────────────────────
# Sampler wrapper — the "proof of execution" layer
# ─────────────────────────────────────────────────────────────────
def _extract_circuit(pub):
    """A PUB may be a SamplerPub, a (circuit, params, shots) tuple, or a circuit."""
    if hasattr(pub, "circuit"):
        return pub.circuit
    if isinstance(pub, (tuple, list)) and len(pub) > 0:
        return pub[0]
    return pub


class _DiagnosticJob:
    """Wraps a sampler job so that, if .result() fails, the REAL IBM error
    message is logged (instead of the generic 'Sampler job failed!'), with an
    optional automatic retry for transient hardware failures."""

    def __init__(self, job, owner, pubs, shots):
        self._job = job
        self._owner = owner
        self._pubs = pubs
        self._shots = shots

    def __getattr__(self, item):           # delegate everything else to the real job
        return getattr(self._job, item)

    def result(self):
        attempt = 0
        job = self._job
        while True:
            try:
                return job.result()
            except Exception as exc:
                self._owner._log_job_failure(job, exc)
                attempt += 1
                if attempt > self._owner.max_retries:
                    raise
                self._owner._log.warning(
                    "      ↻ retrying failed job (attempt %d/%d) ...",
                    attempt, self._owner.max_retries)
                job = (self._owner._inner.run(self._pubs, shots=self._shots)
                       if self._shots is not None else self._owner._inner.run(self._pubs))
                try:
                    self._owner.job_ids.append(job.job_id())
                except Exception:
                    pass


class LoggingSamplerV2(BaseSamplerV2):
    """
    Transparent proxy around a real V2 sampler that logs every submission.

    It is itself a ``BaseSamplerV2`` so qiskit-machine-learning's
    ``isinstance(sampler, BaseSamplerV2)`` checks pass unchanged.
    """

    def __init__(self, inner, label: str, log_every: int = 1, max_retries: int = 1,
                 transpile_pm=None):
        self._inner = inner
        self._label = label
        self._log_every = max(1, log_every)
        self.max_retries = max(0, max_retries)
        # When set, bind parameters LOCALLY and transpile to ISA before submission,
        # so only PARAMETER-FREE circuits reach IBM. This avoids server-side
        # parameter binding, which fails on incompatible qiskit/runtime versions
        # with "Error 3211: Cannot bind following parameters not present in expression".
        self._transpile_pm = transpile_pm
        self._log = get_logger()

        # running tallies, surfaced in the final report
        self.job_count = 0
        self.pub_count = 0
        self.failed_jobs = 0
        self.job_ids: list[str] = []

    def _prepare_pub(self, pub):
        """Bind parameter values into the circuit, then transpile to ISA.
        Returns a parameter-free single-circuit PUB."""
        import numpy as np
        if hasattr(pub, "circuit"):
            circ = pub.circuit
            vals = getattr(pub, "parameter_values", None)
            if vals is not None and hasattr(vals, "as_array"):
                vals = vals.as_array()
        elif isinstance(pub, (tuple, list)):
            circ = pub[0]
            vals = pub[1] if len(pub) > 1 else None
        else:
            circ, vals = pub, None

        if vals is not None and getattr(circ, "num_parameters", 0) > 0:
            flat = np.asarray(vals, dtype=float).ravel()
            if flat.size == circ.num_parameters:
                circ = circ.assign_parameters(flat)
        isa = self._transpile_pm.run(circ)
        return (isa,)

    def _log_job_failure(self, job, exc):
        self.failed_jobs += 1
        ibm_msg = status = ""
        try:
            ibm_msg = job.error_message() or ""
        except Exception:
            pass
        try:
            status = str(job.status())
        except Exception:
            pass
        jid = ""
        try:
            jid = job.job_id()
        except Exception:
            pass
        self._log.error("      ✗ IBM job FAILED | id=%s | status=%s", jid, status)
        if ibm_msg:
            self._log.error("        IBM error_message: %s", ibm_msg)
        self._log.error("        exception: %s", str(exc)[:200])

    # -- BaseSamplerV2 interface --------------------------------------------
    def run(self, pubs, *, shots=None):
        self.job_count += 1

        # Defeat server-side parameter binding by binding + transpiling locally.
        if self._transpile_pm is not None:
            try:
                pubs = [self._prepare_pub(p) for p in pubs]
            except Exception as exc:
                self._log.error("  [%s] local bind/transpile failed: %s", self._label, str(exc)[:160])
                raise

        n_pubs = len(pubs)
        self.pub_count += n_pubs

        circ = _extract_circuit(pubs[0]) if n_pubs else None
        if circ is not None and (self.job_count % self._log_every == 0):
            try:
                self._log.info(
                    "  [%s] job #%d → %d circuit(s) | qubits=%d depth=%d params=%d | total circuits=%d",
                    self._label, self.job_count, n_pubs,
                    circ.num_qubits, circ.depth(), circ.num_parameters, self.pub_count,
                )
            except Exception:
                self._log.info(
                    "  [%s] job #%d → %d circuit(s) | total circuits=%d",
                    self._label, self.job_count, n_pubs, self.pub_count,
                )

        # forward to the real sampler
        job = self._inner.run(pubs, shots=shots) if shots is not None else self._inner.run(pubs)

        # try to capture the REAL job id / backend (this is the IBM proof)
        try:
            jid = job.job_id()
            self.job_ids.append(jid)
            try:
                backend_name = job.backend().name  # IBM RuntimeJobV2
            except Exception:
                backend_name = type(job).__module__.split(".")[0]
            if self.job_count % self._log_every == 0:
                self._log.info("      ↳ job_id=%s  backend=%s", jid, backend_name)
        except Exception:
            pass  # local primitive jobs may not expose a meaningful id

        # wrap so a failed .result() reveals the real IBM error (+ optional retry)
        return _DiagnosticJob(job, self, pubs, shots)

    # -- reporting ----------------------------------------------------------
    def summary(self) -> dict:
        return {
            "label": self._label,
            "submitted_jobs": self.job_count,
            "submitted_circuits": self.pub_count,
            "failed_jobs": self.failed_jobs,
            "sample_job_ids": self.job_ids[:10],
        }
