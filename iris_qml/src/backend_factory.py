"""
backend_factory.py — Builds the quantum execution context (sampler + pass manager
+ backend) for Qiskit 1.x / qiskit-machine-learning 0.8.

Returned object: ``QuantumExecutionContext``.

LOCAL / statevector → qiskit.primitives.StatevectorSampler   (exact, noiseless, no PM)
LOCAL / aer         → qiskit_aer.primitives.SamplerV2         (shot-based, ISA PM)
IBM                 → qiskit_ibm_runtime.SamplerV2(mode=backend) on a REAL QPU,
                      with an ISA pass manager generated for that backend.

Every sampler is wrapped in ``LoggingSamplerV2`` so each submission is logged
with the real job id — that is how you verify the quantum part runs on IBM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.exceptions import TranspilerError

from .config import Config
from .logging_utils import get_logger, LoggingSamplerV2

log = get_logger()


def _make_pass_manager(backend, opt_level: int):
    """Build an ISA pass manager, robust across qiskit / qiskit-ibm-runtime versions.

    Newer IBM backends advertise the ``ibm_dynamic_circuits`` translation plugin as
    their default. That plugin is shipped by recent qiskit-ibm-runtime and is NOT
    registered under qiskit 1.x, which raises:
        TranspilerError: Invalid plugin name ibm_dynamic_circuits for stage translation
    Our circuits are static (no control flow), so we fall back to the always-present
    standard 'translator' stage, which transpiles to the device basis just fine.
    """
    try:
        return generate_preset_pass_manager(optimization_level=opt_level, backend=backend)
    except (TranspilerError, KeyError, ValueError) as exc:
        log.warning("Default transpiler stage failed (%s).", str(exc)[:120])
        log.warning("Falling back to translation_method='translator' (static circuits — safe).")
        return generate_preset_pass_manager(
            optimization_level=opt_level, backend=backend, translation_method="translator"
        )


@dataclass
class QuantumExecutionContext:
    """Everything the quantum models need to actually run circuits."""
    sampler: object                       # LoggingSamplerV2 wrapping a real V2 sampler
    pass_manager: Optional[object] = None  # ISA pass manager (None for exact statevector)
    backend: Optional[object] = None       # backend object (IBM/Aer) or None
    label: str = "LOCAL"
    is_hardware: bool = False
    is_simulator: bool = True
    num_qubits_device: int = 0

    # convenience pass-through for the logging summary
    def execution_summary(self) -> dict:
        s = self.sampler.summary() if hasattr(self.sampler, "summary") else {}
        s.update({
            "backend_label": self.label,
            "is_hardware": self.is_hardware,
            "is_simulator": self.is_simulator,
        })
        return s


# ─────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────
def build_execution_context() -> QuantumExecutionContext:
    if Config.BACKEND == "IBM":
        return _build_ibm_context()
    return _build_local_context()


# ─────────────────────────────────────────────────────────────────
# LOCAL
# ─────────────────────────────────────────────────────────────────
def _build_local_context() -> QuantumExecutionContext:
    method = Config.SIMULATOR_METHOD.lower()

    if method == "statevector":
        from qiskit.primitives import StatevectorSampler
        inner = StatevectorSampler()
        log.info("Backend     : StatevectorSampler (exact, noiseless, LOCAL)")
        wrapped = LoggingSamplerV2(inner, label="LOCAL-SV", log_every=20)
        return QuantumExecutionContext(
            sampler=wrapped, pass_manager=None, backend=None,
            label="StatevectorSampler (exact / noiseless)",
            is_hardware=False, is_simulator=True, num_qubits_device=Config.NUM_QUBITS,
        )

    # shot-based Aer simulator → needs an ISA pass manager
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
    aer = AerSimulator()
    inner = AerSamplerV2()
    pm = _make_pass_manager(aer, Config.OPTIMIZATION_LEVEL)
    log.info("Backend     : Aer SamplerV2 (%d shots, LOCAL, ISA transpiled)", Config.SHOTS)
    wrapped = LoggingSamplerV2(inner, label="LOCAL-AER", log_every=20)
    return QuantumExecutionContext(
        sampler=wrapped, pass_manager=pm, backend=aer,
        label=f"Aer SamplerV2 ({Config.SHOTS} shots)",
        is_hardware=False, is_simulator=True, num_qubits_device=Config.NUM_QUBITS,
    )


# ─────────────────────────────────────────────────────────────────
# IBM
# ─────────────────────────────────────────────────────────────────
def _build_ibm_context() -> QuantumExecutionContext:
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as IBMSampler
    except ImportError as exc:
        raise ImportError("qiskit-ibm-runtime is required for BACKEND=IBM.") from exc

    if not Config.IBM_API_TOKEN or Config.IBM_API_TOKEN.startswith("YOUR_"):
        raise ValueError("IBM_API_TOKEN is not set in .env. Add: IBM_API_TOKEN=<your_token>")

    log.info("Connecting to IBM Quantum Platform ...")

    # New IBM Quantum Platform: omit 'instance' unless explicitly provided (the old
    # hub/group/project string 'ibm-q/open/main' is no longer valid there).
    service_kwargs = {"channel": Config.IBM_CHANNEL, "token": Config.IBM_API_TOKEN}
    if Config.IBM_INSTANCE:
        service_kwargs["instance"] = Config.IBM_INSTANCE
    service = QiskitRuntimeService(**service_kwargs)

    # Resolve the backend: a named one, or the least-busy real QPU.
    name = Config.IBM_BACKEND_NAME.strip()
    if not name or name.lower() in ("least_busy", "auto"):
        backend = service.least_busy(operational=True, simulator=False, min_num_qubits=Config.NUM_QUBITS)
        log.info("Selected least-busy QPU: %s", backend.name)
    else:
        backend = service.backend(name)
        log.info("Selected requested backend: %s", backend.name)

    # Describe the device (verification details)
    is_sim = bool(getattr(getattr(backend, "configuration", lambda: None)(), "simulator", False))
    nq = getattr(backend, "num_qubits", 0)
    try:
        status = backend.status()
        pending = getattr(status, "pending_jobs", "?")
        operational = getattr(status, "operational", "?")
    except Exception:
        pending, operational = "?", "?"

    log.info("Device      : %s | qubits=%s | simulator=%s | operational=%s | pending_jobs=%s",
             backend.name, nq, is_sim, operational, pending)

    # ISA pass manager for THIS backend — mandatory for V2 primitives on hardware.
    pm = _make_pass_manager(backend, Config.OPTIMIZATION_LEVEL)
    log.info("Pass manager: ISA transpilation for %s (opt_level=%d)", backend.name, Config.OPTIMIZATION_LEVEL)

    # ── Option: run LOCALLY on Aer using the real device's noise model ──────
    # Hardware-representative + reproducible, with no remote job failures or
    # QPU-time cost. We translate to the device BASIS GATES only (no 156-qubit
    # layout), so the simulation stays at NUM_QUBITS and never blows up memory.
    if Config.IBM_NOISE_SIM:
        from qiskit_aer.noise import NoiseModel
        from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
        used_pm = pm
        try:
            nm = NoiseModel.from_backend(backend)
            basis = list(backend.configuration().basis_gates)
            used_pm = generate_preset_pass_manager(
                optimization_level=Config.OPTIMIZATION_LEVEL, basis_gates=basis)
            inner = AerSamplerV2(options={"backend_options": {"noise_model": nm}})
            log.info("Sampler     : LOCAL Aer w/ %s noise model | basis=%s | width-safe (no 156q layout).",
                     backend.name, basis)
        except Exception as exc:
            log.warning("Noise-sim setup failed (%s); using noiseless Aer + no transpile.", str(exc)[:100])
            inner = AerSamplerV2()
            used_pm = None
        wrapped = LoggingSamplerV2(inner, label=f"IBM-NOISE:{backend.name}", log_every=20, max_retries=0)
        return QuantumExecutionContext(
            sampler=wrapped, pass_manager=used_pm, backend=backend,
            label=f"IBM noise-sim ({backend.name} model, LOCAL Aer)",
            is_hardware=False, is_simulator=True, num_qubits_device=nq,
        )

    inner = IBMSampler(mode=backend)
    try:
        inner.options.default_shots = Config.SHOTS
    except Exception:
        pass

    # Wrapper binds parameters + transpiles LOCALLY, sending parameter-free ISA
    # circuits to IBM (defeats Error 3211 server-side parameter-binding failures).
    wrapped = LoggingSamplerV2(inner, label=f"IBM:{backend.name}", log_every=1,
                               max_retries=Config.IBM_MAX_RETRIES, transpile_pm=pm)
    log.info("Sampler     : IBM Runtime SamplerV2 (mode=%s, shots=%d) — local bind+transpile, job ids logged.",
             backend.name, Config.SHOTS)

    return QuantumExecutionContext(
        # ctx.pass_manager is None here on purpose: the WRAPPER transpiles, so the
        # qiskit-machine-learning models pass abstract parametric circuits through.
        sampler=wrapped, pass_manager=None, backend=backend,
        label=f"IBM Quantum ({backend.name})",
        is_hardware=not is_sim, is_simulator=is_sim, num_qubits_device=nq,
    )


# ─────────────────────────────────────────────────────────────────
# Backwards-compatible helpers (kept for any external callers)
# ─────────────────────────────────────────────────────────────────
def build_sampler():
    return build_execution_context().sampler


def backend_label() -> str:
    if Config.BACKEND == "IBM":
        return f"IBM Quantum ({Config.IBM_BACKEND_NAME})"
    if Config.SIMULATOR_METHOD == "statevector":
        return "StatevectorSampler (exact / noiseless)"
    return f"Aer SamplerV2 ({Config.SHOTS} shots)"
