"""
config.py — Typed configuration loader from .env
"""

import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


def _as_bool(v: str, default: bool = False) -> bool:
    return str(v).lower() in ("1", "true", "yes", "on") if v is not None else default


class Config:
    # Backend
    BACKEND: str = os.getenv("BACKEND", "LOCAL").upper()

    # IBM credentials / connection
    IBM_API_TOKEN: str = os.getenv("IBM_API_TOKEN", "")
    # New IBM Quantum Platform channel (the old "ibm_quantum" is deprecated).
    IBM_CHANNEL: str = os.getenv("IBM_CHANNEL", "ibm_quantum_platform")
    # Leave empty on the new platform; only set if you have a CRN/instance.
    IBM_INSTANCE: str = os.getenv("IBM_INSTANCE", "")
    # Backend name, or "least_busy" to auto-pick the least-busy real QPU.
    IBM_BACKEND_NAME: str = os.getenv("IBM_BACKEND_NAME", "least_busy")
    # Run LOCALLY on Aer using the REAL device's noise model + coupling map
    # (hardware-representative, reproducible, no remote job failures/quota use).
    IBM_NOISE_SIM: bool = _as_bool(os.getenv("IBM_NOISE_SIM", "false"))
    # Auto-retry a failed IBM job this many times (transient hardware errors).
    IBM_MAX_RETRIES: int = int(os.getenv("IBM_MAX_RETRIES", 1))

    # Simulator (LOCAL)
    SIMULATOR_METHOD: str = os.getenv("SIMULATOR_METHOD", "statevector")
    SHOTS: int = int(os.getenv("SHOTS", 1024))

    # Transpilation
    OPTIMIZATION_LEVEL: int = int(os.getenv("OPTIMIZATION_LEVEL", 1))

    # Dataset
    RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", 42))
    TRAIN_SPLIT: float = float(os.getenv("TRAIN_SPLIT", 0.8))

    # VQC hyperparameters
    NUM_QUBITS: int = int(os.getenv("NUM_QUBITS", 2))
    FEATURE_MAP: str = os.getenv("FEATURE_MAP", "ZZFeatureMap")
    FEATURE_MAP_REPS: int = int(os.getenv("FEATURE_MAP_REPS", 2))
    ANSATZ: str = os.getenv("ANSATZ", "RealAmplitudes")
    ANSATZ_REPS: int = int(os.getenv("ANSATZ_REPS", 3))
    OPTIMIZER: str = os.getenv("OPTIMIZER", "COBYLA")
    MAX_ITER: int = int(os.getenv("MAX_ITER", 150))

    # QkNN: run the similarity as a real quantum-fidelity circuit (slow on hardware)
    # or as a classical proxy. Default False = classical proxy (clearly logged).
    QKNN_QUANTUM: bool = _as_bool(os.getenv("QKNN_QUANTUM", "false"))

    # ── Hardware safety guards ──────────────────────────────────
    # Real QPUs are slow and metered. When running on hardware we subsample the
    # data and cap iterations so a demonstration run stays feasible.
    HW_GUARD: bool = _as_bool(os.getenv("HW_GUARD", "true"))
    HW_MAX_TRAIN: int = int(os.getenv("HW_MAX_TRAIN", 12))
    HW_MAX_TEST: int = int(os.getenv("HW_MAX_TEST", 6))
    HW_MAX_ITER: int = int(os.getenv("HW_MAX_ITER", 8))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "quantum_run.log")

    # Output
    OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "outputs"))
    VERBOSE: bool = _as_bool(os.getenv("VERBOSE", "true"), True)

    @classmethod
    def summary(cls) -> str:
        lines = [
            "=" * 60,
            "  Quantum ML Configuration",
            "=" * 60,
            f"  BACKEND           : {cls.BACKEND}",
        ]
        if cls.BACKEND == "IBM":
            lines += [
                f"  IBM_CHANNEL       : {cls.IBM_CHANNEL}",
                f"  IBM_INSTANCE      : {cls.IBM_INSTANCE or '(default / none)'}",
                f"  IBM_BACKEND_NAME  : {cls.IBM_BACKEND_NAME}",
                f"  IBM_NOISE_SIM     : {cls.IBM_NOISE_SIM} (local Aer w/ device noise)",
                f"  IBM_MAX_RETRIES   : {cls.IBM_MAX_RETRIES}",
                f"  SHOTS             : {cls.SHOTS}",
                f"  OPTIMIZATION_LVL  : {cls.OPTIMIZATION_LEVEL}",
                f"  HW_GUARD          : {cls.HW_GUARD} (train<= {cls.HW_MAX_TRAIN}, "
                f"test<= {cls.HW_MAX_TEST}, iter<= {cls.HW_MAX_ITER})",
            ]
        else:
            lines += [
                f"  SIMULATOR_METHOD  : {cls.SIMULATOR_METHOD}",
                f"  SHOTS             : {cls.SHOTS}",
            ]
        lines += [
            f"  NUM_QUBITS        : {cls.NUM_QUBITS}",
            f"  FEATURE_MAP       : {cls.FEATURE_MAP} (reps={cls.FEATURE_MAP_REPS})",
            f"  ANSATZ            : {cls.ANSATZ} (reps={cls.ANSATZ_REPS})",
            f"  OPTIMIZER         : {cls.OPTIMIZER} (max_iter={cls.MAX_ITER})",
            f"  QKNN_QUANTUM      : {cls.QKNN_QUANTUM}",
            f"  TRAIN_SPLIT       : {cls.TRAIN_SPLIT}",
            f"  RANDOM_SEED       : {cls.RANDOM_SEED}",
            "=" * 60,
        ]
        return "\n".join(lines)
