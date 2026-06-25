"""
circuits.py — Variational Quantum Circuit components for Qiskit 2.x.

Uses the modern functional API to avoid deprecation warnings.

Feature maps  : ZZFeatureMap, PauliFeatureMap
Ansatze       : RealAmplitudes, EfficientSU2, TwoLocal
Optimizers    : COBYLA, SPSA, L_BFGS_B, ADAM
"""

from __future__ import annotations
from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    ZZFeatureMap,
    PauliFeatureMap,
    RealAmplitudes,
    EfficientSU2,
    TwoLocal,
)
from qiskit_machine_learning.optimizers import COBYLA, SPSA, L_BFGS_B, ADAM

from .config import Config


# ─────────────────────────────────────────────────────────────────
# Feature Maps
# ─────────────────────────────────────────────────────────────────

def get_feature_map(name: str, num_qubits: int, reps: int) -> QuantumCircuit:
    if name == "ZZFeatureMap":
        return ZZFeatureMap(feature_dimension=num_qubits, reps=reps)
    if name == "PauliFeatureMap":
        return PauliFeatureMap(
            feature_dimension=num_qubits, reps=reps, paulis=["Z", "ZZ"]
        )
    raise ValueError(f"Unknown feature map: {name}")


# ─────────────────────────────────────────────────────────────────
# Ansatze
# ─────────────────────────────────────────────────────────────────

def get_ansatz(name: str, num_qubits: int, reps: int) -> QuantumCircuit:
    if name == "RealAmplitudes":
        return RealAmplitudes(num_qubits=num_qubits, reps=reps, entanglement="full")
    if name == "EfficientSU2":
        return EfficientSU2(num_qubits=num_qubits, reps=reps, entanglement="full")
    if name == "TwoLocal":
        return TwoLocal(
            num_qubits=num_qubits,
            rotation_blocks=["ry", "rz"],
            entanglement_blocks="cx",
            reps=reps,
        )
    raise ValueError(f"Unknown ansatz: {name}")


# ─────────────────────────────────────────────────────────────────
# Optimizers
# ─────────────────────────────────────────────────────────────────

def get_optimizer(name: str, max_iter: int):
    opts = {
        "COBYLA":   lambda: COBYLA(maxiter=max_iter),
        "SPSA":     lambda: SPSA(maxiter=max_iter),
        "L_BFGS_B": lambda: L_BFGS_B(maxiter=max_iter),
        "ADAM":     lambda: ADAM(maxiter=max_iter, lr=0.01),
    }
    if name not in opts:
        raise ValueError(f"Unknown optimizer: {name}")
    return opts[name]()


# ─────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    name: str
    feature_map_name: str
    ansatz_name: str
    optimizer_name: str
    num_qubits: int
    fm_reps: int
    ansatz_reps: int
    max_iter: int

    @property
    def feature_map(self) -> QuantumCircuit:
        return get_feature_map(self.feature_map_name, self.num_qubits, self.fm_reps)

    @property
    def ansatz(self) -> QuantumCircuit:
        return get_ansatz(self.ansatz_name, self.num_qubits, self.ansatz_reps)

    @property
    def optimizer(self):
        return get_optimizer(self.optimizer_name, self.max_iter)

    @property
    def total_params(self) -> int:
        return self.ansatz.num_parameters

    @property
    def total_depth(self) -> int:
        return self.feature_map.compose(self.ansatz).depth()


def get_all_model_specs(num_qubits: int | None = None) -> list[ModelSpec]:
    """12 benchmark combinations: 2 FMs × 3 ansatze × 3 optimizers."""
    nq      = num_qubits or Config.NUM_QUBITS
    mi      = Config.MAX_ITER
    fm_reps = Config.FEATURE_MAP_REPS
    an_reps = Config.ANSATZ_REPS

    specs = []
    for fm in ["ZZFeatureMap", "PauliFeatureMap"]:
        for ans in ["RealAmplitudes", "EfficientSU2", "TwoLocal"]:
            for opt in ["COBYLA", "SPSA", "L_BFGS_B"]:
                short_fm  = fm.replace("FeatureMap", "FM")
                short_ans = ans.replace("Amplitudes", "Amp")
                specs.append(ModelSpec(
                    name=f"{short_fm}+{short_ans}+{opt}",
                    feature_map_name=fm, ansatz_name=ans, optimizer_name=opt,
                    num_qubits=nq, fm_reps=fm_reps, ansatz_reps=an_reps, max_iter=mi,
                ))
    return specs


def get_default_model_spec() -> ModelSpec:
    return ModelSpec(
        name=f"{Config.FEATURE_MAP}+{Config.ANSATZ}+{Config.OPTIMIZER}",
        feature_map_name=Config.FEATURE_MAP,
        ansatz_name=Config.ANSATZ,
        optimizer_name=Config.OPTIMIZER,
        num_qubits=Config.NUM_QUBITS,
        fm_reps=Config.FEATURE_MAP_REPS,
        ansatz_reps=Config.ANSATZ_REPS,
        max_iter=Config.MAX_ITER,
    )