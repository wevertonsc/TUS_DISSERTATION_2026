"""
quantum_models.py — Quantum machine learning algorithms for comparison.

All variational / kernel models now route their circuits through the execution
context's sampler AND pass manager, so when BACKEND=IBM the work runs on a real
QPU (and every submission is logged with its job id by LoggingSamplerV2).
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import List
import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from sklearn.exceptions import ConvergenceWarning

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, EfficientSU2

from qiskit_machine_learning.algorithms import VQC, QSVC, NeuralNetworkClassifier
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA

from .config import Config
from .logging_utils import get_logger

log = get_logger()
warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass
class QuantumModelResult:
    name: str
    algorithm_type: str
    accuracy: float
    f1_macro: float
    precision_macro: float
    recall_macro: float
    train_time_sec: float
    inference_time_ms: float
    confusion_mat: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    classification_rep: str
    num_qubits: int
    circuit_depth: int
    num_parameters: int
    convergence_values: List[float]
    ran_on_backend: bool = True
    is_variational: bool = True


class QuantumModelWrapper:
    def __init__(self, model, name, algorithm_type, feature_map=None, ansatz=None,
                 sampler=None, ran_on_backend=True, is_variational=True):
        self.model = model
        self.name = name
        self.algorithm_type = algorithm_type
        self.feature_map = feature_map
        self.ansatz = ansatz
        self.sampler = sampler
        self.ran_on_backend = ran_on_backend
        self.is_variational = is_variational
        self._convergence = []

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        where = "QPU/backend" if self.ran_on_backend else "CLASSICAL (no backend)"
        log.info("→ Training %s  [executes on: %s]", self.name, where)

        # Pull convergence out of a VQC/NN classifier callback if present
        if hasattr(self.model, "_fit_result") is False and hasattr(self.model, "callback"):
            pass  # callback already wired at construction

        t0 = time.perf_counter()
        self.model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        t_inf = time.perf_counter()
        y_pred = self.model.predict(X_test)
        inference_time_ms = (time.perf_counter() - t_inf) * 1000

        y_true = y_test.astype(int)
        y_pred = np.asarray(y_pred).astype(int)

        depth = params = 0
        if self.feature_map is not None and self.ansatz is not None:
            full_circuit = self.feature_map.compose(self.ansatz)
            depth = full_circuit.depth()
            params = self.ansatz.num_parameters

        log.info("   done %s | acc=%.4f | train=%.2fs | inference=%.1fms",
                 self.name, accuracy_score(y_true, y_pred), train_time, inference_time_ms)

        return QuantumModelResult(
            name=self.name,
            algorithm_type=self.algorithm_type,
            accuracy=accuracy_score(y_true, y_pred),
            f1_macro=f1_score(y_true, y_pred, average="macro", zero_division=0),
            precision_macro=precision_score(y_true, y_pred, average="macro", zero_division=0),
            recall_macro=recall_score(y_true, y_pred, average="macro", zero_division=0),
            train_time_sec=train_time,
            inference_time_ms=inference_time_ms,
            confusion_mat=confusion_matrix(y_true, y_pred, labels=[0, 1, 2]),
            y_true=y_true,
            y_pred=y_pred,
            classification_rep=classification_report(
                y_true, y_pred,
                labels=[0, 1, 2],
                target_names=["Setosa", "Versicolor", "Virginica"],
                zero_division=0,
            ),
            num_qubits=Config.NUM_QUBITS,
            circuit_depth=depth,
            num_parameters=params,
            convergence_values=self._convergence,
            ran_on_backend=self.ran_on_backend,
            is_variational=self.is_variational,
        )


# ─────────────────────────────────────────────────────────────────
# Quantum QkNN that genuinely uses fidelity circuits (optional, slow)
# ─────────────────────────────────────────────────────────────────
class QuantumFidelityKNN:
    """k-NN where similarity = quantum state fidelity computed via ComputeUncompute.

    Every predict() batches all (test, train) pairs into fidelity circuits that go
    through the supplied sampler — so on IBM these are real QPU submissions.
    """

    def __init__(self, feature_map, fidelity, n_neighbors=3):
        self.fm = feature_map
        self.fidelity = fidelity
        self.k = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        preds = []
        n_train = len(self.X_train)
        for x in X:
            # one fidelity job: x against every training point
            circuits_1 = [self.fm] * n_train
            circuits_2 = [self.fm] * n_train
            values_1 = [x] * n_train
            values_2 = list(self.X_train)
            job = self.fidelity.run(circuits_1, circuits_2, values_1, values_2)
            sims = np.asarray(job.result().fidelities)
            top = np.argsort(-sims)[: self.k]
            labels = list(self.y_train[top])
            preds.append(max(set(labels), key=labels.count))
        return np.array(preds)


class ClassicalProxyKNN:
    """Gaussian-kernel k-NN (NOT quantum) — used when QKNN_QUANTUM=false."""

    def __init__(self, n_neighbors=3):
        self.k = n_neighbors

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        return self

    def predict(self, X):
        preds = []
        for x in np.asarray(X):
            sims = [np.exp(-np.linalg.norm(x - xt) ** 2 / 2) for xt in self.X_train]
            top = np.argsort(sims)[::-1][: self.k]
            labels = list(self.y_train[top])
            preds.append(max(set(labels), key=labels.count))
        return np.array(preds)


# ─────────────────────────────────────────────────────────────────
# NON-VARIATIONAL quantum classifiers (no trainable ansatz, no COBYLA loop).
# These exist to measure the performance cost of the variational approach:
# their circuit count is O(n) instead of O(MAX_ITER × n).
# ─────────────────────────────────────────────────────────────────
class QuantumFidelityClassifier:
    """Non-variational. Classifies a test point by the class whose training
    members have the highest mean quantum-state fidelity to it. Uses a FIXED
    feature map (data encoding only) — no trainable parameters."""

    def __init__(self, feature_map, fidelity):
        self.fm = feature_map
        self.fidelity = fidelity

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        self.classes_ = np.unique(self.y_train)
        return self

    def predict(self, X):
        X = np.asarray(X)
        n_train = len(self.X_train)
        preds = []
        for x in X:
            job = self.fidelity.run([self.fm] * n_train, [self.fm] * n_train,
                                    [x] * n_train, list(self.X_train))
            sims = np.asarray(job.result().fidelities)
            # mean fidelity per class → argmax (nearest class in Hilbert space)
            scores = {c: sims[self.y_train == c].mean() for c in self.classes_}
            preds.append(max(scores, key=scores.get))
        return np.array(preds)


class QuantumEmbeddingLinear:
    """Non-variational. A FIXED feature-map circuit produces measurement-probability
    features (a quantum embedding); a plain classical Logistic Regression is the head.
    The quantum circuit is never trained — only the classical linear layer is.
    This is the non-variational counterpart of the QNN."""

    def __init__(self, feature_map, sampler, shots=1024):
        from sklearn.linear_model import LogisticRegression
        self.fm = feature_map.copy()
        self.fm.measure_all()
        self.sampler = sampler
        self.shots = shots
        self.n = feature_map.num_qubits
        self.head = LogisticRegression(max_iter=2000)

    def _embed(self, X):
        """Run the fixed feature map for every sample, return probability vectors."""
        X = np.asarray(X)
        pubs = [(self.fm, list(map(float, x))) for x in X]
        job = self.sampler.run(pubs, shots=self.shots)
        result = job.result()
        dim = 2 ** self.n
        feats = np.zeros((len(X), dim))
        for i in range(len(X)):
            data = result[i].data
            bitarray = next(iter(data.values())) if hasattr(data, "values") else list(vars(data).values())[0]
            counts = bitarray.get_counts()
            total = sum(counts.values()) or 1
            for bitstr, c in counts.items():
                feats[i, int(bitstr.replace(" ", ""), 2)] = c / total
        return feats

    def fit(self, X, y):
        self.head.fit(self._embed(X), np.asarray(y))
        return self

    def predict(self, X):
        return self.head.predict(self._embed(X))


# ─────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────
def get_quantum_models(ctx) -> list[QuantumModelWrapper]:
    """Instantiate all quantum models, wired to the execution context (ctx)."""
    sampler = ctx.sampler
    pm = ctx.pass_manager
    models: list[QuantumModelWrapper] = []

    num_qubits = Config.NUM_QUBITS
    fm_reps = Config.FEATURE_MAP_REPS
    ans_reps = Config.ANSATZ_REPS

    feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=fm_reps)
    ansatz = RealAmplitudes(num_qubits=num_qubits, reps=ans_reps, entanglement="full")

    # Shared fidelity bound to the (possibly IBM) sampler + pass manager
    fidelity = ComputeUncompute(sampler=sampler, pass_manager=pm)

    # 1. QSVM — now routes through the sampler via an explicit fidelity (FIX)
    try:
        quantum_kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
        qsvc = QSVC(quantum_kernel=quantum_kernel)
        models.append(QuantumModelWrapper(
            model=qsvc, name="QSVM (Quantum Kernel)", algorithm_type="QSVM",
            feature_map=feature_map, sampler=sampler, ran_on_backend=True, is_variational=False,
        ))
        log.info("  ✓ QSVM created (FidelityQuantumKernel ← sampler+pass_manager)")
    except Exception as e:
        log.warning("  ⚠️ Could not create QSVM: %s", e)

    # 2. QkNN — quantum fidelity (optional) OR clearly-labelled classical proxy
    try:
        if Config.QKNN_QUANTUM:
            qknn_model = QuantumFidelityKNN(feature_map=feature_map, fidelity=fidelity, n_neighbors=3)
            ran = True
            log.info("  ✓ QkNN created (TRUE quantum fidelity via sampler)")
        else:
            qknn_model = ClassicalProxyKNN(n_neighbors=3)
            ran = False
            log.info("  ✓ QkNN created (classical Gaussian proxy — set QKNN_QUANTUM=true for real fidelity)")
        models.append(QuantumModelWrapper(
            model=qknn_model, name="QkNN (Quantum k=3)", algorithm_type="QkNN",
            feature_map=feature_map, sampler=sampler, ran_on_backend=ran, is_variational=False,
        ))
    except Exception as e:
        log.warning("  ⚠️ Could not create QkNN: %s", e)

    # 3 & 4. QCL and VQC — pass_manager + callback wired at construction (FIX)
    try:
        qcl_conv, vqc_conv = [], []
        qcl = VQC(
            feature_map=feature_map, ansatz=ansatz,
            optimizer=COBYLA(maxiter=Config.MAX_ITER),
            sampler=sampler, pass_manager=pm,
            callback=lambda w, v: qcl_conv.append(float(v)),
        )
        w_qcl = QuantumModelWrapper(
            model=qcl, name="QCL (Quantum Circuit Learning)", algorithm_type="QCL",
            feature_map=feature_map, ansatz=ansatz, sampler=sampler, ran_on_backend=True, is_variational=True,
        )
        w_qcl._convergence = qcl_conv
        models.append(w_qcl)
        log.info("  ✓ QCL created (VQC + sampler + pass_manager)")

        vqc = VQC(
            feature_map=feature_map, ansatz=ansatz,
            optimizer=COBYLA(maxiter=Config.MAX_ITER),
            sampler=sampler, pass_manager=pm,
            callback=lambda w, v: vqc_conv.append(float(v)),
        )
        w_vqc = QuantumModelWrapper(
            model=vqc, name="VQC (Variational Quantum Classifier)", algorithm_type="VQC",
            feature_map=feature_map, ansatz=ansatz, sampler=sampler, ran_on_backend=True, is_variational=True,
        )
        w_vqc._convergence = vqc_conv
        models.append(w_vqc)
        log.info("  ✓ VQC created (VQC + sampler + pass_manager)")
    except Exception as e:
        log.warning("  ⚠️ Could not create VQC/QCL: %s", e)

    # 5. QNN — fixed import + real feature map inputs + pass_manager (FIX)
    try:
        qnn_fm = ZZFeatureMap(feature_dimension=num_qubits, reps=fm_reps)
        qnn_ansatz = EfficientSU2(num_qubits=num_qubits, reps=ans_reps, entanglement="full")
        qnn_circuit = qnn_fm.compose(qnn_ansatz)

        def parity(x):
            return f"{x:b}".count("1") % 2

        qnn = SamplerQNN(
            circuit=qnn_circuit,
            sampler=sampler,
            input_params=list(qnn_fm.parameters),     # data goes in (FIX: was [])
            weight_params=list(qnn_ansatz.parameters),
            interpret=parity,
            output_shape=2,
            pass_manager=pm,                            # ISA for hardware (FIX)
        )
        qnn_conv = []
        qnn_classifier = NeuralNetworkClassifier(
            neural_network=qnn,
            optimizer=COBYLA(maxiter=max(4, Config.MAX_ITER // 2)),
            callback=lambda w, v: qnn_conv.append(float(v)),
        )
        w_qnn = QuantumModelWrapper(
            model=qnn_classifier, name="QNN (Quantum Neural Network)", algorithm_type="QNN",
            feature_map=qnn_fm, ansatz=qnn_ansatz, sampler=sampler, ran_on_backend=True, is_variational=True,
        )
        w_qnn._convergence = qnn_conv
        models.append(w_qnn)
        log.info("  ✓ QNN created (SamplerQNN + sampler + pass_manager)")
    except Exception as e:
        log.warning("  ⚠️ Could not create QNN: %s", e)

    # 6. QFC — Quantum Fidelity Classifier (NON-variational, fixed feature map)
    try:
        qfc = QuantumFidelityClassifier(feature_map=feature_map, fidelity=fidelity)
        models.append(QuantumModelWrapper(
            model=qfc, name="QFC (Quantum Fidelity, non-variational)", algorithm_type="QFC",
            feature_map=feature_map, sampler=sampler, ran_on_backend=True, is_variational=False,
        ))
        log.info("  ✓ QFC created (fidelity classifier — NON-variational)")
    except Exception as e:
        log.warning("  ⚠️ Could not create QFC: %s", e)

    # 7. QFE-LR — Quantum Feature Embedding + Logistic Regression (NON-variational)
    #    Fixed quantum embedding; only the classical linear head is trained.
    try:
        qfe_fm = ZZFeatureMap(feature_dimension=num_qubits, reps=fm_reps)
        qfe = QuantumEmbeddingLinear(feature_map=qfe_fm, sampler=sampler, shots=Config.SHOTS)
        models.append(QuantumModelWrapper(
            model=qfe, name="QFE-LR (Quantum Embedding + Linear, non-variational)",
            algorithm_type="QFE-LR",
            feature_map=qfe_fm, sampler=sampler, ran_on_backend=True, is_variational=False,
        ))
        log.info("  ✓ QFE-LR created (fixed quantum embedding + classical linear — NON-variational)")
    except Exception as e:
        log.warning("  ⚠️ Could not create QFE-LR: %s", e)

    return models
