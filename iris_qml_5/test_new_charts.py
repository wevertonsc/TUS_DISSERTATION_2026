"""
Standalone validation for the new charts in visualizer_compare.py.
Reconstructs the 2026-06-14 hardware run (ibm_fez) from the application logs
as mock result objects, then runs create_comparison_charts end-to-end.
No Qiskit / QPU needed — this exercises only the plotting code path.
"""
import sys, types
from pathlib import Path
import numpy as np

# stub out the package's heavy .config import so we can import the visualizer alone
pkg = types.ModuleType("src"); pkg.__path__ = ["src"]
sys.modules["src"] = pkg
cfgmod = types.ModuleType("src.config")
class Config:  # minimal
    NUM_QUBITS = 2
cfgmod.Config = Config
sys.modules["src.config"] = cfgmod

from src.visualizer_compare import create_comparison_charts


class R:
    """Mock model result with the attributes the charts read."""
    def __init__(self, name, acc, train_s, infer_ms, depth=0, conv=None,
                 is_var=True, ran=True):
        self.name = name
        self.accuracy = acc
        self.train_time_sec = train_s
        self.inference_time_ms = infer_ms
        self.circuit_depth = depth
        self.num_qubits = 2
        self.num_parameters = 0
        self.convergence_values = conv or []
        self.is_variational = is_var
        self.ran_on_backend = ran
        # derive plausible secondary metrics around accuracy (bounded 0..1)
        self.f1_macro = max(0.0, min(1.0, acc - 0.02))
        self.precision_macro = max(0.0, min(1.0, acc + 0.01))
        self.recall_macro = max(0.0, min(1.0, acc - 0.01))
        # 3-class confusion matrix consistent-ish with accuracy (6 test samples)
        self.confusion_mat = np.array([[2, 0, 0], [0, 1, 1], [0, 1, 1]])
        self.y_true = np.array([0, 0, 1, 1, 2, 2])
        self.y_pred = np.array([0, 0, 1, 2, 1, 2])
        self.classification_rep = ""
        self.hyperparams = {}
        self.algorithm_type = name


# ---- numbers taken directly from console_run.log (2026-06-14 23:22 run) ----
classical = {
    "SVM":    R("SVM (RBF Kernel)",        0.6667, 0.00,  0.2,  is_var=False, ran=False),
    "kNN":    R("k-NN (k=5)",              0.6667, 0.00,  0.2,  is_var=False, ran=False),
    "LogReg": R("Logistic Regression",     1.0000, 0.00,  0.2,  is_var=False, ran=False),
    "MLP":    R("MLP (64-32)",             0.6667, 0.01,  0.3,  is_var=False, ran=False),
}
quantum = {
    "QSVM":   R("QSVM (Quantum Kernel)",   0.3333, 28.47, 29367.0, depth=18, is_var=False),
    "QkNN":   R("QkNN (Quantum k=3)",      0.6667, 0.00,  0.4,     depth=0,  is_var=False, ran=False),
    "QCL":    R("QCL (Quantum Circuit Learning)", 0.1667, 239.15, 8417.6, depth=22,
                conv=[14.8, 13.9, 13.1, 12.7, 12.3, 12.0, 11.8, 11.5961]),
    "VQC":    R("VQC (Variational Quantum Classifier)", 0.5000, 77.70, 7609.2, depth=22,
                conv=[14.6, 13.7, 13.0, 12.6, 12.2, 11.9, 11.7, 11.6059]),
    "QNN":    R("QNN (Quantum Neural Network)", 0.1667, 38.95, 7855.1, depth=23,
                conv=[1.39, 1.21, 1.10, 1.0618]),
    "QFC":    R("QFC (Quantum Fidelity)",  0.3333, 0.00,  56235.9, depth=18, is_var=False),
    "QFE-LR": R("QFE-LR (Embedding+Linear)", 0.3333, 9.41, 8187.4, depth=16, is_var=False),
}

comparisons = [
    {"classical_model": "SVM",    "quantum_model": "QSVM", "accuracy_gap": 0.3333,
     "quantum_better": False, "time_ratio": 21202.1},
    {"classical_model": "kNN",    "quantum_model": "QkNN", "accuracy_gap": 0.0000,
     "quantum_better": False, "time_ratio": 0.0},
    {"classical_model": "LogReg", "quantum_model": "QCL",  "accuracy_gap": 0.8333,
     "quantum_better": False, "time_ratio": 65928.2},
    {"classical_model": "MLP",    "quantum_model": "VQC",  "accuracy_gap": 0.1667,
     "quantum_better": False, "time_ratio": 9161.9},
]

# reconstruct per-job records from the 33 job timestamps in the log (HH:MM:SS)
job_log = [  # (job_idx, "mm:ss" offset from 23:22:21, n_circuits)
    (1,"00:10",66),(2,"00:38",72),(3,"01:08",12),(4,"01:17",12),(5,"01:26",12),
    (6,"01:36",12),(7,"02:11",12),(8,"04:29",12),(9,"04:48",12),(10,"04:57",12),
    (11,"05:07",6),(12,"05:15",12),(13,"05:25",12),(14,"05:35",12),(15,"05:45",12),
    (16,"05:55",12),(17,"06:05",12),(18,"06:14",12),(19,"06:23",12),(20,"06:33",6),
    (21,"06:40",12),(22,"06:50",12),(23,"06:60",12),(24,"07:10",12),(25,"07:19",6),
    (26,"07:27",12),(27,"07:37",12),(28,"07:46",12),(29,"07:56",12),(30,"08:05",12),
    (31,"08:15",12),(32,"08:24",12),(33,"08:33",6),
]
def _sec(ms):
    m, s = ms.split(":"); return int(m) * 60 + int(s)
records, cum = [], 0
for idx, off, n in job_log:
    cum += n
    records.append({"idx": idx, "t": float(_sec(off)), "n_circuits": n,
                    "cumulative_circuits": cum, "depth": 20, "qubits": 156})

execution = {
    "label": "IBM-ibm_fez", "backend_label": "IBM Quantum (ibm_fez)",
    "is_hardware": True, "is_simulator": False,
    "submitted_jobs": 33, "failed_jobs": 0, "submitted_circuits": 486,
    "sample_job_ids": ["d8nija832u0s73fcge0g"], "job_records": records,
}

results = {
    "classical": classical, "quantum": quantum,
    "quantum_expected": list(quantum.keys()), "quantum_failed": {},
    "comparisons": comparisons, "summary": {},
}

out = Path("/tmp/iris_qml/_test_charts")
saved = create_comparison_charts(results, out, execution=execution)
print(f"\nGenerated {len(saved)} files:")
for p in sorted(saved):
    print("  ", Path(p).name)
