"""
comparator.py — Compare classical and quantum ML algorithms.

Generates unified results for all models (classical + quantum)
for fair comparison.
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import List, Dict
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Fixed import
from .config import Config
from .classical_models import get_classical_models, ClassicalModelResult
from .quantum_models import get_quantum_models, QuantumModelResult


@dataclass
class ComparisonResult:
    """Unified comparison result for a classical-quantum pair."""
    classical_model: ClassicalModelResult
    quantum_model: QuantumModelResult
    classical_wins_accuracy: bool
    accuracy_gap: float
    speedup_factor: float  # classical_time / quantum_time


class MLComparator:
    """Compare classical and quantum ML algorithms on equal footing."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.sampler = ctx.sampler
        self.classical_models = get_classical_models()
        self.quantum_models = get_quantum_models(ctx)

        # Map classical to quantum equivalents
        self.mapping = {
            "SVM": "QSVM",
            "kNN": "QkNN",
            "LogReg": "QCL",
            "MLP": "VQC",
        }

    def run_comparison(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """
        Run all classical and quantum models and compare.

        Returns:
            Dictionary with all results and comparisons.
        """
        results = {
            "classical": {},
            "quantum": {},
            "quantum_failed": {},
            "quantum_expected": [m.algorithm_type for m in self.quantum_models],
            "comparisons": [],
            "summary": {},
        }

        print("\n" + "=" * 80)
        print("  CLASSICAL MODELS")
        print("=" * 80)

        # Train classical models
        for model in self.classical_models:
            print(f"\n Training {model.name}...")
            try:
                result = model.train_and_evaluate(X_train, y_train, X_test, y_test)
                results["classical"][model.algorithm_type] = result
                print(f"   Accuracy: {result.accuracy:.4f} | Time: {result.train_time_sec:.2f}s")
            except Exception as e:
                print(f"  ✗ FAILED: {e}")

        print("\n" + "=" * 80)
        print("  QUANTUM MODELS")
        print("=" * 80)

        # Train quantum models
        for model in self.quantum_models:
            print(f"\n Training {model.name}...")
            try:
                result = model.train_and_evaluate(X_train, y_train, X_test, y_test)
                results["quantum"][model.algorithm_type] = result
                print(f"   Accuracy: {result.accuracy:.4f} | Time: {result.train_time_sec:.2f}s")
                if result.convergence_values:
                    print(f"    Converged: {result.convergence_values[-1]:.4f} after {len(result.convergence_values)} iterations")
            except Exception as e:
                results["quantum_failed"][model.algorithm_type] = str(e)
                print(f"   FAILED: {e}")

        # Make the success/failure picture explicit (so 'missing' models in the
        # charts are never a mystery).
        n_total = len(results["quantum_expected"])
        n_ok = len(results["quantum"])
        print("\n" + "-" * 80)
        print(f"  QUANTUM MODELS: {n_ok}/{n_total} succeeded "
              f"→ {list(results['quantum'].keys())}")
        if results["quantum_failed"]:
            print(f"  QUANTUM FAILED: {list(results['quantum_failed'].keys())} "
                  f"(charts will mark these as 'failed')")
            for k, v in results["quantum_failed"].items():
                print(f"     - {k}: {v[:140]}")
        print("-" * 80)

        # Create comparisons
        results["comparisons"] = self._create_comparisons(results)
        results["summary"] = self._create_summary(results)

        return results

    def _create_comparisons(self, results: Dict) -> List[Dict]:
        """Create pairwise comparisons between classical and quantum equivalents."""
        comparisons = []

        for classical_type, quantum_type in self.mapping.items():
            if classical_type in results["classical"] and quantum_type in results["quantum"]:
                classical_res = results["classical"][classical_type]
                quantum_res = results["quantum"][quantum_type]

                comparisons.append({
                    "classical_model": classical_type,
                    "quantum_model": quantum_type,
                    "classical_accuracy": float(classical_res.accuracy),
                    "quantum_accuracy": float(quantum_res.accuracy),
                    "accuracy_gap": abs(classical_res.accuracy - quantum_res.accuracy),
                    "quantum_better": quantum_res.accuracy > classical_res.accuracy,
                    "classical_time_s": float(classical_res.train_time_sec),
                    "quantum_time_s": float(quantum_res.train_time_sec),
                    "time_ratio": quantum_res.train_time_sec / classical_res.train_time_sec if classical_res.train_time_sec > 0 else float("inf"),
                })

        return comparisons

    def _create_summary(self, results: Dict) -> Dict:
        """Create summary statistics."""
        classical_accs = [r.accuracy for r in results["classical"].values()]
        quantum_accs = [r.accuracy for r in results["quantum"].values()]

        return {
            "best_classical": max(results["classical"].values(), key=lambda x: x.accuracy).name if results["classical"] else None,
            "best_classical_accuracy": float(max(classical_accs)) if classical_accs else 0,
            "best_quantum": max(results["quantum"].values(), key=lambda x: x.accuracy).name if results["quantum"] else None,
            "best_quantum_accuracy": float(max(quantum_accs)) if quantum_accs else 0,
            "avg_classical_accuracy": float(np.mean(classical_accs)) if classical_accs else 0,
            "avg_quantum_accuracy": float(np.mean(quantum_accs)) if quantum_accs else 0,
            "total_classical_time": float(sum(r.train_time_sec for r in results["classical"].values())),
            "total_quantum_time": float(sum(r.train_time_sec for r in results["quantum"].values())),
        }

    def print_summary(self, results: Dict):
        """Print formatted comparison summary."""
        summary = results["summary"]

        print("\n" + "=" * 80)
        print("  COMPARISON SUMMARY: CLASSICAL vs QUANTUM")
        print("=" * 80)

        print(f"\n BEST MODELS:")
        print(f"   Classical Best: {summary['best_classical']} → {summary['best_classical_accuracy']:.4f}")
        print(f"   Quantum Best:   {summary['best_quantum']} → {summary['best_quantum_accuracy']:.4f}")

        print(f"\n AVERAGE PERFORMANCE:")
        print(f"   Classical Avg Accuracy: {summary['avg_classical_accuracy']:.4f}")
        print(f"   Quantum Avg Accuracy:   {summary['avg_quantum_accuracy']:.4f}")

        print(f"\n  TRAINING TIME:")
        print(f"   Classical Total: {summary['total_classical_time']:.2f}s")
        print(f"   Quantum Total:   {summary['total_quantum_time']:.2f}s")

        # Variational vs non-variational quantum performance cost
        q = results["quantum"]
        var = [v for v in q.values() if getattr(v, "is_variational", True)]
        non = [v for v in q.values() if not getattr(v, "is_variational", True)]
        if var and non:
            import numpy as _np
            def _cost(v):
                return v.train_time_sec + v.inference_time_ms / 1000.0
            mean_var = _np.mean([_cost(v) for v in var])
            mean_non = _np.mean([_cost(v) for v in non])
            print("\n  VARIATIONAL vs NON-VARIATIONAL (quantum compute cost, train+inference):")
            print(f"   Non-variational ({', '.join(k for k,v in q.items() if not getattr(v,'is_variational',True))}): "
                  f"avg {mean_non:.4g}s")
            print(f"   Variational     ({', '.join(k for k,v in q.items() if getattr(v,'is_variational',True))}): "
                  f"avg {mean_var:.4g}s")
            if mean_non > 0:
                ratio = mean_var / mean_non
                if ratio >= 1:
                    print(f"   → Variational circuits are ~{ratio:.1f}× SLOWER to train "
                          f"(cost grows on real hardware: MAX_ITER × samples jobs)")
                else:
                    print(f"   → Variational circuits are ~{1/ratio:.1f}× faster here "
                          f"(noiseless sim: kernel/fidelity models dominate; on a QPU this inverts)")

        print("\n" + "=" * 80)
        print("  PAIRWISE COMPARISONS")
        print("=" * 80)

        for comp in results["comparisons"]:
            status = " QUANTUM WINS" if comp["quantum_better"] else "✗ QUANTUM LOSES"
            print(f"\n  {comp['classical_model']} → vs → {comp['quantum_model']}")
            print(f"    Classical: {comp['classical_accuracy']:.4f}")
            print(f"    Quantum:   {comp['quantum_accuracy']:.4f}")
            print(f"    Gap:       {comp['accuracy_gap']:.4f} ({status})")
            print(f"    Time ratio: {comp['time_ratio']:.1f}x (quantum/classical)")

    def export_results(self, results: Dict, output_path: Path) -> Path:
        """Export all comparison results to JSON."""

        execution = self.ctx.execution_summary() if hasattr(self, "ctx") else {}
        export_data = {
            "meta": {
                "framework": "Qiskit ML Comparison",
                "dataset": "Iris",
                "num_qubits": Config.NUM_QUBITS,
                "classical_algorithms": list(results["classical"].keys()),
                "quantum_algorithms": list(results["quantum"].keys()),
                "quantum_failed": results.get("quantum_failed", {}),
                "execution": execution,
            },
            "classical": {},
            "quantum": {},
            "comparisons": results["comparisons"],
            "summary": results["summary"],
        }

        # Add classical results
        for k, v in results["classical"].items():
            export_data["classical"][k] = {
                "name": v.name,
                "accuracy": float(v.accuracy),
                "f1_macro": float(v.f1_macro),
                "train_time_sec": float(v.train_time_sec),
                "inference_time_ms": float(v.inference_time_ms),
                "hyperparams": v.hyperparams,
            }

        # Add quantum results
        for k, v in results["quantum"].items():
            export_data["quantum"][k] = {
                "name": v.name,
                "accuracy": float(v.accuracy),
                "f1_macro": float(v.f1_macro),
                "train_time_sec": float(v.train_time_sec),
                "inference_time_ms": float(v.inference_time_ms),
                "num_qubits": v.num_qubits,
                "circuit_depth": v.circuit_depth,
                "num_parameters": v.num_parameters,
                "ran_on_backend": getattr(v, "ran_on_backend", True),
                "is_variational": getattr(v, "is_variational", True),
                "convergence": v.convergence_values if v.convergence_values else [],
            }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        print(f"\n   Results exported to {output_path}")
        return output_path