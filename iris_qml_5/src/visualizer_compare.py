"""
visualizer_compare.py — Visualize comparison between classical and quantum models.
"""

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# Fixed import
from .config import Config

# Style settings
PALETTE = {
    "classical": "#4C72B0",
    "quantum": "#DD8452",
    "SVM": "#55A868",
    "kNN": "#C44E52",
    "LogReg": "#8172B2",
    "MLP": "#937860",
    "QSVM": "#DA8BC3",
    "QkNN": "#8C8C8C",
    "QCL": "#CCB974",
    "VQC": "#64B5CD",
    "QNN": "#E74C3C",
    "QFC": "#17A589",
    "QFE-LR": "#2E86C1",
}


def create_comparison_charts(results: dict, output_dir: Path, execution: dict | None = None) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # make the execution summary (job records, reliability) available to helpers
    if execution is not None:
        results = {**results, "execution": execution}

    saved = []

    if not results.get("classical") and not results.get("quantum"):
        print("  No results to visualize")
        return saved

    saved.append(_plot_accuracy_comparison(results, output_dir))
    saved.append(_plot_time_comparison(results, output_dir))
    saved.append(_plot_accuracy_vs_time(results, output_dir))
    saved.append(_plot_variational_comparison(results, output_dir))

    if results.get("comparisons"):
        saved.append(_plot_accuracy_gap(results, output_dir))

    if results.get("classical") and results.get("quantum"):
        saved.append(_plot_metrics_radar(results, output_dir))
        saved.append(_plot_confusion_matrices(results, output_dir))

    # ---- new charts (from the application logs / in-memory results) --------
    if results.get("quantum"):
        saved.append(_plot_inference_time(results, output_dir))
        saved.append(_plot_circuit_depth(results, output_dir))
        saved.append(_plot_accuracy_vs_depth(results, output_dir))
        conv = _plot_convergence_curves(results, output_dir)
        if conv:
            saved.append(conv)

    if results.get("comparisons"):
        saved.append(_plot_time_ratio_pairs(results, output_dir))

    if results.get("classical") and results.get("quantum"):
        saved.append(_plot_metrics_heatmap(results, output_dir))

    # execution-level charts (need per-job records from LoggingSamplerV2.summary)
    execu = results.get("execution") or {}
    if execu.get("job_records"):
        saved.append(_plot_execution_timeline(results, output_dir))
        saved.append(_plot_circuits_cumulative(results, output_dir))
    if execu.get("submitted_jobs") is not None:
        saved.append(_plot_execution_reliability(results, output_dir))

    saved = [p for p in saved if p is not None]
    print(f"\n  ✓ Generated {len(saved)} comparison charts")
    return saved


# classical → quantum equivalence (mirrors comparator.mapping) + bonus QNN
PAIRS = [("SVM", "QSVM"), ("kNN", "QkNN"), ("LogReg", "QCL"), ("MLP", "VQC")]
QUANTUM_ONLY = ["QNN", "QFC", "QFE-LR"]


def _pair_layout(results: dict):
    """Build ordered (label, classical_type, quantum_type) slots covering every
    expected quantum model, so failed ones still get a visible slot."""
    expected_q = set(results.get("quantum_expected", []) or
                     list(results.get("quantum", {}).keys()))
    slots = []
    for c, q in PAIRS:
        if q in expected_q or q in results.get("quantum", {}):
            slots.append((f"{c} ↔ {q}", c, q))
    for q in QUANTUM_ONLY:
        if q in expected_q or q in results.get("quantum", {}):
            slots.append((f"— ↔ {q}", None, q))
    return slots


def _annotate_failures(ax, slots, x, results, y_at):
    """Put a red '✗ failed' marker where an expected quantum model is missing."""
    failed = results.get("quantum_failed", {})
    present = results.get("quantum", {})
    for i, (_, _c, q) in enumerate(slots):
        if q not in present:
            tag = "✗ failed" if q in failed else "✗ N/A"
            ax.annotate(tag, (x[i] + 0.18, y_at), ha="center", va="bottom",
                        fontsize=8, color="#C0392B", rotation=90)


def _plot_accuracy_comparison(results: dict, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6))

    classical = {k: v.accuracy for k, v in results["classical"].items()}
    quantum = {k: v.accuracy for k, v in results["quantum"].items()}

    slots = _pair_layout(results)
    x = np.arange(len(slots))
    width = 0.36

    classical_accs = [classical.get(c, 0) if c else 0 for _, c, _q in slots]
    quantum_accs = [quantum.get(q, 0) for _, _c, q in slots]

    ax.bar(x - width/2, classical_accs, width, label="Classical", color=PALETTE["classical"])
    ax.bar(x + width/2, quantum_accs, width, label="Quantum", color=PALETTE["quantum"])
    _annotate_failures(ax, slots, x, results, y_at=0.02)

    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Classical ↔ Quantum pair")
    ax.set_title("Classical vs Quantum ML: Accuracy Comparison (Iris Dataset)")
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in slots], rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    path = output_dir / "01_accuracy_comparison.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_time_comparison(results: dict, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6))

    classical_times = {k: v.train_time_sec for k, v in results["classical"].items()}
    quantum_times = {k: v.train_time_sec for k, v in results["quantum"].items()}

    slots = _pair_layout(results)
    x = np.arange(len(slots))
    width = 0.36

    classical_vals = [classical_times.get(c, 0) if c else 0 for _, c, _q in slots]
    quantum_vals = [quantum_times.get(q, 0) for _, _c, q in slots]

    ax.bar(x - width/2, classical_vals, width, label="Classical", color=PALETTE["classical"])
    ax.bar(x + width/2, quantum_vals, width, label="Quantum", color=PALETTE["quantum"])

    ax.set_ylabel("Training Time (seconds)")
    ax.set_xlabel("Classical ↔ Quantum pair")
    ax.set_title("Classical vs Quantum ML: Training Time Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in slots], rotation=15, ha="right")
    ax.legend()

    max_time = max(classical_vals + quantum_vals + [0])
    if max_time > 10:
        ax.set_yscale("log")
    # place failure markers just above the axis floor (works on log scale too)
    floor = min([v for v in classical_vals + quantum_vals if v > 0] + [1e-3]) / 2
    _annotate_failures(ax, slots, x, results, y_at=floor)

    plt.tight_layout()
    path = output_dir / "02_training_time_comparison.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_accuracy_vs_time(results: dict, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 8))

    for algo, res in results["classical"].items():
        ax.scatter(res.train_time_sec, res.accuracy, s=200, color=PALETTE["classical"],
                   marker='o', edgecolors='black', linewidth=1.5, alpha=0.8)
        ax.annotate(algo, (res.train_time_sec, res.accuracy), xytext=(5, 5),
                   textcoords="offset points", fontsize=9)

    for algo, res in results["quantum"].items():
        ax.scatter(res.train_time_sec, res.accuracy, s=200, color=PALETTE["quantum"],
                   marker='s', edgecolors='black', linewidth=1.5, alpha=0.8)
        ax.annotate(algo, (res.train_time_sec, res.accuracy), xytext=(5, 5),
                   textcoords="offset points", fontsize=9)

    ax.set_xlabel("Training Time (seconds)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Training Time: Classical (○) vs Quantum (□)")
    ax.grid(True, alpha=0.3)

    # Document any quantum models that failed (they cannot be plotted as points).
    failed = list(results.get("quantum_failed", {}).keys())
    if failed:
        ax.text(0.98, 0.02, "Quantum failed (no data): " + ", ".join(failed),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                color="#C0392B",
                bbox=dict(boxstyle="round", fc="white", ec="#C0392B", alpha=0.85))

    classical_patch = mpatches.Patch(color=PALETTE["classical"], label="Classical", alpha=0.8)
    quantum_patch = mpatches.Patch(color=PALETTE["quantum"], label="Quantum", alpha=0.8)
    ax.legend(handles=[classical_patch, quantum_patch])

    plt.tight_layout()
    path = output_dir / "03_accuracy_vs_time.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_variational_comparison(results: dict, output_dir: Path) -> Path:
    """Training-time cost: variational vs non-variational quantum models."""
    fig, ax = plt.subplots(figsize=(12, 6))

    q = results["quantum"]
    def _cost(v):  # total quantum compute time (fair for lazy kernel/fidelity models)
        return v.train_time_sec + v.inference_time_ms / 1000.0
    var = [(k, v) for k, v in q.items() if getattr(v, "is_variational", True)]
    non = [(k, v) for k, v in q.items() if not getattr(v, "is_variational", True)]
    ordered = non + var  # non-variational first, then variational

    names = [k for k, _ in ordered]
    times = [_cost(v) for _, v in ordered]
    colors = ["#55A868" if not getattr(v, "is_variational", True) else "#C44E52"
              for _, v in ordered]

    x = np.arange(len(names))
    ax.bar(x, times, color=colors)
    for i, t in enumerate(times):
        ax.annotate(f"{t:.3g}s", (i, t), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Total compute time: train + inference (seconds)")
    ax.set_title("Quantum Compute-Time Cost: Non-variational vs Variational")
    if max(times + [0]) > 10:
        ax.set_yscale("log")

    legend = [mpatches.Patch(color="#55A868", label="Non-variational"),
              mpatches.Patch(color="#C44E52", label="Variational")]
    ax.legend(handles=legend)

    # mean-time annotation + speedup factor
    mean_non = np.mean([_cost(v) for _, v in non]) if non else 0.0
    mean_var = np.mean([_cost(v) for _, v in var]) if var else 0.0
    if mean_non > 0 and mean_var > 0:
        factor = mean_var / mean_non
        rel = (f"variational ~{factor:.1f}× slower" if factor >= 1
               else f"variational ~{1/factor:.1f}× faster (noiseless sim)")
        ax.text(0.98, 0.95,
                f"mean non-variational: {mean_non:.3g}s\n"
                f"mean variational: {mean_var:.3g}s\n"
                f"{rel}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))

    plt.tight_layout()
    path = output_dir / "07_variational_vs_nonvariational.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_accuracy_gap(results: dict, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))

    comparisons = results["comparisons"]

    names = [f"{c['classical_model']} → {c['quantum_model']}" for c in comparisons]
    gaps = [c["accuracy_gap"] for c in comparisons]
    quantum_better = [c["quantum_better"] for c in comparisons]

    colors = [PALETTE["quantum"] if qb else PALETTE["classical"] for qb in quantum_better]
    bars = ax.barh(names, gaps, color=colors, edgecolor='black', linewidth=0.5)

    for i, (bar, qb, gap) in enumerate(zip(bars, quantum_better, gaps)):
        label = f"✓ Quantum +{gap:.3f}" if qb else f"✗ Classical +{gap:.3f}"
        ax.text(gap + 0.01, bar.get_y() + bar.get_height()/2, label, va='center', fontsize=9)

    ax.set_xlabel("Accuracy Gap (absolute difference)")
    ax.set_title("Pairwise Classical vs Quantum Accuracy Gap")
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    path = output_dir / "04_accuracy_gap.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_metrics_radar(results: dict, output_dir: Path) -> Path:
    metrics = ["Accuracy", "F1 Macro", "Precision", "Recall"]
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})

    if results["classical"]:
        classical_best = max(results["classical"].values(), key=lambda x: x.accuracy)
        classical_vals = [classical_best.accuracy, classical_best.f1_macro,
                          classical_best.precision_macro, classical_best.recall_macro]
        classical_vals += classical_vals[:1]
        ax.plot(angles, classical_vals, 'o-', linewidth=2, color=PALETTE["classical"],
                label=f"Classical: {classical_best.name}")
        ax.fill(angles, classical_vals, alpha=0.1, color=PALETTE["classical"])

    if results["quantum"]:
        quantum_best = max(results["quantum"].values(), key=lambda x: x.accuracy)
        quantum_vals = [quantum_best.accuracy, quantum_best.f1_macro,
                        quantum_best.precision_macro, quantum_best.recall_macro]
        quantum_vals += quantum_vals[:1]
        ax.plot(angles, quantum_vals, 'o-', linewidth=2, color=PALETTE["quantum"],
                label=f"Quantum: {quantum_best.name}")
        ax.fill(angles, quantum_vals, alpha=0.1, color=PALETTE["quantum"])

    ax.set_ylim(0, 1.05)
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.2, 1.1), fontsize=10)
    ax.set_title("Performance Metrics: Classical vs Quantum (Best Models)", fontsize=14, pad=20)

    plt.tight_layout()
    path = output_dir / "05_metrics_radar.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_confusion_matrices(results: dict, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    CLASS_NAMES = ["Setosa", "Versicolor", "Virginica"]

    if results["classical"]:
        classical_best = max(results["classical"].values(), key=lambda x: x.accuracy)
        sns.heatmap(classical_best.confusion_mat, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    ax=axes[0], cbar=False)
        axes[0].set_title(f"Classical: {classical_best.name}\nAccuracy: {classical_best.accuracy:.4f}")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")
    else:
        axes[0].text(0.5, 0.5, "No classical results", ha='center', va='center')
        axes[0].set_title("Classical Model")

    if results["quantum"]:
        quantum_best = max(results["quantum"].values(), key=lambda x: x.accuracy)
        sns.heatmap(quantum_best.confusion_mat, annot=True, fmt='d', cmap='Oranges',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    ax=axes[1], cbar=False)
        axes[1].set_title(f"Quantum: {quantum_best.name}\nAccuracy: {quantum_best.accuracy:.4f}")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Actual")
    else:
        axes[1].text(0.5, 0.5, "No quantum results", ha='center', va='center')
        axes[1].set_title("Quantum Model")

    plt.suptitle("Confusion Matrices: Best Classical vs Best Quantum Models", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "06_confusion_matrices.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path

# ════════════════════════════════════════════════════════════════════════
#  NEW CHARTS — derived from the application logs / in-memory results
#  (inference cost, circuit depth, convergence, time-ratio, heatmap,
#   QPU execution timeline, cumulative circuits, reliability)
# ════════════════════════════════════════════════════════════════════════

# fixed display order for quantum models (matches comparator insertion order)
_Q_ORDER = ["QSVM", "QkNN", "QCL", "VQC", "QNN", "QFC", "QFE-LR"]
_C_ORDER = ["SVM", "kNN", "LogReg", "MLP"]


def _q_items(results: dict):
    """Quantum results in canonical order (skips missing models)."""
    q = results.get("quantum", {})
    ordered = [(k, q[k]) for k in _Q_ORDER if k in q]
    ordered += [(k, v) for k, v in q.items() if k not in _Q_ORDER]
    return ordered


def _all_items(results: dict):
    """(name, result, category) for every model, classical first."""
    out = []
    c = results.get("classical", {})
    for k in _C_ORDER:
        if k in c:
            out.append((k, c[k], "classical"))
    for k, v in c.items():
        if k not in _C_ORDER:
            out.append((k, v, "classical"))
    for k, v in _q_items(results):
        out.append((k, v, "quantum"))
    return out


def _plot_inference_time(results: dict, output_dir: Path) -> Path:
    """Inference (prediction) time per model — exposes outliers like QFC."""
    fig, ax = plt.subplots(figsize=(12, 6))

    items = _all_items(results)
    names = [k for k, _v, _c in items]
    vals = [max(v.inference_time_ms, 1e-3) for _k, v, _c in items]
    colors = [PALETTE["classical"] if c == "classical" else PALETTE["quantum"]
              for _k, _v, c in items]

    x = np.arange(len(names))
    ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.4)
    for i, t in enumerate(vals):
        ax.annotate(f"{t:.3g}", (i, t), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Inference time (ms)")
    ax.set_title("Inference-Time Cost per Model (lower is better)")
    if max(vals) / min(vals) > 50:
        ax.set_yscale("log")

    legend = [mpatches.Patch(color=PALETTE["classical"], label="Classical"),
              mpatches.Patch(color=PALETTE["quantum"], label="Quantum")]
    ax.legend(handles=legend)

    plt.tight_layout()
    path = output_dir / "08_inference_time.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_time_ratio_pairs(results: dict, output_dir: Path) -> Path:
    """Quantum/classical training-time ratio per pair (log scale)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    comps = results["comparisons"]
    names, ratios = [], []
    for c in comps:
        r = c.get("time_ratio", float("inf"))
        if r in (float("inf"), 0) or r != r:   # skip inf / 0 / NaN
            continue
        names.append(f"{c['classical_model']} → {c['quantum_model']}")
        ratios.append(r)

    if not ratios:
        plt.close(fig)
        return None

    x = np.arange(len(names))
    ax.bar(x, ratios, color=PALETTE["quantum"], edgecolor="black", linewidth=0.5)
    for i, r in enumerate(ratios):
        ax.annotate(f"{r:,.0f}×", (i, r), ha="center", va="bottom", fontsize=9)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(0.01, 1.0, " parity (1×)", transform=ax.get_yaxis_transform(),
            va="bottom", ha="left", fontsize=8, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Time ratio  (quantum / classical)")
    ax.set_title("How Much Slower Is Quantum? Training-Time Ratio per Pair")
    if max(ratios) > 20:
        ax.set_yscale("log")

    plt.tight_layout()
    path = output_dir / "09_time_ratio_pairs.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_circuit_depth(results: dict, output_dir: Path) -> Path:
    """Transpiled circuit depth per quantum model."""
    fig, ax = plt.subplots(figsize=(11, 6))

    items = _q_items(results)
    names = [k for k, _v in items]
    depths = [getattr(v, "circuit_depth", 0) for _k, v in items]
    colors = [PALETTE.get(k, PALETTE["quantum"]) for k in names]

    x = np.arange(len(names))
    ax.bar(x, depths, color=colors, edgecolor="black", linewidth=0.4)
    for i, d in enumerate(depths):
        ax.annotate(str(d), (i, d), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Circuit depth (transpiled)")
    ax.set_title("Quantum Circuit Depth per Model")

    plt.tight_layout()
    path = output_dir / "10_circuit_depth.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_convergence_curves(results: dict, output_dir: Path):
    """Optimiser loss vs iteration for variational models. Returns None if no
    convergence history was captured (non-variational run / no callback)."""
    series = [(k, list(v.convergence_values))
              for k, v in _q_items(results)
              if getattr(v, "convergence_values", None)]
    if not series:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, vals in series:
        iters = np.arange(1, len(vals) + 1)
        ax.plot(iters, vals, marker="o", linewidth=1.8,
                color=PALETTE.get(name, None), label=name)

    ax.set_xlabel("Optimiser iteration (COBYLA)")
    ax.set_ylabel("Objective value (loss)")
    ax.set_title("Variational Training Convergence")
    ax.grid(True, alpha=0.3)
    ax.legend()
    # integer iteration ticks
    maxlen = max(len(v) for _n, v in series)
    ax.set_xticks(np.arange(1, maxlen + 1))

    plt.tight_layout()
    path = output_dir / "11_convergence_curves.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_accuracy_vs_depth(results: dict, output_dir: Path) -> Path:
    """Accuracy vs circuit depth — is deeper buying anything here?"""
    fig, ax = plt.subplots(figsize=(10, 7))

    for k, v in _q_items(results):
        d = getattr(v, "circuit_depth", 0)
        ax.scatter(d, v.accuracy, s=220, color=PALETTE.get(k, PALETTE["quantum"]),
                   edgecolors="black", linewidth=1.3, alpha=0.85)
        ax.annotate(k, (d, v.accuracy), xytext=(6, 6),
                    textcoords="offset points", fontsize=9)

    ax.set_xlabel("Circuit depth (transpiled)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Quantum Accuracy vs Circuit Depth")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = output_dir / "12_accuracy_vs_depth.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_metrics_heatmap(results: dict, output_dir: Path) -> Path:
    """Models × metrics heatmap (accuracy, F1, precision, recall)."""
    items = _all_items(results)
    metrics = ["Accuracy", "F1 macro", "Precision", "Recall"]
    rows, labels = [], []
    for k, v, _c in items:
        rows.append([v.accuracy, v.f1_macro,
                     getattr(v, "precision_macro", np.nan),
                     getattr(v, "recall_macro", np.nan)])
        labels.append(k)
    mat = np.array(rows, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(labels) + 2))
    sns.heatmap(mat, annot=True, fmt=".3f", cmap="viridis", vmin=0, vmax=1,
                xticklabels=metrics, yticklabels=labels, ax=ax,
                cbar_kws={"label": "score"})
    ax.set_title("Performance Metrics Heatmap: All Models")
    # separator between classical and quantum blocks
    n_classical = sum(1 for _k, _v, c in items if c == "classical")
    ax.axhline(n_classical, color="white", linewidth=2.5)

    plt.tight_layout()
    path = output_dir / "13_metrics_heatmap.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_execution_timeline(results: dict, output_dir: Path) -> Path:
    """QPU job submission timeline — exposes queue gaps on real hardware."""
    recs = results["execution"]["job_records"]
    t = [r["t"] for r in recs]
    idx = [r["idx"] for r in recs]
    sizes = [12 + 4 * r["n_circuits"] for r in recs]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, idx, "-", color="#999999", linewidth=1, alpha=0.7, zorder=1)
    ax.scatter(t, idx, s=sizes, color=PALETTE["quantum"],
               edgecolors="black", linewidth=0.5, zorder=2)

    # highlight the largest inter-job gap (queue wait)
    if len(t) > 1:
        gaps = [(t[i] - t[i - 1], i) for i in range(1, len(t))]
        gmax, gi = max(gaps)
        if gmax > 5:
            ax.annotate(f"queue gap ≈ {gmax:.0f}s",
                        ((t[gi] + t[gi - 1]) / 2, (idx[gi] + idx[gi - 1]) / 2),
                        fontsize=9, color="#C0392B", ha="center",
                        bbox=dict(boxstyle="round", fc="white", ec="#C0392B", alpha=0.85))

    ax.set_xlabel("Seconds since first job")
    ax.set_ylabel("Job number")
    ax.set_title("QPU Execution Timeline (marker size ∝ circuits per job)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "14_execution_timeline.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_circuits_cumulative(results: dict, output_dir: Path) -> Path:
    """Cumulative circuits submitted over wall-clock time."""
    recs = results["execution"]["job_records"]
    t = [r["t"] for r in recs]
    cum = [r["cumulative_circuits"] for r in recs]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.step(t, cum, where="post", color=PALETTE["quantum"], linewidth=2)
    ax.fill_between(t, cum, step="post", alpha=0.15, color=PALETTE["quantum"])

    total = results["execution"].get("submitted_circuits", cum[-1] if cum else 0)
    ax.set_xlabel("Seconds since first job")
    ax.set_ylabel("Cumulative circuits submitted")
    ax.set_title(f"Cumulative Circuit Submissions (total = {total})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "15_circuits_cumulative.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_execution_reliability(results: dict, output_dir: Path) -> Path:
    """Job reliability donut: succeeded vs failed (reproducibility evidence)."""
    execu = results["execution"]
    submitted = int(execu.get("submitted_jobs", 0))
    failed = int(execu.get("failed_jobs", 0))
    ok = max(submitted - failed, 0)

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, _txt, _aut = ax.pie(
        [ok, failed] if (ok + failed) else [1, 0],
        labels=["Succeeded", "Failed"],
        colors=["#2ECC71", "#E74C3C"],
        autopct=lambda p: f"{p*(ok+failed)/100:.0f}",
        startangle=90, counterclock=False,
        wedgeprops=dict(width=0.42, edgecolor="white"),
    )
    backend = execu.get("backend_label", "—")
    circuits = execu.get("submitted_circuits", "—")
    ax.text(0, 0, f"{ok}/{submitted}\njobs OK", ha="center", va="center",
            fontsize=15, fontweight="bold")
    ax.set_title(f"QPU Execution Reliability\n{backend} · {circuits} circuits")

    plt.tight_layout()
    path = output_dir / "16_execution_reliability.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
