# -*- coding: utf-8 -*-
"""
================================================================================
 qml_classical_viz.py
 Visualization module for the Iris QML vs. classical ML study.
 Generates TWO independent sets of figures from the ALREADY-COLLECTED data
 (comparison_results.json) WITHOUT re-running anything on the QPU.
================================================================================

THREE PARADIGMS (taxonomy)
--------------------------
The models are grouped into three families, derived directly from the JSON
fields `is_variational`, `num_parameters` and `convergence`:

  1. CLASSICAL ML                 -> SVM, kNN, LogReg, MLP
  2. QML (non-variational)        -> QSVM, QkNN, QFC, QFE-LR
       (quantum kernel / fidelity / embedding methods; num_parameters = 0,
        no training trajectory -> NO parameter-optimization landscape)
  3. QML (variational circuits)   -> QCL, VQC, QNN
       (parametrized circuits trained by an optimizer; have a `convergence`
        history -> they DO have a cost landscape)

FIGURE SET 1 — LOSS LANDSCAPES (3 panels, one per paradigm)
-----------------------------------------------------------
  (a) CLASSICAL          : REAL convex logistic-loss surface over two weights,
                           computed locally on Iris (no QPU). Explains LogReg's
                           ~1.0 accuracy: a well-conditioned convex basin.
  (b) QML non-variational: a FLAT plane. These methods do not optimize circuit
                           parameters, so there is no gradient/loss landscape to
                           traverse. The flatness IS the message ("training-free
                           kernel/fidelity method"). Height = error level (1-acc).
  (c) QML variational    : surface RECONSTRUCTED from the collected `convergence`
                           statistics (plateau = mean, well = min, ruggedness =
                           std). The overlaid trajectory uses the REAL measured
                           cost per iteration (z = real cost; (theta1,theta2)
                           positions are illustrative — the JSON does not store
                           per-iteration parameters). The near-flat, rugged shape
                           is the visual signature of poor trainability
                           (barren-plateau-like behaviour).

FIGURE SET 2 — CLASSICAL vs QML (non-variational AND variational)
-----------------------------------------------------------------
  Real metrics from the JSON, colour-coded by paradigm:
    (1) Accuracy per model
    (2) Macro-F1 per model
    (3) Training time (s, log scale)
    (4) Accuracy vs inference time (ms, log scale) scatter

ACADEMIC INTEGRITY
------------------
Panel (c) is a conceptual RECONSTRUCTION from the collected metrics, not the
literal sweep of the circuit; this is printed on the figure itself. For the
EXACT ansatz landscape, use `variational_landscape_statevector()` (local
simulator, still no QPU; requires qiskit).

DEPENDENCIES (all local; no quantum SDK required)
-------------------------------------------------
    numpy, matplotlib, scikit-learn

USAGE
-----
    python qml_classical_viz.py --json comparison_results.json --what both \
           --var-model VQC --nonvar-model QSVM --pdf
================================================================================
"""

import json
import argparse

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless backend; remove if running locally with a window
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (enables 3d projection)


# ------------------------------------------------------------------------------
# Taxonomy, labels and colours
# ------------------------------------------------------------------------------
TAXONOMY = {
    "classical":          ["SVM", "kNN", "LogReg", "MLP"],
    "qml_nonvariational": ["QSVM", "QkNN", "QFC", "QFE-LR"],
    "qml_variational":    ["QCL", "VQC", "QNN"],
}

CATEGORY_LABELS = {
    "classical":          "Classical ML",
    "qml_nonvariational": "QML (non-variational)",
    "qml_variational":    "QML (variational circuits)",
}

CATEGORY_COLORS = {
    "classical":          "#2a9d8f",  # teal
    "qml_nonvariational": "#e0a128",  # amber
    "qml_variational":    "#6a4c93",  # purple
}


def category_of(model_key):
    """Return the paradigm a model key belongs to."""
    for cat, members in TAXONOMY.items():
        if model_key in members:
            return cat
    raise KeyError(f"Unknown model '{model_key}'.")


# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------
def load_results(path):
    """Read the comparison_results.json produced by the previous run."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def metric(results, model_key, field):
    """Fetch a metric for any model, regardless of classical/quantum bucket."""
    if model_key in results.get("classical", {}):
        return results["classical"][model_key][field]
    if model_key in results.get("quantum", {}):
        return results["quantum"][model_key][field]
    raise KeyError(f"Model '{model_key}' not found in results.")


def convergence_stats(convergence):
    """
    Summarise a cost-vs-iteration history into the numbers that parametrise the
    reconstructed variational surface. Everything comes from REAL data:

        base      -> plateau level (typical cost)   = mean
        ceil      -> max observed cost              = max
        floor     -> best cost (well bottom)        = min
        roughness -> ruggedness amplitude           = std
    """
    c = np.asarray(convergence, dtype=float)
    if c.size == 0:
        raise ValueError("Model has no convergence history (non-variational).")
    return {
        "base": float(np.mean(c)),
        "ceil": float(np.max(c)),
        "floor": float(np.min(c)),
        "roughness": float(np.std(c)),
        "values": c,
    }


# ------------------------------------------------------------------------------
# FIGURE SET 1 — landscape builders
# ------------------------------------------------------------------------------
def classical_logreg_landscape(n=70):
    """
    REAL logistic-regression loss surface (cross-entropy) over two weights,
    computed locally on Iris (no QPU). Uses two overlapping classes so the
    basin is convex with a finite minimum.

    Returns (W1, W2, L, (w1_opt, w2_opt, L_opt)).
    """
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    iris = load_iris()
    mask = iris.target >= 1                     # versicolor (1) vs virginica (2)
    Xf = iris.data[mask][:, 2:4]                # petal length & width
    y = (iris.target[mask] == 2).astype(float)
    Xf = StandardScaler().fit_transform(Xf)

    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000).fit(Xf, y)
    w_opt = clf.coef_.ravel()
    b_opt = float(clf.intercept_[0])

    def loss(w1, w2):
        z = Xf[:, 0] * w1 + Xf[:, 1] * w2 + b_opt
        p = 1.0 / (1.0 + np.exp(-z))
        eps = 1e-12
        return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

    span = 2.6
    w1 = np.linspace(w_opt[0] - span, w_opt[0] + span, n)
    w2 = np.linspace(w_opt[1] - span, w_opt[1] + span, n)
    W1, W2 = np.meshgrid(w1, w2)
    L = np.vectorize(loss)(W1, W2)
    return W1, W2, L, (w_opt[0], w_opt[1], loss(w_opt[0], w_opt[1]))


def nonvariational_flat_landscape(error_level, n=70, ripple=0.01):
    """
    QML non-variational paradigm: there is NO parameter-optimization landscape.
    Represented as an (essentially) flat plane at height = error_level (1-acc).
    A tiny ripple is added only so the surface renders nicely; the plane is flat
    by design — that is the conceptual distinction from variational circuits.

    Returns (X, Y, Z).
    """
    g = np.linspace(-np.pi, np.pi, n)
    X, Y = np.meshgrid(g, g)
    Z = error_level + ripple * np.sin(0.7 * X) * np.cos(0.7 * Y)
    return X, Y, Z


def variational_landscape_reconstructed(stats, n=70, well_width=0.9, exaggerate=1.0):
    """
    QML variational paradigm: cost surface reconstructed from collected metrics.

        Z(t1,t2) = base
                   - depth * exp(-(t1^2 + t2^2) / (2*width^2))    # central well
                   + roughness * sin(1.5*t1) * cos(1.5*t2)        # ruggedness

      base, depth (= base - floor) and roughness come from the REAL convergence
      stats. `well_width` (sigma) and `exaggerate` are purely DIDACTIC:
        exaggerate=1.0 -> faithful to the measured scale (tends to look almost
                          flat, which is itself the trainability-problem signal);
        exaggerate>1.0 -> deepens the well for readability (gets flagged on plot).

    Returns (X, Y, Z).
    """
    g = np.linspace(-np.pi, np.pi, n)
    X, Y = np.meshgrid(g, g)
    depth = (stats["base"] - stats["floor"]) * exaggerate
    well = depth * np.exp(-(X**2 + Y**2) / (2.0 * well_width**2))
    rough = stats["roughness"] * np.sin(1.5 * X) * np.cos(1.5 * Y)
    return X, Y, stats["base"] - well + rough


def real_trajectory(stats, r0=2.6, turns=1.5):
    """
    Place the REAL measured cost history as a spiral descending toward the well.
        z  = real measured cost per iteration (collected data)
        (t1, t2) = ILLUSTRATIVE positions (the JSON stores only cost, not the
                   parameters at each step). This caveat is printed on the plot.
    """
    z = stats["values"]
    m = len(z)
    t = np.linspace(0.0, 1.0, m)
    r = (1.0 - t) * r0
    ang = t * turns * 2.0 * np.pi
    return r * np.cos(ang), r * np.sin(ang), z


def generate_landscape_figure(results, var_model="VQC", nonvar_model="QSVM",
                              out="landscapes_by_paradigm", pdf=False,
                              exaggerate=1.0):
    """Set 1: three-panel landscape figure, one panel per paradigm."""
    # --- gather data ---
    conv = results["quantum"][var_model]["convergence"]
    stats = convergence_stats(conv)
    acc_var = results["quantum"][var_model]["accuracy"]
    acc_nonvar = results["quantum"][nonvar_model]["accuracy"]
    acc_cls = results["classical"]["LogReg"]["accuracy"]

    Wc1, Wc2, Lc, (wo1, wo2, lo) = classical_logreg_landscape()
    Xn, Yn, Zn = nonvariational_flat_landscape(1.0 - acc_nonvar)
    Xv, Yv, Zv = variational_landscape_reconstructed(stats, exaggerate=exaggerate)
    tx, ty, tz = real_trajectory(stats)

    fig = plt.figure(figsize=(16.5, 5.6))

    # (a) Classical -----------------------------------------------------------
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.plot_surface(Wc1, Wc2, Lc, cmap=cm.GnBu, alpha=0.7, linewidth=0,
                     antialiased=True, rstride=2, cstride=2)
    ax1.scatter([wo1], [wo2], [lo], color="#c0392b", s=55, marker="o", zorder=6)
    ax1.set_title(f"(a) Classical ML — Logistic Regression\n"
                  f"convex basin · accuracy = {acc_cls:.2f}",
                  fontsize=10.5, color=CATEGORY_COLORS["classical"])
    ax1.set_xlabel(r"$w_1$"); ax1.set_ylabel(r"$w_2$"); ax1.set_zlabel("loss")
    ax1.view_init(elev=28, azim=-120)

    # (b) QML non-variational -------------------------------------------------
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.plot_surface(Xn, Yn, Zn, color=CATEGORY_COLORS["qml_nonvariational"],
                     alpha=0.5, linewidth=0, antialiased=True,
                     rstride=2, cstride=2)
    ax2.text2D(0.5, 0.78, "no parameter landscape\n(training-free)",
               transform=ax2.transAxes, ha="center", fontsize=8.5,
               color="#7a5a00")
    ax2.set_title(f"(b) QML (non-variational) — {nonvar_model}\n"
                  f"flat · no trainable params · accuracy = {acc_nonvar:.2f}",
                  fontsize=10.5, color=CATEGORY_COLORS["qml_nonvariational"])
    ax2.set_xlabel(r"$\theta_1$"); ax2.set_ylabel(r"$\theta_2$")
    ax2.set_zlabel("error (1 - acc)")
    ax2.set_zlim(0.0, 1.0)
    ax2.view_init(elev=24, azim=-60)

    # (c) QML variational -----------------------------------------------------
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.plot_surface(Xv, Yv, Zv, cmap=cm.Purples, alpha=0.6, linewidth=0,
                     antialiased=True, rstride=2, cstride=2)
    ax3.plot(tx, ty, tz, color="#d9730d", lw=1.6, alpha=0.9, zorder=5)
    ax3.scatter(tx, ty, tz, color="#b35900", s=26, zorder=6)
    ax3.set_title(f"(c) QML (variational) — {var_model}\n"
                  f"rugged / near-flat · accuracy = {acc_var:.2f}",
                  fontsize=10.5, color=CATEGORY_COLORS["qml_variational"])
    ax3.set_xlabel(r"$\theta_1$"); ax3.set_ylabel(r"$\theta_2$")
    ax3.set_zlabel("cost")
    pad = max(0.05, 0.25 * (stats["ceil"] - stats["floor"]))
    ax3.set_zlim(stats["floor"] - pad, stats["ceil"] + pad)
    ax3.view_init(elev=22, azim=-60)

    note = ("(a) real logistic loss on Iris.  "
            "(b) flat by design: non-variational QML has no parameter-training "
            "landscape; height = 1 - accuracy.  "
            "(c) surface RECONSTRUCTED from collected convergence stats; "
            "trajectory z = real measured cost, (theta1, theta2) illustrative. "
            "No QPU was used.")
    if exaggerate != 1.0:
        note = f"[well visually exaggerated x{exaggerate:g}]  " + note
    fig.text(0.5, 0.015, note, ha="center", va="bottom", fontsize=7.5,
             color="#444444", wrap=True)
    fig.suptitle("Cost landscapes by paradigm: Classical ML · QML (non-variational) · QML (variational)",
                 fontsize=12.5, y=0.99)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    png = f"{out}.png"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    print(f"[ok] saved: {png}")
    if pdf:
        fig.savefig(f"{out}.pdf", bbox_inches="tight")
        print(f"[ok] saved: {out}.pdf")
    plt.close(fig)
    return png


# ------------------------------------------------------------------------------
# FIGURE SET 2 — classical vs QML (non-variational AND variational) metrics
# ------------------------------------------------------------------------------
def _ordered_models(results):
    """Return models grouped by paradigm, keeping only those present in JSON."""
    order, cats = [], []
    for cat in ("classical", "qml_nonvariational", "qml_variational"):
        for m in TAXONOMY[cat]:
            present = m in results.get("classical", {}) or m in results.get("quantum", {})
            if present:
                order.append(m)
                cats.append(cat)
    return order, cats


def generate_comparison_charts(results, out="classical_vs_qml", pdf=False):
    """Set 2: four panels comparing all paradigms on real collected metrics."""
    models, cats = _ordered_models(results)
    colors = [CATEGORY_COLORS[c] for c in cats]

    acc = [metric(results, m, "accuracy") for m in models]
    f1 = [metric(results, m, "f1_macro") for m in models]
    ttrain = [max(metric(results, m, "train_time_sec"), 1e-7) for m in models]
    tinfer = [max(metric(results, m, "inference_time_ms"), 1e-4) for m in models]

    x = np.arange(len(models))
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))

    def _bars(ax, vals, title, ylabel, log=False):
        ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale("log")
        # light separators between paradigm groups
        boundaries = [i + 0.5 for i in range(len(cats) - 1) if cats[i] != cats[i + 1]]
        for b in boundaries:
            ax.axvline(b, color="#cccccc", ls="--", lw=0.8)
        ax.grid(axis="y", alpha=0.25)

    _bars(axes[0, 0], acc, "Accuracy by model", "accuracy")
    axes[0, 0].set_ylim(0, 1.05)
    _bars(axes[0, 1], f1, "Macro-F1 by model", "macro-F1")
    axes[0, 1].set_ylim(0, 1.05)
    _bars(axes[1, 0], ttrain, "Training time (log scale)", "seconds", log=True)

    # (4) accuracy vs inference time scatter
    ax4 = axes[1, 1]
    for xi, yi, ci, mi in zip(tinfer, acc, colors, models):
        ax4.scatter(xi, yi, color=ci, s=90, edgecolor="white", linewidth=0.8, zorder=3)
        ax4.annotate(mi, (xi, yi), textcoords="offset points", xytext=(6, 4),
                     fontsize=8)
    ax4.set_xscale("log")
    ax4.set_xlabel("inference time (ms, log scale)")
    ax4.set_ylabel("accuracy")
    ax4.set_ylim(0, 1.05)
    ax4.set_title("Accuracy vs inference time", fontsize=11)
    ax4.grid(alpha=0.25)

    # shared legend
    handles = [Patch(facecolor=CATEGORY_COLORS[c], label=CATEGORY_LABELS[c])
               for c in ("classical", "qml_nonvariational", "qml_variational")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Classical ML vs QML (non-variational and variational) — Iris dataset",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    png = f"{out}.png"
    fig.savefig(png, dpi=170, bbox_inches="tight")
    print(f"[ok] saved: {png}")
    if pdf:
        fig.savefig(f"{out}.pdf", bbox_inches="tight")
        print(f"[ok] saved: {out}.pdf")
    plt.close(fig)
    return png


# ------------------------------------------------------------------------------
# OPTIONAL — exact variational landscape via LOCAL simulator (still no QPU)
# ------------------------------------------------------------------------------
def variational_landscape_statevector(n=40, reps=1):
    """
    [OPTIONAL — exact ansatz landscape for the thesis]
    Evaluate a 2-qubit variational ansatz on a LOCAL statevector simulator
    (no QPU) over a grid of two parameters. Replace the ansatz and the cost
    function with the ones from your iris_qml_5 project to reproduce YOUR model.
    Requires qiskit.
    """
    try:
        from qiskit.circuit.library import RealAmplitudes
        from qiskit.quantum_info import Statevector
    except ImportError as e:
        raise ImportError("qiskit not found. `pip install qiskit` to use the "
                          "exact statevector landscape (still no QPU).") from e

    ansatz = RealAmplitudes(num_qubits=2, reps=reps)
    base = np.zeros(ansatz.num_parameters)
    g = np.linspace(-np.pi, np.pi, n)
    T1, T2 = np.meshgrid(g, g)
    C = np.zeros_like(T1)
    for i in range(n):
        for j in range(n):
            p = base.copy()
            p[0] = T1[i, j]
            if ansatz.num_parameters > 1:
                p[1] = T2[i, j]
            sv = Statevector.from_instruction(ansatz.assign_parameters(p))
            C[i, j] = 1.0 - sv.probabilities()[0]  # replace with your cost
    return T1, T2, C


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Generate landscape and comparison figures for the Iris "
                    "QML vs classical study, from collected data only (no QPU).")
    ap.add_argument("--json", default="comparison_results.json")
    ap.add_argument("--what", choices=["landscapes", "comparison", "both"],
                    default="both")
    ap.add_argument("--var-model", default="VQC", choices=["QCL", "VQC", "QNN"])
    ap.add_argument("--nonvar-model", default="QSVM",
                    choices=["QSVM", "QkNN", "QFC", "QFE-LR"])
    ap.add_argument("--exaggerate", type=float, default=1.0)
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()

    results = load_results(args.json)
    if args.what in ("landscapes", "both"):
        generate_landscape_figure(results, var_model=args.var_model,
                                  nonvar_model=args.nonvar_model,
                                  pdf=args.pdf, exaggerate=args.exaggerate)
    if args.what in ("comparison", "both"):
        generate_comparison_charts(results, pdf=args.pdf)


if __name__ == "__main__":
    main()
