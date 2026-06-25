#!/usr/bin/env python3
"""
main.py — Classical vs Quantum ML Comparison Entry Point

Usage:
    python main.py                      # use BACKEND from .env
    python main.py --mode simulator     # force local quantum simulation
    python main.py --mode physical      # force real IBM QPU
    python main.py --mode noise         # IBM device noise model, run locally
    python main.py --no-charts          # skip chart generation

Each run writes a fresh, timestamped log and its own results folder:
    outputs/run_<YYYY-MM-DD_HH-MM-SS>/quantum_run.log
    outputs/run_<YYYY-MM-DD_HH-MM-SS>/comparison/...
"""

import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.logging_utils import setup_logging
from src.data_loader import load_iris_data
from src.backend_factory import build_execution_context
from src.comparator import MLComparator
from src.visualizer_compare import create_comparison_charts


def _parse_mode(argv):
    """Return the requested execution mode (or None to use .env)."""
    for i, a in enumerate(argv):
        if a == "--mode" and i + 1 < len(argv):
            return argv[i + 1].lower()
        if a.startswith("--mode="):
            return a.split("=", 1)[1].lower()
    return None


def _apply_mode(mode, log):
    """Override Config to pin a physical-QPU or simulation configuration."""
    if mode in ("simulator", "sim", "simulation"):
        Config.BACKEND = "LOCAL"
        Config.IBM_NOISE_SIM = False
        log.info("MODE: simulator → local quantum simulation (%s)", Config.SIMULATOR_METHOD)
    elif mode in ("physical", "hardware", "qpu", "ibm"):
        Config.BACKEND = "IBM"
        Config.IBM_NOISE_SIM = False
        log.info("MODE: physical → real IBM QPU (%s)", Config.IBM_BACKEND_NAME)
    elif mode in ("noise", "noise-sim", "noisy"):
        Config.BACKEND = "IBM"
        Config.IBM_NOISE_SIM = True
        log.info("MODE: noise → IBM device noise model, executed locally on Aer")
    elif mode:
        log.warning("Unknown --mode '%s' (use: simulator | physical | noise). Using .env.", mode)


def _stratified_head(X, y, n, seed):
    """Pick ~n samples while keeping all classes represented (avoids missing-class
    crashes in metrics and degenerate training on tiny hardware subsets)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per = max(1, n // len(classes))
    idx = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        rng.shuffle(c_idx)
        idx.extend(c_idx[:per])
    idx = np.array(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


def _apply_hardware_guard(data, log):
    """On a real QPU, subsample data (stratified) + cap iterations."""
    n_train = min(Config.HW_MAX_TRAIN, len(data["y_train"]))
    n_test = min(Config.HW_MAX_TEST, len(data["y_test"]))
    log.warning("HARDWARE GUARD ACTIVE — real QPU detected.")
    log.warning("  Subsampling (stratified): train %d→%d, test %d→%d | MAX_ITER %d→%d",
                len(data["y_train"]), n_train, len(data["y_test"]), n_test,
                Config.MAX_ITER, Config.HW_MAX_ITER)
    log.warning("  (Disable with HW_GUARD=false — but expect long queues and QPU-time cost.)")
    data["X_train"], data["y_train"] = _stratified_head(
        data["X_train"], data["y_train"], n_train, Config.RANDOM_SEED)
    data["X_test"], data["y_test"] = _stratified_head(
        data["X_test"], data["y_test"], n_test, Config.RANDOM_SEED + 1)
    Config.MAX_ITER = Config.HW_MAX_ITER
    return data


def main():
    skip_charts = "--no-charts" in sys.argv
    mode = _parse_mode(sys.argv)
    t_global = time.perf_counter()

    # Per-run versioned folder + timestamped log (new log every run).
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(Config.OUTPUT_DIR) / f"run_{stamp}"
    output_dir = run_dir / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / f"quantum_run_{stamp}.log"
    log = setup_logging(level=Config.LOG_LEVEL, log_file=log_file)
    log.info("RUN ID: %s", stamp)

    _apply_mode(mode, log)

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║    CLASSICAL vs QUANTUM ML COMPARISON                                        ║
║    Iris Dataset | Full Algorithm Suite                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    print(Config.summary())

    # [1/4] Data
    log.info("[1/4] Loading Iris dataset...")
    data = load_iris_data()
    log.info("  Samples: %d total | Train=%d Test=%d",
             len(data["y_full"]), len(data["y_train"]), len(data["y_test"]))
    log.info("  Features: 4 raw → %d qubits via PCA", data["num_qubits"])

    # [2/4] Backend / execution context
    log.info("[2/4] Initializing quantum execution context...")
    ctx = build_execution_context()
    log.info("  Backend label : %s", ctx.label)
    log.info("  Is hardware   : %s | Is simulator: %s", ctx.is_hardware, ctx.is_simulator)

    if ctx.is_hardware and Config.HW_GUARD:
        data = _apply_hardware_guard(data, log)

    # [3/4] Comparison
    log.info("[3/4] Running Classical vs Quantum comparison...")
    warnings.filterwarnings(
        "ignore",
        message="No gradient function provided, creating a gradient function. "
                "If your Sampler requires transpilation, please provide a pass manager.")
    comparator = MLComparator(ctx)
    results = comparator.run_comparison(
        X_train=data["X_train"], y_train=data["y_train"],
        X_test=data["X_test"], y_test=data["y_test"],
    )
    comparator.print_summary(results)

    # Execution proof
    exec_sum = ctx.execution_summary()
    log.info("=" * 80)
    log.info("EXECUTION PROOF (from LoggingSamplerV2)")
    log.info("  Backend           : %s", exec_sum.get("backend_label"))
    log.info("  Ran on hardware   : %s", exec_sum.get("is_hardware"))
    log.info("  Jobs submitted    : %s", exec_sum.get("submitted_jobs"))
    log.info("  Jobs failed       : %s", exec_sum.get("failed_jobs"))
    log.info("  Circuits submitted: %s", exec_sum.get("submitted_circuits"))
    if exec_sum.get("sample_job_ids"):
        log.info("  Sample job ids    : %s", exec_sum.get("sample_job_ids"))
        log.info("  → verify at https://quantum.ibm.com/jobs")
    log.info("=" * 80)

    comparator.export_results(results, output_dir / "comparison_results.json")

    if not skip_charts:
        log.info("[4/4] Generating comparison charts...")
        create_comparison_charts(results, output_dir, execution=exec_sum)

    elapsed = time.perf_counter() - t_global
    log.info("✓ Comparison completed in %.1fs", elapsed)
    log.info("Run folder : %s", run_dir.resolve())
    log.info("Log file   : %s", log_file.resolve())


if __name__ == "__main__":
    main()
