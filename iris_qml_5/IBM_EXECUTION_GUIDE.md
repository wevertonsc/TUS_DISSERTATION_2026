# Running the Quantum Part on IBM Quantum — and Verifying It

## TL;DR of what was wrong
The project was configured with `BACKEND=LOCAL`, so the "quantum" models ran on a
**local simulator**, not on IBM. Even switching to `BACKEND=IBM` would not have sent
most work to IBM because of several wiring bugs (now fixed):

| Model | Before | After |
|-------|--------|-------|
| QSVM  | `FidelityQuantumKernel` ignored the sampler → always local | kernel built from `ComputeUncompute(sampler, pass_manager)` → runs on the selected backend |
| QkNN  | pure NumPy Gaussian kernel (not quantum at all) | honest: labelled *classical proxy* by default; set `QKNN_QUANTUM=true` for a real fidelity circuit |
| QCL / VQC | no `pass_manager` → fail on real hardware (non-ISA circuits) | `sampler` **and** `pass_manager` passed → ISA-transpiled for the device |
| QNN   | `from qiskit.algorithms...` (removed in Qiskit 1.0) → silently skipped; `input_params=[]` ignored data | imports from `qiskit_machine_learning.optimizers`; feature-map inputs + ansatz weights + `pass_manager` |
| Conn. | `instance="ibm-q/open/main"` (legacy, invalid on new platform) | `channel=ibm_quantum_platform`, instance omitted, `least_busy` auto-select |

## How to run on IBM
1. Put a **valid** IBM Quantum Platform token in `.env` (`IBM_API_TOKEN=...`).
   Get it from https://quantum.ibm.com (the *new* platform).
2. In `.env`: `BACKEND=IBM`. Pick a device with `IBM_BACKEND_NAME=least_busy`
   (auto) or a name like `ibm_fez`.
3. `pip install -r requirements.txt`
4. `python main.py`

## How to verify it really ran on IBM (the logs)
Every circuit submission is intercepted by `LoggingSamplerV2` and written to the
console **and** to `outputs/quantum_run.log`, e.g.:

```
[IBM:ibm_fez] job #3 → 12 circuit(s) | qubits=2 depth=27 | total circuits=156
      ↳ job_id=ch1abcd...  backend=ibm_fez
```

- `backend=ibm_fez` and a real `job_id` = it executed on the QPU.
- The end-of-run **EXECUTION PROOF** block prints total jobs/circuits and sample
  job ids. Paste any id into https://quantum.ibm.com/jobs to confirm.
- `comparison_results.json → meta.execution` stores the same proof, and each
  quantum model has `ran_on_backend: true/false`.

## Important: cost & time on real hardware
A full run (120 train samples × `MAX_ITER` COBYLA steps, plus gradients) submits
**thousands** of circuits — each `.run()` is a queued IBM job. On the Open plan
that is slow and will burn your monthly QPU minutes fast.

So when a real QPU is detected, a **HARDWARE GUARD** automatically:
- subsamples train→`HW_MAX_TRAIN` (12), test→`HW_MAX_TEST` (6),
- caps iterations to `HW_MAX_ITER` (8).

Disable with `HW_GUARD=false` (only if you accept long queues and QPU-time cost).
For development, keep `BACKEND=LOCAL` (statevector) and switch to IBM for a final
demonstration run.

## Security note
Your API token is stored in `.env`. Treat it as a secret (don't commit/share it).
Since it was included in an uploaded archive, consider **regenerating** it on the
IBM dashboard.

## Troubleshooting

**`TranspilerError: Invalid plugin name ibm_dynamic_circuits for stage translation`**
A version mismatch: recent `qiskit-ibm-runtime` advertises the `ibm_dynamic_circuits`
translation plugin, which is not registered under qiskit 1.x. The pass-manager builder
now detects this and automatically falls back to the standard `translator` stage
(safe here — our circuits have no dynamic/control-flow features). You will see:
```
WARNING | Default transpiler stage failed (...ibm_dynamic_circuits...).
WARNING | Falling back to translation_method='translator' (static circuits — safe).
```
If you prefer to remove the fallback entirely, align versions instead — either
`pip install "qiskit>=2" "qiskit-ibm-runtime>=0.40"` or pin an older runtime that
matches qiskit 1.x.

---

## `'Sampler job failed!'` on a real QPU (QSVM/VQC/QNN)

This generic message comes from qiskit-machine-learning when the **remote IBM job**
fails — it hides the real reason. The code now does three things to handle it:

1. **Reveals the real IBM error.** The sampler wrapper catches the failure and logs
   `job.error_message()` + status before re-raising, e.g.:
   ```
   ✗ IBM job FAILED | id=d8m8m4bqv2lc... | status=JobStatus.ERROR
     IBM error_message: <the actual device/validation reason>
   ```
   Re-run and read that line — it tells you exactly why the QPU rejected the job.

2. **Auto-retries transient failures.** `IBM_MAX_RETRIES` (default 2) resubmits a
   failed job — most live-hardware failures are transient (queue/calibration).

3. **Robust metrics.** `confusion_matrix`/`classification_report` now pin
   `labels=[0,1,2]`, and the hardware-guard subsampling is **stratified**, so a tiny
   hardware test set that happens to miss a class no longer crashes a model whose
   quantum job actually succeeded.

### Recommended: noise-sim mode for reproducible thesis runs
If live hardware keeps failing or queueing, set in `.env`:
```
IBM_NOISE_SIM=true
```
This connects to IBM, takes the **real device's noise model + basis gates**, but runs
the circuits **locally on Aer** at NUM_QUBITS width (no 156-qubit blow-up, no remote
job failures, no QPU-time cost). Results are hardware-representative and fully
reproducible — a defensible methodology for a thesis. Set it back to `false` for a
final on-hardware demonstration run.

---

## `Error code 3211; ... Cannot bind following parameters not present in expression`

This is a **server-side parameter-binding failure**: qiskit-machine-learning transpiles
a *parametric* circuit and asks IBM to bind the parameter values server-side, and the
binding fails (the root cause is the same qiskit 1.x ↔ qiskit-ibm-runtime 0.44 version
mismatch behind the `ibm_dynamic_circuits` issue — the serialized parameter format does
not match).

**Fix (implemented):** on the IBM hardware path the sampler wrapper now binds parameter
values **locally** and transpiles to ISA **before** submission, so only
**parameter-free** circuits are sent to the QPU. With no parameters left to bind on the
server, Error 3211 cannot occur. (Verified: across QSVM/QCL/VQC/QNN, 0 parameter-bearing
circuits are submitted.) You will see `params=0` in the per-job log lines.

If you would rather fix it by aligning versions, either upgrade to
`qiskit>=2` (the combo qiskit-ibm-runtime 0.44 officially supports) or pin an older
runtime that matches qiskit 1.x. The local-bind fix above works regardless and needs no
environment change.

---

## New in this version: execution modes, non-variational tests, versioned logs

### Execution modes (physical QPU vs simulation)
Pick the configuration from the command line (overrides `.env`):
```
python main.py --mode simulator   # local quantum simulation (no IBM, no QPU time)
python main.py --mode physical     # real IBM QPU (BACKEND=IBM, IBM_NOISE_SIM=false)
python main.py --mode noise        # IBM device noise model, executed locally on Aer
```
Without `--mode`, the `.env` values are used.

### Non-variational quantum tests (performance-cost comparison)
Three of the quantum models are **variational** (trainable ansatz + COBYLA loop): QCL, VQC, QNN.
This version adds **non-variational** quantum classifiers so you can measure the cost of the
variational approach:
- **QFC** — Quantum Fidelity Classifier: fixed feature map, classify by nearest class by
  quantum-state fidelity. No trainable circuit, no optimization loop.
- **QFE-LR** — Quantum Embedding + Logistic Regression: a fixed feature-map circuit produces
  measurement-probability features; only a classical linear head is trained.

Every quantum result carries an `is_variational` flag. The run prints a
`VARIATIONAL vs NON-VARIATIONAL` time summary, and a new chart
`07_variational_vs_nonvariational.png` compares total compute time (train + inference) of the
two groups. Note: on a noiseless simulator the kernel/fidelity models (QSVM, QFC) dominate
the time (O(n²) fidelities), while on a real QPU the variational loop (MAX_ITER × samples
sequential jobs) becomes the expensive part.

### Versioned logs and outputs (one per run)
Every run creates its own timestamped folder and log:
```
outputs/run_<YYYY-MM-DD_HH-MM-SS>/quantum_run_<YYYY-MM-DD_HH-MM-SS>.log
outputs/run_<YYYY-MM-DD_HH-MM-SS>/comparison/<charts + comparison_results.json>
```
Nothing is overwritten between runs, so you can compare runs by date/time.
