## Comparison Table: Classical vs Quantum Algorithms

Below is the complete comparison table of algorithms implemented in the application:

---

### Table 1: Algorithm Overview

| # | Classical Algorithm | Quantum Algorithm | Status in Code | Complexity | Theoretical Basis |
|---|--------------------|-------------------|----------------|------------|--------------------|
| 1 | **SVM (RBF Kernel)** | **QSVM (Quantum Kernel)** | Implemented | Classical: O(n³) / Quantum: O(log n) | Both use kernel trick. QSVM computes kernel in quantum Hilbert space |
| 2 | **k-NN (k=5)** | **QkNN (Quantum Distance)** | Implemented | Classical: O(n) / Quantum: O(log n) | Distance/fidelity-based classification |
| 3 | **Logistic Regression** | **QCL (Quantum Circuit Learning)** | Implemented | Classical: O(n) / Quantum: O(poly(n)) | Variational circuit for regression/classification |
| 4 | **MLP (64-32 neurons)** | **VQC (Variational Quantum Classifier)** | Implemented | Classical: O(n²) / Quantum: O(poly(n)) | Parameter optimization analogous to neural networks |
| 5 | *Extension* | **QNN (Quantum Neural Network)** | Implemented | Classical: O(n²) / Quantum: O(poly(n)) | Quantum neural network with SamplerQNN |

---

### Table 2: Hyperparameters and Configurations

| Algorithm | Hyperparameters | Configured Value | Equivalence |
|-----------|-----------------|------------------|-------------|
| **SVM** | Kernel, C, gamma | rbf, 1.0, scale | Non-linear kernel maps to high-dimensional space |
| **QSVM** | Feature Map, Kernel | ZZFeatureMap, FidelityKernel | Kernel computed via quantum circuit |
| **k-NN** | n_neighbors, metric | 5, euclidean | Euclidean distance in feature space |
| **QkNN** | n_neighbors, fidelity | 3, gaussian kernel | Quantum fidelity as similarity proxy |
| **Logistic Regression** | solver, max_iter, C | lbfgs, 150, 1.0 | Multiclass linear regression (OVR) |
| **QCL** | Feature Map, Ansatz, Optimizer | ZZFeatureMap, RealAmplitudes, COBYLA | Variational circuit for classification |
| **MLP** | hidden_layers, activation | (64,32), relu | Feed-forward multi-layer neural network |
| **VQC** | Feature Map, Ansatz, Optimizer | ZZFeatureMap, RealAmplitudes, COBYLA | Variational classifier equivalent to MLP |
| **QNN** | Ansatz, Optimizer, output_shape | EfficientSU2, COBYLA, 2 | Quantum neural network with parity interpretation |

---

### Table 3: Evaluation Metrics Collected

| Metric | Classical | Quantum | Description |
|--------|-----------|---------|-------------|
| **Accuracy** | Yes | Yes | Proportion of correct predictions (0-1) |
| **F1 Score (macro)** | Yes | Yes | Harmonic mean of precision and recall |
| **Precision (macro)** | Yes | Yes | True positives / total predicted positives |
| **Recall (macro)** | Yes | Yes | True positives / total actual positives |
| **Training Time (s)** | Yes | Yes | Training time in seconds |
| **Inference Time (ms)** | Yes | Yes | Prediction time in milliseconds |
| **Confusion Matrix** | Yes | Yes | 3x3 confusion matrix |
| **Convergence Curve** | No | Yes | Objective function evolution per iteration |
| **Circuit Depth** | No | Yes | Quantum circuit depth |
| **Num Parameters** | No | Yes | Number of variational parameters |
| **Num Qubits** | No | Yes | Number of qubits used |

---

### Table 4: Computational Complexity

| Algorithm | Training | Inference | Space | Observation |
|-----------|----------|-----------|-------|-------------|
| **SVM** | O(n³) | O(k·d) | O(n²) | n=samples, d=dimensions, k=support vectors |
| **QSVM** | O(log n) * | O(log n) * | O(n) | *Estimated, depends on circuit |
| **k-NN** | O(1) | O(n·d) | O(n·d) | n=training size, d=dimensions |
| **QkNN** | O(1) | O(n·d) | O(n·d) | Similar to classical, with fidelity computation |
| **LogReg** | O(n·d) | O(d) | O(d) | d=dimensions |
| **QCL** | O(poly(n)) | O(poly(n)) | O(n) | Depends on number of iterations |
| **MLP** | O(n·h·i) | O(h) | O(h) | h=neurons, i=iterations |
| **VQC** | O(i·S) | O(S) | O(n) | S=shots per circuit, i=iterations |
| **QNN** | O(i·S·p) | O(S·p) | O(p) | p=parameters, S=shots |

---

### Table 5: Iris Dataset Specifications

| Property | Value | Description |
|----------|-------|-------------|
| **Samples** | 150 | Total samples |
| **Original Features** | 4 | Sepal Length, Sepal Width, Petal Length, Petal Width |
| **Quantum Features (PCA)** | 2 | Reduced to NUM_QUBITS |
| **Classes** | 3 | Setosa, Versicolor, Virginica |
| **Train/Test Split** | 120/30 | 80% training, 20% testing |
| **Preprocessing** | PCA + Angle Encoding | Scale to [0, π] |
| **Explained Variance (PC1)** | ~92.5% | First principal component |
| **Explained Variance (PC2)** | ~5.3% | Second principal component |

---

### Table 6: Quantum Circuits Implemented

| Component | Type | Parameters | Depth | Gates |
|-----------|------|------------|-------|-------|
| **ZZFeatureMap** | Feature Map | 2 qubits, reps=2 | Variable | RZ, CNOT, RZZ |
| **PauliFeatureMap** | Feature Map | 2 qubits, reps=2 | Variable | RZ, RX, CNOT |
| **RealAmplitudes** | Ansatz | 2 qubits, reps=3 | ~30 | RY, CX |
| **EfficientSU2** | Ansatz | 2 qubits, reps=3 | ~24 | RY, RZ, CX |
| **TwoLocal** | Ansatz | 2 qubits, reps=3 | ~18 | RY, RZ, CX |

---

### Table 7: Available Optimizers

| Optimizer | Type | Classical Use | Quantum Use | Max Iterations |
|-----------|------|---------------|-------------|----------------|
| **COBYLA** | Derivative-free | No | Yes (VQC/QCL/QNN) | 150 |
| **SPSA** | Gradient approximation | No | Yes (VQC) | 150 |
| **L-BFGS-B** | Quasi-Newton | No | Yes (VQC) | 150 |
| **ADAM** | Gradient descent | Yes (MLP) | Yes (Optional) | 150 |
| **lbfgs** | Quasi-Newton | Yes (LogReg) | No | 150 |

---

### Table 8: Expected Results (Benchmark)

| Metric | SVM | QSVM | k-NN | QkNN | LogReg | QCL | MLP | VQC |
|--------|-----|------|------|------|--------|-----|-----|-----|
| **Expected Accuracy** | 0.96-1.00 | 0.90-0.98 | 0.93-1.00 | 0.85-0.95 | 0.90-0.97 | 0.85-0.95 | 0.95-1.00 | 0.85-0.95 |
| **Training Time (s)** | 0.01-0.05 | 2-10 | <0.01 | 1-5 | 0.01-0.03 | 5-20 | 0.5-2 | 5-20 |
| **Inference (ms)** | <1 | 10-100 | <1 | 10-100 | <1 | 10-50 | <1 | 10-50 |
| **Overfit Risk** | Low | Medium | Low | Medium | Low | High | Medium | High |

*Note: Values based on statevector simulation. Real hardware may have different performance.*

---

### Table 9: Classical to Quantum Concept Mapping

| Classical Concept | Classical Implementation | Quantum Concept | Quantum Implementation |
|-------------------|-------------------------|-----------------|------------------------|
| **Kernel Function** | RBF, Polynomial | Fidelity Kernel | `FidelityQuantumKernel` |
| **Distance Metric** | Euclidean, Manhattan | Fidelity / Overlap | Swap Test / Loschmidt Echo |
| **Weight Optimization** | Gradient Descent | Parameter Shift Rule | `ParameterShift` gradient |
| **Activation Function** | ReLU, Sigmoid | Interpretation | Parity, Threshold |
| **Loss Function** | Cross-Entropy, MSE | Objective Function | Negative Log-Likelihood |
| **Regularization** | L1, L2, Dropout | Circuit Structure | Ansatz depth, Repetitions |
| **Feature Engineering** | PCA, Normalization | Feature Map | ZZFeatureMap, PauliFeatureMap |

---

### Table 10: System Architecture

| Component | File | Responsibility | Lines of Code |
|-----------|------|----------------|----------------|
| **Configuration** | `config.py` | Environment variable management | ~50 |
| **Data Loading** | `data_loader.py` | Load and preprocess Iris dataset | ~80 |
| **Backend** | `backend_factory.py` | Create quantum samplers | ~60 |
| **Circuits** | `circuits.py` | Feature maps and ansatze | ~100 |
| **Classical Models** | `classical_models.py` | SVM, k-NN, LogReg, MLP | ~120 |
| **Quantum Models** | `quantum_models.py` | QSVM, QkNN, QCL, VQC, QNN | ~250 |
| **Comparator** | `comparator.py` | Execute and compare models | ~180 |
| **Visualization** | `visualizer_compare.py` | Generate comparison charts | ~300 |
| **Main Entry** | `main.py` | Orchestrate execution | ~150 |

---

### Conclusion

The application implements **5 classical-quantum algorithm pairs**, covering:

1. **Kernel Methods** (SVM ↔ QSVM)
2. **Instance-based Learning** (k-NN ↔ QkNN)  
3. **Linear Models** (Logistic Regression ↔ QCL)
4. **Neural Networks** (MLP ↔ VQC)
5. **Advanced Quantum Models** (Bonus: QNN)

The comparison is fair because both sides operate on the same preprocessed data and are evaluated using identical metrics, enabling objective analysis of the advantages and disadvantages of each approach on the Iris dataset.