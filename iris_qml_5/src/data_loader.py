"""
data_loader.py — Iris dataset preprocessing for Quantum ML.

Pipeline:
  1. Load sklearn Iris (150 samples, 4 features, 3 classes)
  2. StandardScaler normalisation
  3. PCA reduction → NUM_QUBITS dimensions (quantum register size)
  4. Angle encoding rescaling → [0, π]
  5. Train/test stratified split
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# Fixed import
from .config import Config

CLASS_NAMES = ["Setosa", "Versicolor", "Virginica"]
FEATURE_NAMES_FULL = [
    "Sepal Length (cm)",
    "Sepal Width (cm)",
    "Petal Length (cm)",
    "Petal Width (cm)",
]


def load_iris_data() -> dict:
    """
    Load and preprocess Iris data for VQC training.

    Returns dict with split arrays, PCA object, scalers, and metadata.
    """
    iris = load_iris()
    X_raw = iris.data.astype(np.float64)   # (150, 4)
    y     = iris.target.astype(np.int64)   # (150,)

    # ── 1. Standardise ──────────────────────────────────────────
    std_scaler = StandardScaler()
    X_std = std_scaler.fit_transform(X_raw)

    # ── 2. PCA → NUM_QUBITS ─────────────────────────────────────
    n_components = Config.NUM_QUBITS
    pca = PCA(n_components=n_components, random_state=Config.RANDOM_SEED)
    X_pca = pca.fit_transform(X_std)        # (150, NUM_QUBITS)

    # ── 3. Angle encoding: rescale to [0, π] ────────────────────
    angle_scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = angle_scaler.fit_transform(X_pca)

    # ── 4. Train / test split ───────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=1.0 - Config.TRAIN_SPLIT,
        random_state=Config.RANDOM_SEED,
        stratify=y,
    )

    pca_explained = pca.explained_variance_ratio_
    feature_names_pca = [f"PC{i+1} ({v*100:.1f}%)" for i, v in enumerate(pca_explained)]

    return {
        # Raw (unprocessed) for EDA plots
        "X_raw":          X_raw,
        "y_full":         y,
        # PCA-scaled data
        "X_full":         X_scaled,
        "X_train":        X_train,
        "X_test":         X_test,
        "y_train":        y_train,
        "y_test":         y_test,
        # Metadata
        "class_names":    CLASS_NAMES,
        "feature_names":  FEATURE_NAMES_FULL,
        "feature_names_pca": feature_names_pca,
        "pca":            pca,
        "pca_explained":  pca_explained,
        "std_scaler":     std_scaler,
        "angle_scaler":   angle_scaler,
        "num_qubits":     n_components,
        "num_classes":    len(np.unique(y)),
    }