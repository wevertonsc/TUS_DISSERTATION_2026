"""
classical_models.py — Classical machine learning algorithms for comparison.
"""

from __future__ import annotations
import time
from dataclasses import dataclass

import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

from .config import Config


@dataclass
class ClassicalModelResult:
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
    hyperparams: dict


class ClassicalModelWrapper:
    def __init__(self, model, name: str, algorithm_type: str, hyperparams: dict = None):
        self.model = model
        self.name = name
        self.algorithm_type = algorithm_type
        self.hyperparams = hyperparams or {}

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        t0 = time.perf_counter()
        self.model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        t_inf = time.perf_counter()
        y_pred = self.model.predict(X_test)
        inference_time_ms = (time.perf_counter() - t_inf) * 1000

        y_true = y_test.astype(int)
        y_pred = y_pred.astype(int)

        return ClassicalModelResult(
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
            hyperparams=self.hyperparams,
        )


def get_classical_models() -> list[ClassicalModelWrapper]:
    models = []

    models.append(ClassicalModelWrapper(
        model=SVC(kernel="rbf", C=1.0, gamma="scale", random_state=Config.RANDOM_SEED),
        name="SVM (RBF Kernel)",
        algorithm_type="SVM",
        hyperparams={"kernel": "rbf", "C": 1.0, "gamma": "scale"},
    ))

    models.append(ClassicalModelWrapper(
        model=KNeighborsClassifier(n_neighbors=5, weights="uniform", metric="euclidean"),
        name="k-NN (k=5)",
        algorithm_type="kNN",
        hyperparams={"n_neighbors": 5, "weights": "uniform", "metric": "euclidean"},
    ))

    models.append(ClassicalModelWrapper(
        model=LogisticRegression(
            max_iter=Config.MAX_ITER,
            random_state=Config.RANDOM_SEED,
            C=1.0,
            solver='lbfgs',
        ),
        name="Logistic Regression",
        algorithm_type="LogReg",
        hyperparams={"max_iter": Config.MAX_ITER, "C": 1.0, "solver": "lbfgs"},
    ))

    models.append(ClassicalModelWrapper(
        model=MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=Config.MAX_ITER,
            random_state=Config.RANDOM_SEED,
            early_stopping=True,
        ),
        name="MLP (64-32)",
        algorithm_type="MLP",
        hyperparams={"hidden_layers": [64, 32], "activation": "relu", "max_iter": Config.MAX_ITER},
    ))

    return models