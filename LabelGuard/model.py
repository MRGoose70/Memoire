"""
Module de gestion du modèle.
Ajout : Hamming-loss et F1-macro.
"""

from sklearn.multioutput import ClassifierChain
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    f1_score,
)
from sklearn.model_selection import cross_val_score, KFold

import numpy as np
from typing import Tuple


# ------------------------------------------------------------------
def bootstrap_ci_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_iter: int = 1000,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Intervalle de confiance (bootstrap) pour le F1-macro.
    Retourne (borne_inf, borne_sup) à (1-alpha)*100 %.
    """
    stats = []
    idx = np.arange(len(y_true))
    for _ in range(n_iter):
        sample = resample(idx, replace=True)
        stats.append(f1_score(y_true[sample], y_pred[sample], average="macro"))
    lower = np.percentile(stats, 100 * (alpha / 2))
    upper = np.percentile(stats, 100 * (1 - alpha / 2))
    return lower, upper
# ------------------------------------------------------------------


def train_classifier_chain(X_train: np.ndarray, y_train: np.ndarray) -> ClassifierChain:
    base_estimator = LogisticRegression(max_iter=1000)
    model = ClassifierChain(
        base_estimator=base_estimator, order="random", random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: ClassifierChain, X_test: np.ndarray, y_test: np.ndarray, label_cols: list
) -> tuple:
    """
    Renvoie : accuracy, hamming-loss, f1-macro, (ci_low, ci_high), prédictions
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    h_loss = hamming_loss(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    ci_low, ci_high = bootstrap_ci_f1(y_test, y_pred)

    return acc, h_loss, f1_macro, (ci_low, ci_high), y_pred


def cross_validate_model(X: np.ndarray, y: np.ndarray) -> float:
    """
    Validation croisée 5-fold sur la F1-macro (plus pertinente que l'accuracy).
    """
    base_estimator = LogisticRegression(max_iter=1000)
    model = ClassifierChain(
        base_estimator=base_estimator, order="random", random_state=42
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring="f1_macro")
    return cv_scores.mean()
