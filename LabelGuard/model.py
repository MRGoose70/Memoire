"""
Module de gestion du modèle.
Contient les fonctions pour entraîner le modèle ClassifierChain et pour l'évaluer.
"""

from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

def train_classifier_chain(X_train: np.ndarray, y_train: np.ndarray) -> ClassifierChain:
    """
    Entraîne un modèle ClassifierChain avec LogisticRegression comme estimateur de base.
    """
    base_estimator = LogisticRegression(max_iter=1000)
    model = ClassifierChain(base_estimator=base_estimator, order='random', random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: ClassifierChain, X_test: np.ndarray, y_test: np.ndarray, label_cols: list) -> tuple:
    """
    Évalue le modèle sur le jeu de test et renvoie la précision et les prédictions.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc, y_pred

def cross_validate_model(X: np.ndarray, y: np.ndarray) -> float:
    """
    Effectue une validation croisée 5-fold et renvoie la précision moyenne.
    """
    base_estimator = LogisticRegression(max_iter=1000)
    model = ClassifierChain(base_estimator=base_estimator, order='random', random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    return cv_scores.mean()
