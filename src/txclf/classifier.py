"""Supervised step: Logistic Regression on the SVD features."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from txclf.config import ClassifierConfig


def train_classifier(
    Z: np.ndarray, y: np.ndarray, weights: np.ndarray | None, cfg: ClassifierConfig, seed: int
) -> LogisticRegression:
    clf = LogisticRegression(
        C=cfg.C, max_iter=cfg.max_iter, class_weight=cfg.class_weight, random_state=seed
    )
    clf.fit(Z, y, sample_weight=weights)
    return clf
