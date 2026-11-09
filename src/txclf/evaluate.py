"""Metrics and figures.

Three different questions, kept apart on purpose:

1. Cluster quality (silhouette, sizes): are the pseudo-labels coherent groups?
2. Consistency: how well does the classifier reproduce the KMeans pseudo-labels,
   on descriptions it saw and on descriptions it never saw?
3. Hand-labelled check: on a small set labelled by reading the text, how often
   is the predicted category the one a person would pick?
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, silhouette_score


def cluster_metrics(Z: np.ndarray, labels: np.ndarray, weights: np.ndarray, seed: int) -> dict:
    n, k = len(labels), len(set(labels))
    sil = (
        float(silhouette_score(Z, labels, sample_size=min(n, 5000), random_state=seed))
        if 1 < k < n
        else float("nan")
    )
    sizes = np.bincount(labels, weights=weights)
    return {
        "silhouette": sil,
        "largest_cluster_row_share": float(sizes.max() / sizes.sum()),
        "smallest_cluster_rows": int(sizes.min()),
    }


def consistency_metrics(y_true, y_pred, weights=None, prefix: str = "") -> dict[str, float]:
    return {
        f"{prefix}accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f"{prefix}row_weighted_accuracy": float(
            accuracy_score(y_true, y_pred, sample_weight=weights)
        ),
    }


def hand_label_metrics(model, path: str | Path) -> dict[str, Any]:
    """CSV with columns description,label. label 'none' = no category fits."""
    df = pd.read_csv(path)
    preds = model.predict(df["description"].tolist(), top_k=3)
    top1 = [p["category"] for p in preds]
    top3 = [[t["category"] for t in p["top_k"]] for p in preds]
    labels = df["label"].astype(str).tolist()
    fits = np.array([lab in set(model.categories) for lab in labels])
    unknown = sorted({lab for lab, f in zip(labels, fits, strict=True) if not f} - {"none"})
    correct = np.array([p == lab for p, lab in zip(top1, labels, strict=True)])
    in_top3 = np.array([lab in t for t, lab in zip(top3, labels, strict=True)])
    nan = float("nan")
    return {
        "hand_n": len(df),
        "hand_n_with_fitting_category": int(fits.sum()),
        "hand_accuracy_all": float(correct.mean()) if len(df) else nan,
        "hand_accuracy_where_category_fits": float(correct[fits].mean()) if fits.any() else nan,
        "hand_top3_where_category_fits": float(in_top3[fits].mean()) if fits.any() else nan,
        "hand_unknown_labels": unknown,
    }


def plot_confusion(y_true, y_pred, names: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = len(names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(k))).astype(float)
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    size = max(6, k * 0.2)
    fig, ax = plt.subplots(figsize=(size + 2.5, size + 1))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    short = [n if len(n) <= 34 else n[:33] + "..." for n in names]
    ax.set_xticks(range(k), short, rotation=90, fontsize=6)
    ax.set_yticks(range(k), short, fontsize=6)
    ax.set_xlabel("predicted")
    ax.set_ylabel("pseudo-label (KMeans)")
    ax.set_title("Held-out descriptions: confusion matrix (row-normalised)")
    fig.colorbar(im, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_clusters(
    Z: np.ndarray, labels: np.ndarray, weights: np.ndarray, names, path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    xy = PCA(n_components=2, random_state=0).fit_transform(Z)
    k = len(names)
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(15, max(6.5, k * 0.16)), gridspec_kw={"width_ratios": [1.1, 1]}
    )
    a.scatter(xy[:, 0], xy[:, 1], c=labels, cmap="nipy_spectral", s=6, alpha=0.7, linewidths=0)
    a.set_title("Unique descriptions in SVD space (PCA to 2-D), coloured by cluster")
    a.set_xticks([])
    a.set_yticks([])
    sizes = np.bincount(labels, weights=weights, minlength=k)
    order = np.argsort(sizes)
    b.barh(range(k), sizes[order], color="#3b6fb6")
    b.set_yticks(range(k), [names[i][:40] for i in order], fontsize=6)
    b.set_xlabel("transactions (rows)")
    b.set_title("Cluster sizes")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
