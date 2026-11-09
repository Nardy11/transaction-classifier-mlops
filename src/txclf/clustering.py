"""KMeans pseudo-labels and automatic cluster names.

The dataset has no category column. Clusters of similar descriptions become
the label set; each cluster is named after the heaviest TF-IDF terms of its
centroid (back-projected through the SVD), so names are reproducible from the
config and seed instead of being typed in by hand after every run.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

from txclf.config import ClusteringConfig


def fit_kmeans(Z: np.ndarray, weights: np.ndarray, cfg: ClusteringConfig, seed: int) -> KMeans:
    km = KMeans(n_clusters=cfg.n_clusters, n_init=cfg.n_init, random_state=seed)
    km.fit(Z, sample_weight=weights)
    return km


def name_clusters(km: KMeans, text_pipeline: Pipeline, n_terms: int = 3) -> list[str]:
    tfidf = text_pipeline.named_steps["tfidf"]
    svd = text_pipeline.named_steps["svd"]
    vocab = np.asarray(tfidf.get_feature_names_out())
    term_weights = km.cluster_centers_ @ svd.components_  # (k, V)
    names: list[str] = []
    for row in term_weights:
        picked: list[str] = []
        for idx in np.argsort(row)[::-1]:
            term = vocab[idx]
            words = term.split()
            # skip a term whose words are all already in the name
            if all(any(w in p.split() for p in picked) for w in words):
                continue
            picked.append(term)
            if len(picked) == n_terms:
                break
        names.append(" / ".join(picked))
    seen: dict[str, int] = {}
    unique = []
    for name in names:  # guarantee unique names
        seen[name] = seen.get(name, 0) + 1
        unique.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    return unique
