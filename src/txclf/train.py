"""Training run: data -> features -> KMeans pseudo-labels -> classifier -> eval.

Logged to MLflow (params, metrics, figures, model). Writes the joblib artifact
and ``reports/metrics.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from txclf import __version__
from txclf.classifier import train_classifier
from txclf.clustering import fit_kmeans, name_clusters
from txclf.config import Config
from txclf.data import clean, load_raw, unique_descriptions
from txclf.drift import build_reference
from txclf.evaluate import (
    cluster_metrics,
    consistency_metrics,
    hand_label_metrics,
    plot_clusters,
    plot_confusion,
)
from txclf.features import build_text_pipeline
from txclf.model import TxModel


def train(cfg: Config, log_mlflow: bool = True, figures: bool = True) -> tuple[TxModel, dict]:
    t0 = time.time()
    raw = load_raw(cfg.data)
    df = clean(raw, remove_outliers=cfg.data.remove_outliers)
    uniq = unique_descriptions(df)
    texts, weights = uniq["Description"].tolist(), uniq["count"].to_numpy(float)

    # 1. features + pseudo-labels, fitted on every unique description
    text_pipe = build_text_pipeline(cfg.features, cfg.seed)
    Z = text_pipe.fit_transform(texts)
    km = fit_kmeans(Z, weights, cfg.clustering, cfg.seed)
    labels = km.labels_.astype(int)
    names = name_clusters(km, text_pipe, cfg.clustering.name_terms)

    # 2. classifier evaluated on descriptions it has never seen
    idx = np.arange(len(texts))
    counts = np.bincount(labels)
    stratify = labels if counts.min() >= 2 else None
    tr, te = train_test_split(
        idx, test_size=cfg.eval.test_size, random_state=cfg.seed, stratify=stratify
    )
    clf = train_classifier(Z[tr], labels[tr], weights[tr], cfg.classifier, cfg.seed)
    pred_te, pred_tr = clf.predict(Z[te]), clf.predict(Z[tr])
    rng = np.random.default_rng(cfg.seed)
    shuffled = train_classifier(
        Z[tr], rng.permutation(labels[tr]), weights[tr], cfg.classifier, cfg.seed
    )

    metrics: dict[str, Any] = {
        "rows_raw": len(raw),
        "rows_clean": len(df),
        "unique_descriptions": len(texts),
        "vocabulary_size": len(text_pipe.named_steps["tfidf"].vocabulary_),
        "svd_explained_variance": float(
            text_pipe.named_steps["svd"].explained_variance_ratio_.sum()
        ),
        **cluster_metrics(Z, labels, weights, cfg.seed),
        **consistency_metrics(labels[tr], pred_tr, weights[tr], "seen_"),
        **consistency_metrics(labels[te], pred_te, weights[te], "heldout_"),
        "heldout_shuffled_label_accuracy": float(np.mean(shuffled.predict(Z[te]) == labels[te])),
    }

    # 3. final classifier on all descriptions -> shipped model
    final_clf = train_classifier(Z, labels, weights, cfg.classifier, cfg.seed)
    pipeline = Pipeline([*text_pipe.steps, ("clf", final_clf)])
    model = TxModel(pipeline=pipeline, kmeans=km, categories=names)
    model.reference = build_reference(model, texts, weights)
    if cfg.eval.hand_labels and Path(cfg.eval.hand_labels).exists():
        metrics.update(hand_label_metrics(model, cfg.eval.hand_labels))
    metrics["train_seconds"] = round(time.time() - t0, 1)
    model.metadata = {
        "txclf_version": __version__,
        "sklearn_version": sklearn.__version__,
        "config": cfg.flat(),
        "metrics": metrics,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    model_path = model.save(cfg.output.model_path)
    reports = Path(cfg.output.reports_dir)
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (reports / "categories.json").write_text(
        json.dumps(
            [
                {"id": i, "name": n, "rows": int(c)}
                for i, (n, c) in enumerate(
                    zip(names, np.bincount(labels, weights=weights), strict=True)
                )
            ],
            indent=2,
        )
        + "\n"
    )
    figs = []
    if figures:
        figs = [reports / "figures" / "confusion_matrix.png", reports / "figures" / "clusters.png"]
        plot_confusion(labels[te], pred_te, names, figs[0])
        plot_clusters(Z, labels, weights, names, figs[1])

    if log_mlflow:
        _log_mlflow(cfg, metrics, model, model_path, figs, reports)
    return model, metrics


def _log_mlflow(cfg, metrics, model, model_path, figs, reports) -> None:
    import mlflow

    # local SQLite store + artifact folder, both inside mlruns/ (git-ignored)
    root = Path(cfg.output.mlflow_uri).resolve()
    root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(root / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(cfg.output.experiment) is None:
        mlflow.create_experiment(
            cfg.output.experiment, artifact_location=(root / "artifacts").as_uri()
        )
    mlflow.set_experiment(cfg.output.experiment)
    with mlflow.start_run():
        mlflow.log_params(cfg.flat())
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, int | float)})
        mlflow.log_artifact(str(model_path), artifact_path="joblib")
        mlflow.log_artifact(str(reports / "metrics.json"))
        mlflow.log_artifact(str(reports / "categories.json"))
        for f in figs:
            mlflow.log_artifact(str(f), artifact_path="figures")
        mlflow.sklearn.log_model(
            model.pipeline, name="model", skops_trusted_types=["txclf.data.clean_many"]
        )
