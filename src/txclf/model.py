"""The saved artifact and the predict function."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

from txclf.data import clean_description


@dataclass
class TxModel:
    pipeline: Pipeline  # clean -> tfidf -> svd -> clf
    kmeans: KMeans  # regenerates pseudo-labels for `evaluate`
    categories: list[str]  # index = cluster id = classifier class
    reference: dict[str, Any] = field(default_factory=dict)  # drift baseline
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text_pipeline(self) -> Pipeline:
        return self.pipeline[:-1]

    @property
    def tfidf(self):
        return self.pipeline.named_steps["tfidf"]

    @property
    def clf(self):
        return self.pipeline.named_steps["clf"]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(list(texts))

    def pseudo_labels(self, texts: list[str]) -> np.ndarray:
        return self.kmeans.predict(self.text_pipeline.transform(list(texts)))

    def predict(self, texts: list[str], top_k: int = 3) -> list[dict[str, Any]]:
        texts = list(texts)
        proba = self.predict_proba(texts)
        known = np.asarray(
            self.tfidf.transform([clean_description(t) for t in texts]).getnnz(axis=1)
        )
        classes = self.clf.classes_
        out = []
        for text, row, n_known in zip(texts, proba, known, strict=True):
            order = np.argsort(row)[::-1][:top_k]
            top = [
                {"category": self.categories[int(classes[i])], "probability": float(row[i])}
                for i in order
            ]
            out.append(
                {
                    "description": text,
                    "category": top[0]["category"],
                    "probability": top[0]["probability"],
                    "top_k": top,
                    "known_terms": int(n_known),
                }
            )
        return out

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        return path


def load_model(path: str | Path) -> TxModel:
    """Load a joblib artifact. Only load files you trust: joblib uses pickle."""
    model = joblib.load(path)
    if not isinstance(model, TxModel):
        raise TypeError(f"{path} is not a txclf model")
    return model
