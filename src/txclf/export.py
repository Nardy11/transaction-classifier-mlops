"""Export the model as compact JSON for the in-browser demo (docs/) and the Space.

TF-IDF -> SVD -> LogReg is linear up to the softmax, so the SVD components and
the LogReg weights fold into one matrix W = coef @ components (classes x vocab):

    logits = tfidf(x) @ W.T + intercept,   proba = softmax(logits)

W is stored as int16 (one scale for the whole matrix, base64). Everything else
(cleaning, tokens, stop words, bigrams, idf, l2 norm) is reproduced exactly.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from txclf.model import TxModel


def export_web(model: TxModel, path: str | Path) -> dict:
    tfidf, svd, clf = (model.pipeline.named_steps[s] for s in ("tfidf", "svd", "clf"))
    vocab = tfidf.get_feature_names_out().tolist()
    W = clf.coef_ @ svd.components_  # (n_classes, V)
    scale = float(np.abs(W).max() / 32767)
    q = np.round(W / scale).astype("<i2")
    payload = {
        "format": "txclf-web-1",
        "version": model.metadata.get("txclf_version"),
        "trained_at": model.metadata.get("trained_at"),
        "ngram_max": tfidf.ngram_range[1],
        "stop_words": sorted(ENGLISH_STOP_WORDS),
        "vocabulary": vocab,
        "idf": [round(float(v), 6) for v in tfidf.idf_],
        "categories": [model.categories[int(c)] for c in clf.classes_],
        "intercept": [round(float(v), 6) for v in clf.intercept_],
        "weights_shape": list(q.shape),
        "weights_scale": scale,
        "weights_int16_b64": base64.b64encode(q.tobytes()).decode("ascii"),
        "metrics": {
            k: v for k, v in model.metadata.get("metrics", {}).items() if isinstance(v, int | float)
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return {"path": str(path), "bytes": path.stat().st_size, "vocab": len(vocab)}
