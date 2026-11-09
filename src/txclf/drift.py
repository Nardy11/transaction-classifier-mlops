"""Input drift check for a new batch of transactions.

Compares a batch against the training reference stored inside the model:

- vocabulary: share of (unigram) tokens the TF-IDF vocabulary has never seen,
  share of rows with no known term at all, and the Jensen-Shannon distance
  between the batch's term distribution and the training one;
- predictions: Population Stability Index (PSI) and JS distance between the
  predicted category distribution of the batch and of the training data.

Thresholds are rules of thumb (PSI > 0.2 is the usual "significant shift").
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from scipy.spatial.distance import jensenshannon

from txclf.data import clean_description
from txclf.features import unigram_tokens

DEFAULT_THRESHOLDS = {
    "oov_token_rate_increase": 0.10,
    "empty_row_rate": 0.05,
    "term_js_distance": 0.30,
    "category_psi": 0.20,
}


def _stats(model, texts, weights=None) -> dict[str, Any]:
    cleaned = [clean_description(t) for t in texts]
    w = np.ones(len(cleaned)) if weights is None else np.asarray(weights, dtype=float)
    present = (model.tfidf.transform(cleaned) > 0).astype(float)
    terms = np.asarray(present.multiply(w[:, None]).sum(axis=0)).ravel()
    vocab = model.tfidf.vocabulary_
    n_tokens, unseen = 0.0, Counter()
    for c, wi in zip(cleaned, w, strict=True):
        toks = unigram_tokens(c)
        n_tokens += wi * len(toks)
        for t in toks:
            if t not in vocab:
                unseen[t] += wi
    empty = present.getnnz(axis=1) == 0
    pred = model.pipeline.predict(list(texts)).astype(int)
    cats = np.bincount(pred, weights=w, minlength=len(model.categories))
    return {
        "terms": terms,
        "categories": cats,
        "oov_token_rate": float(sum(unseen.values()) / max(n_tokens, 1e-12)),
        "empty_row_rate": float((w * empty).sum() / max(w.sum(), 1e-12)),
        "unseen": unseen,
    }


def build_reference(model, texts, weights=None) -> dict[str, Any]:
    """Training baseline, stored in ``model.reference`` at train time."""
    s = _stats(model, list(texts), weights)
    return {
        "term_distribution": (s["terms"] / s["terms"].sum()).tolist(),
        "category_distribution": (s["categories"] / s["categories"].sum()).tolist(),
        "oov_token_rate": s["oov_token_rate"],
        "empty_row_rate": s["empty_row_rate"],
    }


def psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-4) -> float:
    e = np.clip(expected / expected.sum(), eps, None)
    a = np.clip(actual / max(actual.sum(), 1e-12), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(model, texts, thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    texts = list(texts)
    if not texts:
        raise ValueError("empty batch")
    ref = model.reference
    s = _stats(model, texts)
    ref_terms = np.asarray(ref["term_distribution"])
    term_js = (
        float(jensenshannon(ref_terms, s["terms"] / s["terms"].sum(), base=2))
        if s["terms"].sum() > 0
        else 1.0
    )
    ref_cat = np.asarray(ref["category_distribution"])
    batch_share = s["categories"] / s["categories"].sum()
    cat_psi = psi(ref_cat, s["categories"])
    shifts = sorted(
        (
            {
                "category": model.categories[i],
                "train_share": round(float(ref_cat[i]), 4),
                "batch_share": round(float(batch_share[i]), 4),
            }
            for i in range(len(ref_cat))
        ),
        key=lambda d: -abs(d["batch_share"] - d["train_share"]),
    )[:10]
    checks = {
        "oov_token_rate_increase": s["oov_token_rate"] - ref["oov_token_rate"]
        > thresholds["oov_token_rate_increase"],
        "empty_row_rate": s["empty_row_rate"] > thresholds["empty_row_rate"],
        "term_js_distance": term_js > thresholds["term_js_distance"],
        "category_psi": cat_psi > thresholds["category_psi"],
    }
    return {
        "n_rows": len(texts),
        "drift_detected": any(checks.values()),
        "failed_checks": [k for k, v in checks.items() if v],
        "vocabulary": {
            "oov_token_rate": round(s["oov_token_rate"], 4),
            "train_oov_token_rate": round(ref["oov_token_rate"], 4),
            "empty_row_rate": round(s["empty_row_rate"], 4),
            "term_js_distance": round(term_js, 4),
            "top_unseen_tokens": [
                {"token": t, "count": int(c)} for t, c in s["unseen"].most_common(15)
            ],
        },
        "categories": {
            "psi": round(cat_psi, 4),
            "js_distance": round(float(jensenshannon(ref_cat, batch_share, base=2)), 4),
            "largest_shifts": shifts,
        },
        "thresholds": thresholds,
    }
