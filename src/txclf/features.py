"""Text features: clean -> TF-IDF (1-2 grams) -> TruncatedSVD."""

from __future__ import annotations

import re

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from txclf.config import FeaturesConfig
from txclf.data import clean_many

_TOKEN = re.compile(r"(?u)\b\w\w+\b")  # sklearn's default token_pattern


def build_text_pipeline(cfg: FeaturesConfig, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("clean", FunctionTransformer(clean_many)),
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=cfg.max_features,
                    ngram_range=(1, cfg.ngram_max),
                ),
            ),
            ("svd", TruncatedSVD(n_components=cfg.svd_components, random_state=seed)),
        ]
    )


def unigram_tokens(cleaned: str) -> list[str]:
    """Unigrams exactly as TfidfVectorizer sees them (stop words removed)."""
    return [t for t in _TOKEN.findall(cleaned) if t not in ENGLISH_STOP_WORDS]
