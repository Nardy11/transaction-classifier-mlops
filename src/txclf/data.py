"""Loading and cleaning, ported from the notebook's cleaning cells."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from txclf.config import DataConfig

NUMERIC_COLS = ["TotalCost", "NumberOfItemsPurchased", "CostPerItem"]
_NON_ALNUM = re.compile(r"[^a-zA-Z0-9 ]+")
_SPACES = re.compile(r"\s+")


def clean_description(text: object) -> str:
    """Lowercase, replace symbols with spaces, collapse whitespace.

    Same rule as the notebook's ``clear_Description``. The browser demo
    (docs/txclf.js) implements the identical rule.
    """
    if text is None or (isinstance(text, float) and text != text):
        return ""
    text = _NON_ALNUM.sub(" ", str(text).lower())
    return _SPACES.sub(" ", text).strip()


def clean_many(texts) -> list[str]:
    """Vectorised form, used as the first step of the sklearn pipeline."""
    return [clean_description(t) for t in texts]


def resolve_path(cfg: DataConfig) -> Path:
    if cfg.path:
        return Path(cfg.path)
    import kagglehub  # optional: only needed for the full dataset

    folder = kagglehub.dataset_download(cfg.kaggle_dataset)
    return Path(folder) / cfg.kaggle_file


def load_raw(cfg: DataConfig) -> pd.DataFrame:
    df = pd.read_csv(resolve_path(cfg), nrows=cfg.max_rows)
    missing = {"ItemDescription", "NumberOfItemsPurchased", "CostPerItem"} - set(df.columns)
    if missing:
        raise ValueError(f"input is missing columns: {sorted(missing)}")
    return df


def remove_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    q1, q3 = df[column].quantile(0.25), df[column].quantile(0.75)
    iqr = q3 - q1
    return df[(df[column] >= q1 - 1.5 * iqr) & (df[column] <= q3 + 1.5 * iqr)]


def clean(df: pd.DataFrame, remove_outliers: bool = True) -> pd.DataFrame:
    """Notebook cleaning: TotalCost, text cleaning, de-dup, nulls, IQR outliers.

    Rows whose description is empty after cleaning are dropped (the notebook kept
    them as a "nan" cluster, which is noise rather than a category).
    """
    df = df.copy()
    if "UserId" in df.columns:
        df["UserId"] = df["UserId"].replace(-1, 1)
    df["TotalCost"] = df["NumberOfItemsPurchased"] * df["CostPerItem"]
    df["Description"] = df["ItemDescription"].map(clean_description)
    df = df.drop(columns=[c for c in ("ItemDescription", "TransactionId") if c in df.columns])
    df = df.drop_duplicates()
    df = df.dropna(subset=["Description", *NUMERIC_COLS])
    df = df[~df["Description"].isin(["", "nan", "none"])]
    if remove_outliers:
        for col in NUMERIC_COLS:
            df = remove_outliers_iqr(df, col)
    return df.reset_index(drop=True)


def unique_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct description with its row count.

    ~540K rows collapse to a few thousand product descriptions, so the model is
    fitted on these (weighted by count) instead of on repeated rows.
    """
    counts = df["Description"].value_counts()
    return pd.DataFrame({"Description": counts.index, "count": counts.values})
