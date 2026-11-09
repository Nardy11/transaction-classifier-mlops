"""YAML configuration, loaded into plain dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    path: str | None = None  # local CSV; None -> download with kagglehub
    kaggle_dataset: str = "vipin20/transaction-data"
    kaggle_file: str = "transaction_data.csv"
    remove_outliers: bool = True
    max_rows: int | None = None


@dataclass
class FeaturesConfig:
    max_features: int = 5000
    ngram_max: int = 2
    svd_components: int = 150


@dataclass
class ClusteringConfig:
    n_clusters: int = 56
    n_init: int = 10
    name_terms: int = 3


@dataclass
class ClassifierConfig:
    C: float = 1.0
    max_iter: int = 2000
    class_weight: str | None = "balanced"


@dataclass
class EvalConfig:
    test_size: float = 0.2
    hand_labels: str | None = None


@dataclass
class OutputConfig:
    model_path: str = "models/model.joblib"
    reports_dir: str = "reports"
    mlflow_uri: str = "mlruns"
    experiment: str = "transaction-classifier"


@dataclass
class Config:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def flat(self) -> dict[str, Any]:
        """Flattened `section.key` dict, used as MLflow params."""
        out: dict[str, Any] = {}
        for key, value in asdict(self).items():
            if isinstance(value, dict):
                out.update({f"{key}.{k}": v for k, v in value.items()})
            else:
                out[key] = value
        return out


_SECTIONS = {
    "data": DataConfig,
    "features": FeaturesConfig,
    "clustering": ClusteringConfig,
    "classifier": ClassifierConfig,
    "eval": EvalConfig,
    "output": OutputConfig,
}


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    raw: dict[str, Any] = {}
    if path:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    kwargs: dict[str, Any] = {"seed": raw.get("seed", 42)}
    for name, cls in _SECTIONS.items():
        kwargs[name] = cls(**(raw.get(name) or {}))
    cfg = Config(**kwargs)
    for dotted, value in overrides.items():
        if value is None:
            continue
        section, key = dotted.split("__", 1)
        setattr(getattr(cfg, section), key, value)
    return cfg
