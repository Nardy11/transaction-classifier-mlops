from pathlib import Path

import pandas as pd
import pytest

from txclf.config import load_config
from txclf.train import train

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample" / "transactions_sample.csv"


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    return pd.read_csv(SAMPLE)


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("model")
    cfg = load_config(
        ROOT / "configs" / "ci.yaml",
        data__path=str(SAMPLE),
        output__model_path=str(out / "model.joblib"),
        output__reports_dir=str(out / "reports"),
    )
    model, metrics = train(cfg, log_mlflow=False, figures=False)
    return model, metrics, out / "model.joblib"
