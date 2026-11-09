"""The browser model (docs/txclf.js) must agree with the Python model."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from txclf.export import export_web
from txclf.model import load_model

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")

EXTRA = ["", "qqq unknown words", "Set/3 RED Retrospot cake-tins!!", "the and of"]


def _compare(model, model_json: Path, texts: list[str], tmp_path: Path) -> None:
    tpath = tmp_path / "texts.json"
    tpath.write_text(json.dumps(texts), encoding="utf-8")
    out = subprocess.run(
        [NODE, str(ROOT / "scripts" / "web_predict.mjs"), str(model_json), str(tpath)],
        check=True,
        capture_output=True,
        text=True,
    )
    js = json.loads(out.stdout)
    py = model.predict_proba(texts)
    order = [model.categories[int(c)] for c in model.clf.classes_]
    for i, text in enumerate(texts):
        assert np.abs(np.asarray(js[i]["proba"]) - py[i]).max() < 2e-3, text
        top2 = np.sort(py[i])[-2:]
        if top2[1] - top2[0] > 2e-3:  # no near-tie: the winner must match exactly
            assert js[i]["category"] == order[int(np.argmax(py[i]))], text


def test_js_matches_python_on_ci_model(trained, sample_df, tmp_path):
    model, _, _ = trained
    path = tmp_path / "model.json"
    export_web(model, path)
    texts = sample_df["ItemDescription"].astype(str).sample(20, random_state=1).tolist()
    _compare(model, path, texts + EXTRA, tmp_path)


@pytest.mark.skipif(
    not (ROOT / "models" / "model.joblib").exists(),
    reason="full model not present (download model.joblib from the v1.0.0 release)",
)
def test_js_matches_python_on_published_model(sample_df, tmp_path):
    model = load_model(ROOT / "models" / "model.joblib")
    texts = sample_df["ItemDescription"].astype(str).sample(20, random_state=2).tolist()
    _compare(model, ROOT / "docs" / "model.json", texts + EXTRA, tmp_path)
