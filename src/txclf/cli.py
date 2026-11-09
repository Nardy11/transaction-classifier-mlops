"""python -m txclf train | evaluate | predict | export-web | drift"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd


def _texts_from_csv(path: str, column: str) -> list[str]:
    df = pd.read_csv(path)
    if column not in df.columns:
        sys.exit(f"column {column!r} not in {path}")
    return df[column].astype(str).tolist()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="txclf", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train, evaluate, save and log to MLflow")
    t.add_argument("--config", default="configs/default.yaml")
    t.add_argument("--data", help="local CSV (default: config / kagglehub download)")
    t.add_argument("--model-path")
    t.add_argument("--no-mlflow", action="store_true")
    t.add_argument("--no-figures", action="store_true")

    e = sub.add_parser("evaluate", help="pseudo-label consistency + hand-label check")
    e.add_argument("--model", default="models/model.joblib")
    e.add_argument("--data", help="CSV with ItemDescription to score against KMeans labels")
    e.add_argument("--hand-labels", default="data/hand_labels.csv")

    pr = sub.add_parser("predict", help="predict categories")
    pr.add_argument("text", nargs="*")
    pr.add_argument("--model", default="models/model.joblib")
    pr.add_argument("--input", help="CSV file instead of positional text")
    pr.add_argument("--column", default="ItemDescription")
    pr.add_argument("--top-k", type=int, default=3)

    w = sub.add_parser("export-web", help="export compact JSON for the browser demo")
    w.add_argument("--model", default="models/model.joblib")
    w.add_argument("--out", default="docs/model.json")

    d = sub.add_parser("drift", help="drift report for a new batch (JSON)")
    d.add_argument("batch", help="CSV with a description column")
    d.add_argument("--model", default="models/model.joblib")
    d.add_argument("--column", default="ItemDescription")
    d.add_argument("--out", help="write the JSON report here as well")
    d.add_argument("--fail-on-drift", action="store_true", help="exit 1 when drift is found")

    a = p.parse_args(argv)

    if a.cmd == "train":
        from txclf.config import load_config
        from txclf.train import train

        cfg = load_config(a.config, data__path=a.data, output__model_path=a.model_path)
        _, metrics = train(cfg, log_mlflow=not a.no_mlflow, figures=not a.no_figures)
        print(json.dumps(metrics, indent=2))
        return 0

    from txclf.model import load_model

    model = load_model(a.model)

    if a.cmd == "evaluate":
        import os

        from txclf.data import clean
        from txclf.evaluate import consistency_metrics, hand_label_metrics

        out: dict = {}
        if a.data:
            df = clean(pd.read_csv(a.data), remove_outliers=False)
            texts = df["Description"].tolist()
            out.update(
                consistency_metrics(model.pseudo_labels(texts), model.pipeline.predict(texts))
            )
        if a.hand_labels and os.path.exists(a.hand_labels):
            out.update(hand_label_metrics(model, a.hand_labels))
        print(json.dumps(out, indent=2))
        return 0

    if a.cmd == "predict":
        texts = _texts_from_csv(a.input, a.column) if a.input else a.text
        if not texts:
            sys.exit("give a description or --input file.csv")
        print(json.dumps(model.predict(texts, top_k=a.top_k), indent=2))
        return 0

    if a.cmd == "export-web":
        from txclf.export import export_web

        print(json.dumps(export_web(model, a.out)))
        return 0

    if a.cmd == "drift":
        from txclf.drift import drift_report

        report = drift_report(model, _texts_from_csv(a.batch, a.column))
        text = json.dumps(report, indent=2)
        print(text)
        if a.out:
            with open(a.out, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
        return 1 if (a.fail_on_drift and report["drift_detected"]) else 0
    return 2
