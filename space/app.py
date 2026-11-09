"""Hugging Face Space: Gradio front-end for the transaction classifier.

Uses the same exported model.json as the GitHub Pages demo (numpy only, no
pickle). Put model.json next to this file, or it is fetched from GitHub Pages.
"""

from __future__ import annotations

import base64
import json
import re
import urllib.request
from pathlib import Path

import gradio as gr
import numpy as np

MODEL_URL = "https://nardy11.github.io/transaction-classifier-mlops/model.json"
_here = Path(__file__).parent / "model.json"
_raw = (
    json.loads(_here.read_text(encoding="utf-8"))
    if _here.exists()
    else json.load(urllib.request.urlopen(MODEL_URL, timeout=30))
)

VOCAB = {t: i for i, t in enumerate(_raw["vocabulary"])}
IDF = np.asarray(_raw["idf"])
STOP = set(_raw["stop_words"])
NGRAM = _raw["ngram_max"]
CATS = _raw["categories"]
B = np.asarray(_raw["intercept"])
W = (
    np.frombuffer(base64.b64decode(_raw["weights_int16_b64"]), dtype="<i2")
    .reshape(_raw["weights_shape"])
    .astype(np.float64)
    * _raw["weights_scale"]
)


def clean(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9 ]+", " ", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()


def classify(text: str) -> dict[str, float]:
    tokens = [t for t in re.findall(r"\b\w\w+\b", clean(text)) if t not in STOP]
    x = np.zeros(len(IDF))
    for n in range(1, NGRAM + 1):
        for i in range(len(tokens) - n + 1):
            j = VOCAB.get(" ".join(tokens[i : i + n]))
            if j is not None:
                x[j] += 1
    x *= IDF
    norm = np.linalg.norm(x)
    if norm:
        x /= norm
    z = W @ x + B
    p = np.exp(z - z.max())
    p /= p.sum()
    top = np.argsort(p)[::-1][:3]
    return {CATS[i]: float(p[i]) for i in top}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(label="Transaction description", value="jumbo bag red retrospot"),
    outputs=gr.Label(num_top_classes=3, label="Predicted category"),
    title="Transaction classifier",
    description=(
        "TF-IDF + SVD + logistic regression, trained on KMeans pseudo-labels from a UK "
        "gift retailer's transactions. Code: github.com/Nardy11/transaction-classifier-mlops"
    ),
    examples=[["regency cakestand 3 tier"], ["hot water bottle tea and sympathy"]],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
