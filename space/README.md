---
title: Transaction Classifier
emoji: 🧾
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
license: mit
short_description: Categorise retail transaction descriptions (TF-IDF + LogReg)
---

# Transaction classifier (Gradio Space)

Gradio front-end for [transaction-classifier-mlops](https://github.com/Nardy11/transaction-classifier-mlops).
It loads the same exported `model.json` as the GitHub Pages demo and runs the
TF-IDF -> SVD -> logistic regression maths with numpy. It does not load a pickle.

The categories are KMeans clusters (pseudo-labels), not hand-made classes. See
the main repository for the evaluation and its caveats.
