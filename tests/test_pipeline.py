import numpy as np

from txclf.drift import drift_report
from txclf.model import load_model


def test_metrics_are_reported(trained):
    _, metrics, _ = trained
    for key in (
        "unique_descriptions",
        "silhouette",
        "heldout_accuracy",
        "heldout_macro_f1",
        "seen_accuracy",
        "heldout_shuffled_label_accuracy",
    ):
        assert key in metrics
    # the classifier must beat a shuffled-label baseline by a wide margin
    assert metrics["heldout_accuracy"] > metrics["heldout_shuffled_label_accuracy"] + 0.3


def test_predict_shape_and_probabilities(trained):
    model, _, _ = trained
    preds = model.predict(["white hanging heart t-light holder", "jumbo bag red retrospot"])
    assert len(preds) == 2
    for p in preds:
        assert p["category"] in model.categories
        probs = [t["probability"] for t in p["top_k"]]
        assert len(probs) == 3 and probs == sorted(probs, reverse=True)
        assert 0 < sum(probs) <= 1 + 1e-9
    assert np.allclose(model.predict_proba(["x y z"]).sum(), 1.0)


def test_unknown_text_has_no_known_terms(trained):
    model, _, _ = trained
    assert model.predict(["qqqzzz"])[0]["known_terms"] == 0


def test_save_and_load_roundtrip(trained):
    model, _, path = trained
    loaded = load_model(path)
    texts = ["regency cakestand 3 tier", "paper chain kit"]
    assert np.allclose(loaded.predict_proba(texts), model.predict_proba(texts))
    assert loaded.categories == model.categories


def test_drift_quiet_on_training_like_batch(trained, sample_df):
    model, _, _ = trained
    report = drift_report(model, sample_df["ItemDescription"].astype(str).head(800))
    assert not report["drift_detected"], report["failed_checks"]
    assert report["categories"]["psi"] < 0.2


def test_drift_flags_foreign_batch(trained):
    model, _, _ = trained
    batch = ["uber trip airport", "netflix monthly subscription", "rent payment march"] * 30
    report = drift_report(model, batch)
    assert report["drift_detected"]
    assert "oov_token_rate_increase" in report["failed_checks"]
    assert report["vocabulary"]["top_unseen_tokens"]
