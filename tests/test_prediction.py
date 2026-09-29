"""Tests for prediction pipeline (requires a trained model)."""

from pathlib import Path

import pytest

from src.predict import predict_single

MODEL = Path(__file__).resolve().parent.parent / "models" / "phishing_email_model.pkl"

needs_model = pytest.mark.skipif(
    not MODEL.exists(), reason="trained model not found; run python src/train.py"
)


@needs_model
def test_model_loads_and_predicts():
    label, proba, feats, _ = predict_single("Hello", "Team meeting at 10am tomorrow.")
    assert label in ("PHISHING", "SAFE")
    assert proba is None or 0.0 <= proba <= 1.0
    assert "num_urls" in feats


@needs_model
def test_phishing_example_flagged():
    label, _, _, _ = predict_single(
        "Your account has been suspended",
        "Urgent: click http://192.168.1.1/verify immediately to verify "
        "your password or your account will be closed.",
    )
    assert label == "PHISHING"


def test_empty_input_raises():
    with pytest.raises((ValueError, FileNotFoundError)):
        predict_single("", "", MODEL)
