"""Tests for prediction pipeline (requires a trained model)."""

from pathlib import Path

import pytest

from src.predict import load_model, predict_single

MODEL = Path(__file__).resolve().parent.parent / "models" / "phishing_email_model.pkl"

needs_model = pytest.mark.skipif(
    not MODEL.exists(), reason="trained model not found; run python src/train.py"
)


@needs_model
def test_model_loading():
    pipeline, name, payload = load_model(MODEL)
    assert hasattr(pipeline, "predict")
    assert isinstance(name, str)


@needs_model
def test_model_loads_and_predicts_normal_email():
    label, proba, feats, keywords, _ = predict_single(
        "Meeting Scheduled for Tomorrow",
        "Hi team, the meeting is scheduled for tomorrow at 10 AM.",
    )
    assert label in ("PHISHING", "SAFE")
    assert proba is None or 0.0 <= proba <= 1.0
    assert "num_urls" in feats
    assert isinstance(keywords, list)


@needs_model
def test_phishing_example_flagged():
    label, _, _, _, _ = predict_single(
        "Your account has been suspended",
        "Urgent: click http://192.168.1.1/verify immediately to verify "
        "your password or your account will be closed.",
    )
    assert label == "PHISHING"


@needs_model
def test_prediction_output_shape():
    out = predict_single("Hi", "Team lunch tomorrow?")
    assert len(out) == 5
    label, proba, feats, keywords, model_name = out
    assert label in ("PHISHING", "SAFE")


def test_empty_input_raises():
    with pytest.raises((ValueError, FileNotFoundError)):
        predict_single("", "", MODEL)
