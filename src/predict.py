"""CLI prediction for the trained phishing email detector.

Usage (from project root):
    python src/predict.py
    python src/predict.py --subject "..." --body "..."
    python src/predict.py --model models/phishing_email_model.pkl
"""

import argparse
from pathlib import Path

import joblib

from .preprocessing import combine_subject_body
from .url_features import extract_url_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "phishing_email_model.pkl"


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train it first with: python src/train.py"
        )
    try:
        payload = joblib.load(model_path)
    except Exception as exc:
        raise ValueError(f"Corrupted model file at {model_path}: {exc}") from exc
    if isinstance(payload, dict) and "pipeline" in payload:
        return payload["pipeline"], payload.get("model_name", "unknown"), payload
    # Backwards compatibility: raw pipeline was saved directly.
    return payload, "unknown", {}


def predict_single(subject: str, body: str, model_path: Path = DEFAULT_MODEL):
    """Return (label_str, probability, url_features_dict)."""
    subject = subject or ""
    body = body or ""
    if not subject.strip() and not body.strip():
        raise ValueError("Empty email: provide a subject and/or body.")
    pipeline, model_name, _ = load_model(Path(model_path))
    combined = combine_subject_body(subject, body)
    pred = pipeline.predict([combined])[0]
    proba = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = float(max(pipeline.predict_proba([combined])[0]))
        except Exception:
            proba = None
    label = "PHISHING" if int(pred) == 1 else "SAFE"
    return label, proba, extract_url_features(combined), model_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Phishing email detector (CLI).")
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--body", type=str, default=None)
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    args = parser.parse_args()

    print("================================")
    print("PHISHING EMAIL DETECTOR")
    print("================================")
    try:
        if args.subject is not None or args.body is not None:
            subject = args.subject or ""
            body = args.body or ""
        else:
            subject = input("Enter email subject: ")
            body = input("Enter email body: ")
        label, proba, url_feats, model_name = predict_single(
            subject, body, Path(args.model)
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return

    print(f"\nPrediction: {label}")
    if proba is not None:
        print(f"Probability: {proba * 100:.2f}%")
        print("(Model confidence, not a guarantee the email is/ isn't malicious.)")
    print(f"Model: {model_name}")
    print(
        f"URL signals: num_urls={int(url_feats['num_urls'])}, "
        f"ip_urls={int(url_feats['num_ip_urls'])}, "
        f"suspicious_tld={int(url_feats['has_suspicious_tld'])}, "
        f"https={int(url_feats['has_https'])}"
    )


if __name__ == "__main__":
    main()
