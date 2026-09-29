"""CLI prediction for the trained phishing email detector.

Running (from project root):
    python src/predict.py
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib

from src.preprocessing import combine_subject_body
from src.text_features import find_suspicious_keywords
from src.url_features import extract_url_features, extract_urls

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
    return payload, "unknown", {}


def predict_single(subject: str, body: str, model_path: Path = DEFAULT_MODEL):
    """Return (label, phishing_proba, url_features, keywords, model_name)."""
    subject = subject or ""
    body = body or ""
    if not subject.strip() and not body.strip():
        raise ValueError("Empty email: provide a subject and/or body.")
    pipeline, model_name, _ = load_model(Path(model_path))
    combined = combine_subject_body(subject, body)
    pred = pipeline.predict([combined])[0]
    phishing_proba = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba([combined])[0]
            classes = list(pipeline.classes_)
            phishing_proba = float(proba[classes.index(1)]) if 1 in classes else None
        except Exception:
            phishing_proba = None
    label = "PHISHING" if int(pred) == 1 else "SAFE"
    url_feats = extract_url_features(combined)
    keywords = find_suspicious_keywords(combined)
    return label, phishing_proba, url_feats, keywords, model_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Phishing email detector (CLI).")
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--body", type=str, default=None)
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    args = parser.parse_args()

    print("--------------------------------")
    print("PHISHING EMAIL DETECTOR")
    print("--------------------------------")
    try:
        if args.subject is not None or args.body is not None:
            subject = args.subject or ""
            body = args.body or ""
        else:
            subject = input("Enter email subject: ")
            body = input("Enter email body: ")
        label, phishing_proba, url_feats, keywords, model_name = predict_single(
            subject, body, Path(args.model)
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return

    combined = combine_subject_body(subject, body)
    urls = extract_urls(combined)
    n_https = sum(1 for u in urls if u.lower().startswith("https://"))

    print(f"\nPrediction: {label}")
    if phishing_proba is not None:
        print(f"\nPhishing Probability: {phishing_proba * 100:.2f}%")
        print("(Model confidence only — not a guarantee the email is/ isn't malicious.)")
    print(f"\nDetected URLs: {int(url_feats['num_urls'])}")
    print(f"Suspicious Keywords: {len(keywords)}"
          + (f" ({', '.join(keywords)})" if keywords else ""))
    print(f"HTTPS URLs: {n_https}")
    print(f"Model: {model_name}")
    print(
        f"URL statistics: length={int(url_feats['max_url_length'])}, "
        f"unique={int(url_feats['num_unique_urls'])}, "
        f"ip_urls={int(url_feats['num_ip_urls'])}, "
        f"dots={int(url_feats['num_dots'])}, "
        f"hyphens={int(url_feats['num_hyphens'])}, "
        f"query_params={int(url_feats['num_query_params'])}, "
        f"suspicious_pattern={int(url_feats['has_suspicious_pattern'])}, "
        f"shortener={int(url_feats['has_url_shortener'])}"
    )


if __name__ == "__main__":
    main()
