"""Streamlit UI: Phishing Email Detection System.

Run with:
    streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import DEFAULT_MODEL, load_model
from src.preprocessing import combine_subject_body
from src.text_features import find_suspicious_keywords
from src.url_features import FEATURE_NAMES, extract_url_features, extract_urls

PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_CM = PROJECT_ROOT / "reports" / "confusion_matrix.png"
REPORT_CSV = PROJECT_ROOT / "reports" / "model_comparison.csv"

st.set_page_config(page_title="Phishing Email Detection System", page_icon="🛡️")

st.title("Phishing Email Detection System")
st.write(
    "Static ML analysis of email subject + body (text + URL features). "
    "Links are never opened, requested, or downloaded."
)


@st.cache_resource
def get_model():
    return load_model(Path(DEFAULT_MODEL))


try:
    pipeline, model_name, payload = get_model()
    st.caption(f"Loaded model: {model_name}")
except (FileNotFoundError, ValueError) as exc:
    st.error(
        f"Model not found or corrupted: {exc}. "
        "Train it first with `python src/train.py`."
    )
    st.stop()

metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}

# ---------- 1. Email Input ----------
st.header("1. Email Input")
subject = st.text_input("Email Subject:")
body = st.text_area("Email Body:", height=200)
analyze = st.button("ANALYZE EMAIL")

# ---------- 2 & 3. Feature Analysis + Prediction ----------
st.header("2. Feature Analysis")
st.header("3. Prediction")

if analyze:
    if not subject.strip() and not body.strip():
        st.warning("Please enter a subject and/or email body.")
    else:
        combined = combine_subject_body(subject, body)
        urls = extract_urls(combined)
        feats = extract_url_features(combined)
        keywords = find_suspicious_keywords(combined)
        n_https = sum(1 for u in urls if u.lower().startswith("https://"))
        n_ip = int(feats["num_ip_urls"])

        pred = pipeline.predict([combined])[0]
        phishing_proba = None
        if hasattr(pipeline, "predict_proba"):
            try:
                proba = pipeline.predict_proba([combined])[0]
                classes = list(pipeline.classes_)
                phishing_proba = (
                    float(proba[classes.index(1)]) if 1 in classes else None
                )
            except Exception:
                phishing_proba = None

        is_phishing = int(pred) == 1
        if is_phishing:
            st.error("Prediction: PHISHING")
        else:
            st.success("Prediction: SAFE")

        if phishing_proba is not None:
            st.metric("Probability", f"{phishing_proba * 100:.2f}%")
            st.caption(
                "Model confidence only — not a guarantee the email "
                "is or isn't malicious. Verify independently."
            )

        st.subheader("Extracted static features")
        st.write(f"URLs detected: {int(feats['num_urls'])}")
        st.write(
            f"Suspicious keywords: {len(keywords)}"
            + (f" ({', '.join(keywords)})" if keywords else "")
        )
        st.write(f"URL length: {int(feats['max_url_length'])}")
        st.write(f"HTTPS URLs: {n_https}")
        st.write(f"IP-based URLs: {n_ip}")
        with st.expander("All URL features"):
            st.table({name: [f"{feats[name]:.4f}"] for name in FEATURE_NAMES})
        if urls:
            with st.expander("Detected URL strings (do not open)"):
                for u in urls:
                    st.code(u)
else:
    st.info("Enter an email above and press ANALYZE EMAIL.")

# ---------- 4. Model Performance ----------
st.header("4. Model Performance")
if metrics:
    st.write(f"Model Accuracy: {metrics.get('accuracy', 0) * 100:.2f}%")
    st.write(f"Precision: {metrics.get('precision', 0):.4f}")
    st.write(f"Recall: {metrics.get('recall', 0):.4f}")
    st.write(f"F1 Score: {metrics.get('f1', 0):.4f}")
    st.caption(
        "Computed on the held-out test set (never hardcoded). "
        "Accuracy alone is insufficient: a model can be accurate yet miss "
        "rare phishing attacks, so precision (false alarms) and recall "
        "(missed attacks) are reported alongside."
    )
    if REPORT_CSV.exists():
        st.subheader("Model comparison (actual training results)")
        st.dataframe(pd.read_csv(REPORT_CSV))
else:
    st.warning("No metrics found in the saved model. Retrain with `python src/train.py`.")

# ---------- 5. Confusion Matrix ----------
st.header("5. Confusion Matrix")
if REPORT_CM.exists():
    st.image(str(REPORT_CM), caption="Confusion matrix (test set: Safe vs Phishing)")
    st.caption(
        "Rows = actual, columns = predicted. Diagonal = correct "
        "(True Negative, True Positive); off-diagonal = errors "
        "(False Positive, False Negative)."
    )
else:
    st.warning(
        "reports/confusion_matrix.png not found. Run `python src/train.py` to generate it."
    )

st.divider()
st.warning(
    "⚠️ Disclaimer: educational defensive-security demo. Predictions can produce "
    "false positives and false negatives. Never click suspicious links; verify "
    "sensitive requests through official channels."
)
