"""Streamlit UI for the phishing email detector.

Run with:
    streamlit run app.py
"""

from pathlib import Path

import streamlit as st

from src.predict import DEFAULT_MODEL, load_model
from src.preprocessing import combine_subject_body
from src.url_features import FEATURE_NAMES, extract_url_features

st.set_page_config(page_title="Phishing Email Detector", page_icon="🛡️")

st.title("🛡️ Phishing Email Detection Model")
st.write(
    "Static ML analysis of email subject + body. "
    "The tool never opens links, downloads content, or sends email."
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

subject = st.text_input("Email subject")
body = st.text_area("Email body", height=200)

if st.button("Analyze Email"):
    if not subject.strip() and not body.strip():
        st.warning("Please enter a subject and/or email body.")
    else:
        combined = combine_subject_body(subject, body)
        pred = pipeline.predict([combined])[0]
        proba = None
        if hasattr(pipeline, "predict_proba"):
            try:
                proba = float(max(pipeline.predict_proba([combined])[0]))
            except Exception:
                proba = None

        is_phishing = int(pred) == 1
        if is_phishing:
            st.error("🔴 PHISHING")
        else:
            st.success("🟢 SAFE")

        if proba is not None:
            st.metric("Confidence", f"{proba * 100:.2f}%")
            st.caption(
                "Model confidence only — not a guarantee the email "
                "is or isn't malicious. Verify independently."
            )

        feats = extract_url_features(combined)
        st.subheader("Extracted URL signals (static analysis)")
        st.table(
            {name: [f"{feats[name]:.0f}"] for name in FEATURE_NAMES}
        )

st.divider()
st.warning(
    "⚠️ Disclaimer: this is an educational defensive-security demo. "
    "ML predictions can produce false positives and false negatives. "
    "Never click suspicious links. Verify sensitive requests through "
    "official channels."
)
