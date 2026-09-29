"""Reusable text preprocessing for phishing email detection.

Design note: phishing indicators (URLs, domains, numbers, suspicious
words, special URL characters) are PRESERVED. clean_text only
lowercases, unescapes/strips HTML, replaces raw URLs/emails with
placeholder tokens (so the model still sees a URL signal without
memorising exact domains), and collapses whitespace. Punctuation and
numbers are left for the TF-IDF tokenizer.
"""

import re
from html import unescape

import pandas as pd

URL_PLACEHOLDER = "urltoken"
EMAIL_PLACEHOLDER = "emailtoken"

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Clean a single email string while preserving phishing signals."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = text.lower()
    text = _URL_RE.sub(f" {URL_PLACEHOLDER} ", text)
    text = _EMAIL_RE.sub(f" {EMAIL_PLACEHOLDER} ", text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def combine_subject_body(subject: str | None, body: str | None) -> str:
    """Combine subject and body into a single raw string."""
    subject = "" if subject is None else str(subject)
    body = "" if body is None else str(body)
    if subject.lower() == "nan":
        subject = ""
    if body.lower() == "nan":
        body = ""
    combined = f"{subject.strip()} {body.strip()}".strip()
    combined = _WHITESPACE_RE.sub(" ", combined)
    return combined


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a canonical dataframe with subject/body/label columns.

    Prefer src.data_loader.normalise_columns for loading raw CSVs with
    arbitrary column names; this function handles the final cleaning step
    and also accepts the legacy 'email_text' body column.
    """
    if df is None or df.empty:
        raise ValueError("Dataset is empty.")

    body_col = None
    for candidate in ("body", "email_text", "email_body", "text", "content"):
        if candidate in df.columns:
            body_col = candidate
            break
    if body_col is None:
        raise ValueError(
            f"Missing email body column. Found columns: {list(df.columns)}. "
            "Expected columns: subject, body, label."
        )
    if "label" not in df.columns:
        raise ValueError(
            f"Missing 'label' column. Found columns: {list(df.columns)}."
        )

    cleaned = df.copy()
    if "subject" not in cleaned.columns:
        cleaned["subject"] = ""

    cleaned["subject"] = cleaned["subject"].fillna("").astype(str)
    cleaned[body_col] = cleaned[body_col].fillna("").astype(str)
    if body_col != "body":
        cleaned = cleaned.rename(columns={body_col: "body"})

    mask_empty = (
        cleaned["subject"].str.strip().eq("") & cleaned["body"].str.strip().eq("")
    )
    cleaned = cleaned.loc[~mask_empty].copy()
    if cleaned.empty:
        raise ValueError("Dataset has no usable rows after removing empty emails.")

    try:
        cleaned["label"] = cleaned["label"].astype(int)
    except (ValueError, TypeError) as exc:
        raise ValueError("Column 'label' must contain only 0 and 1.") from exc
    invalid = set(cleaned["label"].unique()) - {0, 1}
    if invalid:
        raise ValueError(f"Invalid labels found: {invalid}. Expected only 0 and 1.")

    cleaned["combined_text"] = [
        combine_subject_body(s, b)
        for s, b in zip(cleaned["subject"], cleaned["body"])
    ]
    return cleaned[["subject", "body", "label", "combined_text"]]
