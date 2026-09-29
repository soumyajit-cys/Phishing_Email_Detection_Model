"""Reusable text preprocessing for phishing email detection."""

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
    """Clean a single email text string.

    Steps: unescape HTML entities, strip HTML tags, lowercase,
    replace URLs/emails with placeholder tokens, collapse whitespace.

    Punctuation is left for the TF-IDF tokenizer to handle, except
    that excessive non-alphanumeric noise is normalised to spaces
    so placeholder tokens remain meaningful.
    """
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
    # Handle NaN floats coming from pandas.
    if subject.lower() == "nan":
        subject = ""
    if body.lower() == "nan":
        body = ""
    combined = f"{subject.strip()} {body.strip()}".strip()
    combined = _WHITESPACE_RE.sub(" ", combined)
    return combined


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean a raw email dataframe.

    Expected columns: subject (optional), email_text/body, label.
    Accepts 'email_text', 'body', or 'email_body' as the body column.
    Returns a dataframe with columns: subject, email_text, label, combined_text.
    """
    if df is None or df.empty:
        raise ValueError("Dataset is empty.")

    body_col = None
    for candidate in ("email_text", "body", "email_body", "text", "content"):
        if candidate in df.columns:
            body_col = candidate
            break
    if body_col is None:
        raise ValueError(
            f"Missing email body column. Found columns: {list(df.columns)}. "
            "Expected one of: email_text, body, email_body."
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
    cleaned = cleaned.rename(columns={body_col: "email_text"})

    # Drop rows where both subject and body are empty.
    mask_empty = (
        cleaned["subject"].str.strip().eq("")
        & cleaned["email_text"].str.strip().eq("")
    )
    cleaned = cleaned.loc[~mask_empty].copy()
    if cleaned.empty:
        raise ValueError("Dataset has no usable rows after removing empty emails.")

    # Validate labels are 0/1.
    try:
        cleaned["label"] = cleaned["label"].astype(int)
    except (ValueError, TypeError) as exc:
        raise ValueError("Column 'label' must contain only 0 and 1.") from exc
    invalid = set(cleaned["label"].unique()) - {0, 1}
    if invalid:
        raise ValueError(f"Invalid labels found: {invalid}. Expected only 0 and 1.")

    cleaned["combined_text"] = [
        combine_subject_body(s, b)
        for s, b in zip(cleaned["subject"], cleaned["email_text"])
    ]
    return cleaned[["subject", "email_text", "label", "combined_text"]]
