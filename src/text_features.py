"""Text features: TF-IDF + suspicious-keyword analysis.

A single keyword NEVER decides the label on its own. Keyword counts
are explanatory signals shown next to the ML prediction; the actual
classification comes from the trained TF-IDF + URL-feature model.
"""

import re
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocessing import clean_text

TFIDF_MAX_FEATURES = 5000

SUSPICIOUS_KEYWORDS: List[str] = [
    "verify",
    "account",
    "password",
    "login",
    "suspended",
    "urgent",
    "immediately",
    "click",
    "confirm",
    "security",
    "payment",
    "invoice",
    "bank",
    "winner",
    "congratulations",
    "update",
    "credential",
]

_WORD_RE = re.compile(r"[a-z0-9]+")


def build_tfidf_vectorizer(
    max_features: int = TFIDF_MAX_FEATURES,
) -> TfidfVectorizer:
    """TF-IDF over cleaned text (unigrams + bigrams, English stop words)."""
    return TfidfVectorizer(
        preprocessor=clean_text,
        lowercase=False,  # clean_text already lowercases
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
    )


def find_suspicious_keywords(text: str) -> List[str]:
    """Return sorted list of suspicious keywords present in text (case-insensitive)."""
    if text is None:
        return []
    words = set(_WORD_RE.findall(str(text).lower()))
    return sorted(kw for kw in SUSPICIOUS_KEYWORDS if kw in words)


def count_suspicious_keywords(text: str) -> int:
    """Count distinct suspicious keywords present in text."""
    return len(find_suspicious_keywords(text))


def keyword_features(text: str) -> Dict[str, int]:
    """Explanatory keyword stats for display (not a classifier)."""
    matched = find_suspicious_keywords(text)
    return {"num_suspicious_keywords": len(matched), "matched_keywords": matched}
