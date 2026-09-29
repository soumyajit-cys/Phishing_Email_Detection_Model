"""Static URL feature extraction (no network requests).

Only static string/regex analysis is performed. URLs are never
visited, requested, crawled, downloaded, or opened.
"""

import re
from typing import Dict, List
from urllib.parse import urlparse

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

URL_REGEX = re.compile(r"https?://[^\s<>'\"]+|www\.[^\s<>'\"]+", re.IGNORECASE)
IP_URL_RE = re.compile(r"https?://\d{1,3}(?:\.\d{1,3}){3}")
SUSPICIOUS_CHARS = set("@-_%&=~?#$!+")
SUSPICIOUS_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".click",
    ".link", ".work", ".country", ".stream", ".download", ".loan",
    ".win", ".bid", ".party", ".review",
}
URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "bit.do", "cutt.ly", "tiny.cc", "shorturl.at",
}

FEATURE_NAMES: List[str] = [
    "num_urls",
    "max_url_length",
    "num_unique_urls",
    "num_ip_urls",
    "max_num_subdomains",
    "num_dots",
    "num_hyphens",
    "num_suspicious_chars",
    "num_query_params",
    "has_https",
    "has_suspicious_pattern",
    "has_url_shortener",
    "url_to_text_ratio",
]


def extract_urls(text: str) -> List[str]:
    """Extract raw URL strings from text using regex only."""
    if not text or not isinstance(text, str):
        return []
    return URL_REGEX.findall(text)


def _domain_of(url: str) -> str:
    candidate = url if "://" in url else f"http://{url}"
    try:
        return urlparse(candidate).netloc.lower()
    except Exception:
        return ""


def _has_suspicious_pattern(url: str, domain: str) -> bool:
    lower = url.lower()
    if any(domain.endswith(tld) for tld in SUSPICIOUS_TLDS):
        return True
    if "@" in url:
        return True
    after_protocol = lower.split("://", 1)[-1] if "://" in lower else lower
    if "//" in after_protocol:
        return True
    if any(k in lower for k in ("verify", "login", "secure", "account", "redirect")) and (
        "%" in lower or "redirect" in lower
    ):
        return True
    return False


def extract_url_features(text: str) -> Dict[str, float]:
    """Extract static numeric URL features from a single email string."""
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)

    urls = extract_urls(text)
    num_urls = len(urls)
    unique_urls = set(u.lower() for u in urls)
    num_ip_urls = sum(1 for u in urls if IP_URL_RE.search(u))
    max_url_length = max((len(u) for u in urls), default=0)

    max_subdomains = 0
    num_dots = 0
    num_hyphens = 0
    num_suspicious_chars = 0
    num_query_params = 0
    has_https = 0
    has_suspicious_pattern = 0
    has_url_shortener = 0

    for url in urls:
        lower = url.lower()
        if lower.startswith("https://"):
            has_https = 1
        num_suspicious_chars += sum(url.count(c) for c in SUSPICIOUS_CHARS)
        num_dots += url.count(".")
        num_hyphens += url.count("-")
        # Query parameters: count '&' plus one if '?' present.
        if "?" in url:
            num_query_params += url.count("&") + 1

        domain = _domain_of(url).split(":")[0].split("@")[-1]
        bare = domain[4:] if domain.startswith("www.") else domain
        parts = [p for p in bare.split(".") if p]
        if len(parts) > 2:
            max_subdomains = max(max_subdomains, len(parts) - 2)
        if _has_suspicious_pattern(url, domain):
            has_suspicious_pattern = 1
        if bare in URL_SHORTENERS or any(
            bare == s or bare.endswith("." + s) for s in URL_SHORTENERS
        ):
            has_url_shortener = 1

    text_len = max(len(text), 1)
    total_url_chars = sum(len(u) for u in urls)
    url_to_text_ratio = total_url_chars / text_len if num_urls else 0.0

    return {
        "num_urls": float(num_urls),
        "max_url_length": float(max_url_length),
        "num_unique_urls": float(len(unique_urls)),
        "num_ip_urls": float(num_ip_urls),
        "max_num_subdomains": float(max_subdomains),
        "num_dots": float(num_dots),
        "num_hyphens": float(num_hyphens),
        "num_suspicious_chars": float(num_suspicious_chars),
        "num_query_params": float(num_query_params),
        "has_https": float(has_https),
        "has_suspicious_pattern": float(has_suspicious_pattern),
        "has_url_shortener": float(has_url_shortener),
        "url_to_text_ratio": float(url_to_text_ratio),
    }


def url_features_to_matrix(texts) -> np.ndarray:
    """Convert an iterable of raw texts to a (n_samples, n_features) array."""
    rows = [[extract_url_features(t)[name] for name in FEATURE_NAMES] for t in texts]
    if not rows:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=float)
    return np.asarray(rows, dtype=float)


class UrlFeatureExtractor(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible stateless URL feature transformer."""

    def fit(self, X, y=None):  # noqa: N803 - sklearn convention
        return self

    def transform(self, X):
        return url_features_to_matrix(list(X))
