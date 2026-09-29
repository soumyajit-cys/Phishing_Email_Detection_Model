"""Static URL feature extraction (no network requests).

Only static string/regex analysis is performed. URLs are never
visited, downloaded, or opened.
"""

import re
from typing import Dict, List
from urllib.parse import urlparse

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

URL_REGEX = re.compile(r"https?://[^\s<>'\"]+|www\.[^\s<>'\"]+", re.IGNORECASE)
IP_URL_RE = re.compile(r"https?://\d{1,3}(?:\.\d{1,3}){3}")
SUSPICIOUS_CHARS = set("@-_%&=~?#$!+")
REDIRECT_KEYWORDS = ("redirect", "forward", "verify", "login", "secure", "account")
SUSPICIOUS_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".click",
    ".link", ".work", ".country", ".stream", ".download", ".loan",
    ".win", ".bid", ".party", ".review",
}

FEATURE_NAMES: List[str] = [
    "num_urls",
    "num_unique_urls",
    "num_ip_urls",
    "num_suspicious_chars",
    "num_redirects",
    "has_https",
    "max_url_length",
    "max_num_subdomains",
    "has_suspicious_tld",
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


def extract_url_features(text: str) -> Dict[str, float]:
    """Extract numeric URL features from a single email string."""
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)

    urls = extract_urls(text)
    num_urls = len(urls)
    num_unique = len(set(u.lower() for u in urls))
    num_ip_urls = sum(1 for u in urls if IP_URL_RE.search(u))
    num_suspicious_chars = sum(u.count(c) for u in urls for c in SUSPICIOUS_CHARS)

    num_redirects = 0
    for url in urls:
        lower = url.lower()
        # Count '//' appearing after the protocol as a redirect/obfuscation hint.
        after_protocol = lower.split("://", 1)[-1] if "://" in lower else lower
        if "//" in after_protocol:
            num_redirects += 1
        if "@" in url:
            num_redirects += 1
        if any(k in lower for k in REDIRECT_KEYWORDS) and (
            "%" in lower or "@" in lower or "redirect" in lower
        ):
            num_redirects += 1

    has_https = int(any(u.lower().startswith("https://") for u in urls))
    max_url_length = max((len(u) for u in urls), default=0)

    max_subdomains = 0
    for url in urls:
        domain = _domain_of(url).split(":")[0].split("@")[-1]
        # Strip leading www.
        if domain.startswith("www."):
            domain = domain[4:]
        parts = [p for p in domain.split(".") if p]
        if len(parts) > 2:
            max_subdomains = max(max_subdomains, len(parts) - 2)

    has_suspicious_tld = 0
    for url in urls:
        domain = _domain_of(url).split(":")[0].split("@")[-1]
        if any(domain.endswith(tld) for tld in SUSPICIOUS_TLDS):
            has_suspicious_tld = 1
            break

    return {
        "num_urls": float(num_urls),
        "num_unique_urls": float(num_unique),
        "num_ip_urls": float(num_ip_urls),
        "num_suspicious_chars": float(num_suspicious_chars),
        "num_redirects": float(num_redirects),
        "has_https": float(has_https),
        "max_url_length": float(max_url_length),
        "max_num_subdomains": float(max_subdomains),
        "has_suspicious_tld": float(has_suspicious_tld),
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
