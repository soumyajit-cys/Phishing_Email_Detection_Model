"""Tests for static URL feature extraction."""

from src.url_features import (
    FEATURE_NAMES,
    UrlFeatureExtractor,
    extract_url_features,
    extract_urls,
)


def test_extract_urls_finds_http_links():
    urls = extract_urls("Visit http://evil.com/login and https://safe.com today")
    assert len(urls) == 2


def test_ip_url_detected():
    feats = extract_url_features("Click http://192.168.1.1/secure login")
    assert feats["num_ip_urls"] >= 1


def test_no_urls_gives_zeros():
    feats = extract_url_features("Hello, just checking in about tomorrow.")
    assert feats["num_urls"] == 0
    assert feats["max_url_length"] == 0
    assert feats["has_https"] == 0


def test_suspicious_tld_detected():
    feats = extract_url_features("Claim prize at http://winner-prize.xyz/claim")
    assert feats["has_suspicious_tld"] == 1


def test_feature_names_complete():
    feats = extract_url_features("https://example.com")
    assert set(feats.keys()) == set(FEATURE_NAMES)


def test_sklearn_transformer_shape():
    ext = UrlFeatureExtractor()
    mat = ext.fit_transform(["http://a.com", "plain text"])
    assert mat.shape == (2, len(FEATURE_NAMES))
