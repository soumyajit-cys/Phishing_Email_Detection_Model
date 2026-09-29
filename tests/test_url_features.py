"""Tests for static URL feature extraction."""

from src.text_features import (
    count_suspicious_keywords,
    find_suspicious_keywords,
)
from src.url_features import (
    FEATURE_NAMES,
    UrlFeatureExtractor,
    extract_url_features,
    extract_urls,
)


def test_extract_urls_finds_http_links():
    urls = extract_urls("Visit http://evil.com/login and https://safe.com today")
    assert len(urls) == 2


def test_extract_urls_empty_email():
    assert extract_urls("") == []
    assert extract_urls(None) == []


def test_ip_url_detected():
    feats = extract_url_features("Click http://192.168.1.1/secure login")
    assert feats["num_ip_urls"] >= 1


def test_no_urls_gives_zeros():
    feats = extract_url_features("Hello, just checking in about tomorrow.")
    assert feats["num_urls"] == 0
    assert feats["max_url_length"] == 0
    assert feats["has_https"] == 0
    assert feats["url_to_text_ratio"] == 0.0


def test_suspicious_pattern_detected():
    feats = extract_url_features("Claim prize at http://winner-prize.xyz/claim")
    assert feats["has_suspicious_pattern"] == 1


def test_https_presence():
    feats = extract_url_features("See https://example.com/docs")
    assert feats["has_https"] == 1


def test_dots_hyphens_query_params():
    feats = extract_url_features(
        "Go to http://sub1.sub2.example-site.com/path?x=1&y=2 now"
    )
    assert feats["num_dots"] >= 2
    assert feats["num_hyphens"] >= 1
    assert feats["num_query_params"] >= 1
    assert feats["max_num_subdomains"] >= 1


def test_url_shortener_detected():
    feats = extract_url_features("Click http://bit.ly/abc123 now")
    assert feats["has_url_shortener"] == 1


def test_url_ratio_positive():
    feats = extract_url_features("Click https://example.com/very/long/path here")
    assert feats["url_to_text_ratio"] > 0


def test_feature_names_complete():
    feats = extract_url_features("https://example.com")
    assert set(feats.keys()) == set(FEATURE_NAMES)


def test_sklearn_transformer_shape():
    ext = UrlFeatureExtractor()
    mat = ext.fit_transform(["http://a.com", "plain text"])
    assert mat.shape == (2, len(FEATURE_NAMES))


def test_suspicious_keyword_detection():
    matched = find_suspicious_keywords("Please verify your account password urgently")
    assert "verify" in matched
    assert "account" in matched
    assert count_suspicious_keywords("Hello team, lunch tomorrow?") == 0


def test_keyword_alone_is_not_a_classifier():
    # Keywords are signals only; a safe mail may contain one without being phishing.
    assert count_suspicious_keywords("Password reset tips: never share credentials") >= 1
