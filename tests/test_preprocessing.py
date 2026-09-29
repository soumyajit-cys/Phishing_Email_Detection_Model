"""Tests for text preprocessing."""

import pandas as pd
import pytest

from src.preprocessing import clean_dataframe, clean_text, combine_subject_body


def test_clean_text_lowercases_and_strips_html():
    out = clean_text("<p>Hello  WORLD</p>  ")
    assert out == "hello world"


def test_clean_text_replaces_url_with_token():
    out = clean_text("Click http://evil.com/login now")
    assert "urltoken" in out
    assert "http" not in out


def test_clean_text_preserves_numbers_and_words():
    out = clean_text("Invoice 12345: verify your account, winner!")
    assert "12345" in out
    assert "verify" in out
    assert "winner" in out


def test_combine_subject_body():
    assert combine_subject_body("Hi", "there") == "Hi there"
    assert combine_subject_body(None, "body") == "body"
    assert combine_subject_body("", "") == ""


def test_empty_email_combination():
    assert combine_subject_body("", "") == ""


def test_normal_email_cleaning():
    out = clean_text("Hi team, meeting tomorrow at 10 AM. Thanks!")
    assert "meeting" in out
    assert out == out.lower()


def test_clean_dataframe_validates_columns():
    with pytest.raises(ValueError, match="Missing"):
        clean_dataframe(pd.DataFrame({"foo": [1]}))


def test_clean_dataframe_rejects_bad_labels():
    df = pd.DataFrame({"subject": ["a"], "body": ["hello"], "label": [5]})
    with pytest.raises(ValueError, match="Invalid labels"):
        clean_dataframe(df)


def test_clean_dataframe_accepts_legacy_email_text_column():
    df = pd.DataFrame({"subject": ["a"], "email_text": ["hello"], "label": [0]})
    out = clean_dataframe(df)
    assert list(out.columns) == ["subject", "body", "label", "combined_text"]
