"""Tests for text preprocessing."""

from src.preprocessing import clean_dataframe, clean_text, combine_subject_body
import pandas as pd
import pytest


def test_clean_text_lowercases_and_strips_html():
    out = clean_text("<p>Hello  WORLD</p>  ")
    assert out == "hello world"


def test_clean_text_replaces_url_with_token():
    out = clean_text("Click http://evil.com/login now")
    assert "urltoken" in out
    assert "http" not in out


def test_combine_subject_body():
    assert combine_subject_body("Hi", "there") == "Hi there"
    assert combine_subject_body(None, "body") == "body"
    assert combine_subject_body("", "") == ""


def test_clean_dataframe_validates_columns():
    with pytest.raises(ValueError, match="Missing"):
        clean_dataframe(pd.DataFrame({"foo": [1]}))


def test_clean_dataframe_rejects_bad_labels():
    df = pd.DataFrame(
        {"subject": ["a"], "email_text": ["hello"], "label": [5]}
    )
    with pytest.raises(ValueError, match="Invalid labels"):
        clean_dataframe(df)
