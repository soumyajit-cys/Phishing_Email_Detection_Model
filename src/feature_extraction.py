"""Feature extraction: TF-IDF text + numeric URL features."""

from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .preprocessing import clean_text
from .url_features import UrlFeatureExtractor

TFIDF_MAX_FEATURES = 5000


def build_tfidf_vectorizer(
    max_features: int = TFIDF_MAX_FEATURES,
) -> TfidfVectorizer:
    """Build a TF-IDF vectorizer that reuses clean_text as preprocessor."""
    return TfidfVectorizer(
        preprocessor=clean_text,
        lowercase=False,  # clean_text already lowercases
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
    )


class TextAndUrlFeatures(BaseEstimator, TransformerMixin):
    """Combine TF-IDF text features with scaled URL features.

    Fitted ONLY on training data when used inside a Pipeline to
    prevent data leakage. Produces a sparse matrix compatible
    with linear models and tree models.
    """

    def __init__(self, max_features: int = TFIDF_MAX_FEATURES):
        self.max_features = max_features

    def fit(self, X, y=None):  # noqa: N803 - sklearn convention
        texts = list(X)
        self.tfidf_ = build_tfidf_vectorizer(max_features=self.max_features)
        self.tfidf_.fit(texts)
        self.url_extractor_ = UrlFeatureExtractor()
        url_matrix = self.url_extractor_.transform(texts)
        self.scaler_ = StandardScaler()
        self.scaler_.fit(url_matrix)
        return self

    def transform(self, X):
        texts = list(X)
        text_matrix = self.tfidf_.transform(texts)
        url_matrix = self.url_extractor_.transform(texts)
        url_scaled = self.scaler_.transform(url_matrix)
        return hstack([text_matrix, csr_matrix(url_scaled)], format="csr")

    def get_feature_names(self):
        tfidf_names = list(self.tfidf_.get_feature_names_out())
        from .url_features import FEATURE_NAMES

        return tfidf_names + [f"URL_{n}" for n in FEATURE_NAMES]


def build_text_pipeline(classifier) -> Pipeline:
    """Text-only pipeline (used for MultinomialNB, which needs non-negative input)."""
    return Pipeline(
        [
            ("tfidf", build_tfidf_vectorizer()),
            ("clf", classifier),
        ]
    )


def build_combined_pipeline(classifier) -> Pipeline:
    """Combined TF-IDF + URL numeric feature pipeline (used for LR / RF)."""
    return Pipeline(
        [
            ("features", TextAndUrlFeatures()),
            ("clf", classifier),
        ]
    )
