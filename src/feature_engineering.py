"""
Feature Engineering Module for Scam Shield.
Implements Word TF-IDF, Character N-gram TF-IDF, and Hybrid Feature Pipelines.
"""

from typing import Tuple, List, Optional
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from src.preprocessor import (
    clean_text_for_word_model,
    clean_text_for_char_model,
    RuleBasedFeatureExtractor
)


class HybridFeaturePipeline:
    """
    Combines:
    1. Word-level TF-IDF (captures semantic keywords like 'urgent', 'otp', 'cashback')
    2. Character n-gram TF-IDF (captures obfuscation, leetspeak like 'Bl0cked', 'K.Y.C')
    3. Handcrafted rule-based domain flags (URL shortenings, urgency scores, credential requests)
    """

    def __init__(
        self,
        word_max_features: int = 3000,
        char_max_features: int = 3000,
        word_ngram_range: Tuple[int, int] = (1, 2),
        char_ngram_range: Tuple[int, int] = (3, 5),
        include_rules: bool = True
    ):
        self.word_max_features = word_max_features
        self.char_max_features = char_max_features
        self.word_ngram_range = word_ngram_range
        self.char_ngram_range = char_ngram_range
        self.include_rules = include_rules

        self.word_vectorizer = TfidfVectorizer(
            ngram_range=self.word_ngram_range,
            max_features=self.word_max_features,
            sublinear_tf=True
        )

        self.char_vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=self.char_ngram_range,
            max_features=self.char_max_features,
            sublinear_tf=True
        )

        self.rule_extractor = RuleBasedFeatureExtractor() if include_rules else None
        # Use MinMaxScaler so that all feature matrices are strictly non-negative (compatible with MultinomialNB)
        self.scaler = MinMaxScaler(feature_range=(0, 1)) if include_rules else None
        self.feature_names_: List[str] = []

    def fit(self, texts: List[str], y=None):
        """
        Fit all vectorizers and scalers on raw training messages.
        """
        cleaned_words = [clean_text_for_word_model(t) for t in texts]
        cleaned_chars = [clean_text_for_char_model(t) for t in texts]

        self.word_vectorizer.fit(cleaned_words)
        self.char_vectorizer.fit(cleaned_chars)

        word_names = [f"word__{w}" for w in self.word_vectorizer.get_feature_names_out()]
        char_names = [f"char__{c}" for c in self.char_vectorizer.get_feature_names_out()]

        rule_names = []
        if self.include_rules:
            rule_features = self.rule_extractor.transform(texts)
            self.scaler.fit(rule_features)
            rule_names = [f"rule__{r}" for r in self.rule_extractor.FEATURE_NAMES]

        self.feature_names_ = word_names + char_names + rule_names
        return self

    def transform(self, texts: List[str]) -> csr_matrix:
        """
        Transforms raw text messages into a combined sparse feature matrix.
        """
        cleaned_words = [clean_text_for_word_model(t) for t in texts]
        cleaned_chars = [clean_text_for_char_model(t) for t in texts]

        X_word = self.word_vectorizer.transform(cleaned_words)
        X_char = self.char_vectorizer.transform(cleaned_chars)

        matrices_to_stack = [X_word, X_char]

        if self.include_rules:
            rule_features = self.rule_extractor.transform(texts)
            rule_scaled = self.scaler.transform(rule_features)
            matrices_to_stack.append(csr_matrix(rule_scaled))

        return hstack(matrices_to_stack).tocsr()

    def fit_transform(self, texts: List[str], y=None) -> csr_matrix:
        self.fit(texts, y)
        return self.transform(texts)

    def get_feature_names(self) -> List[str]:
        return self.feature_names_
