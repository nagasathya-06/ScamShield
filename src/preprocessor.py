"""
Text Preprocessing and Rule-Based Feature Extraction Module.
Cleans raw SMS/WhatsApp text and extracts domain-specific financial fraud indicators.
"""

import re
from typing import Dict, Any, List
import numpy as np
from src.config import (
    URGENCY_KEYWORDS,
    FINANCIAL_LURE_KEYWORDS,
    SENSITIVE_REQUEST_KEYWORDS,
    URL_SHORTENERS
)

# Regex Patterns
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
PHONE_PATTERN = re.compile(r'(?:\+91[\-\s]?)?[6-9]\d{9}\b|\b\d{10}\b|\b\d{5}[\s\-]\d{5}\b')
CURRENCY_PATTERN = re.compile(r'(?:rs\.?|inr|₹)\s*[\d,]+(?:\.\d{2})?|\b[\d,]+\s*(?:rupees|lakh|lakhs|crore|crores)\b', re.IGNORECASE)
LEET_CHAR_PATTERN = re.compile(r'[@$0-9!|€\+\<\>]')
DOTTED_WORD_PATTERN = re.compile(r'\b[A-Za-z0-9](\.[A-Za-z0-9])+\b')


def clean_text_for_word_model(text: str) -> str:
    """
    Standard text normalization for word-level models:
    - Lowercase
    - Replace URLs with placeholder token
    - Replace phone numbers with placeholder token
    - Replace currency amounts with placeholder token
    - Strip punctuation and excess whitespace
    """
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    text = URL_PATTERN.sub(' url_token ', text)
    text = PHONE_PATTERN.sub(' phone_token ', text)
    text = CURRENCY_PATTERN.sub(' money_token ', text)
    
    # Remove special characters but keep basic word tokens
    text = re.sub(r'[^a-z0-9\s_]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_text_for_char_model(text: str) -> str:
    """
    Text preparation for character n-gram models:
    Preserves character substitutions, punctuation, and case nuances
    which are essential for detecting obfuscation (e.g. 'Bl0cked', 'K.Y.C').
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class RuleBasedFeatureExtractor:
    """
    Extracts handcrafted domain-specific features specifically designed
    for Indian UPI, KYC, and Banking scam detection.
    """
    
    FEATURE_NAMES = [
        "has_url",
        "has_shortened_url",
        "url_count",
        "has_phone_number",
        "phone_count",
        "has_currency",
        "urgency_score",
        "financial_lure_score",
        "sensitive_request_flag",
        "obfuscation_char_ratio",
        "has_dotted_words",
        "caps_ratio",
        "char_length",
        "word_count"
    ]
    
    def __init__(self):
        pass

    def extract_features(self, text: str) -> Dict[str, float]:
        """
        Extracts a dictionary of numerical rule-based flags for a single message.
        """
        if not isinstance(text, str):
            text = ""

        lower_text = text.lower()
        total_chars = max(len(text), 1)
        words = text.split()
        total_words = max(len(words), 1)

        # 1. URL Analysis
        urls = URL_PATTERN.findall(text)
        url_count = len(urls)
        has_url = 1.0 if url_count > 0 else 0.0
        
        has_shortened_url = 0.0
        for shortener in URL_SHORTENERS:
            if shortener in lower_text:
                has_shortened_url = 1.0
                break

        # 2. Phone Numbers
        phones = PHONE_PATTERN.findall(text)
        phone_count = len(phones)
        has_phone_number = 1.0 if phone_count > 0 else 0.0

        # 3. Currency / Monetary lure
        has_currency = 1.0 if bool(CURRENCY_PATTERN.search(text)) else 0.0

        # 4. Keyword Frequencies
        urgency_score = sum(1 for kw in URGENCY_KEYWORDS if kw in lower_text)
        financial_lure_score = sum(1 for kw in FINANCIAL_LURE_KEYWORDS if kw in lower_text)
        sensitive_request_flag = 1.0 if any(kw in lower_text for kw in SENSITIVE_REQUEST_KEYWORDS) else 0.0

        # 5. Obfuscation Indicators
        leet_chars = len(LEET_CHAR_PATTERN.findall(text))
        obfuscation_char_ratio = leet_chars / total_chars
        has_dotted_words = 1.0 if bool(DOTTED_WORD_PATTERN.search(text)) else 0.0

        # 6. Styling / Formatting
        caps_count = sum(1 for c in text if c.isupper())
        caps_ratio = caps_count / total_chars

        return {
            "has_url": has_url,
            "has_shortened_url": has_shortened_url,
            "url_count": float(url_count),
            "has_phone_number": has_phone_number,
            "phone_count": float(phone_count),
            "has_currency": has_currency,
            "urgency_score": float(urgency_score),
            "financial_lure_score": float(financial_lure_score),
            "sensitive_request_flag": sensitive_request_flag,
            "obfuscation_char_ratio": float(obfuscation_char_ratio),
            "has_dotted_words": has_dotted_words,
            "caps_ratio": float(caps_ratio),
            "char_length": float(len(text)),
            "word_count": float(total_words)
        }

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transforms a collection of texts into a 2D NumPy array of rule features.
        """
        features_list = []
        for text in texts:
            feat_dict = self.extract_features(text)
            features_list.append([feat_dict[name] for name in self.FEATURE_NAMES])
        return np.array(features_list, dtype=np.float32)

    def fit(self, texts: List[str] = None, y=None):
        return self

    def fit_transform(self, texts: List[str], y=None) -> np.ndarray:
        return self.transform(texts)
