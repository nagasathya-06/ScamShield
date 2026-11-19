"""
Configuration module for Scam Shield project.
Holds paths, seeds, thresholds, and model hyperparameters.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

# Dataset Paths
SMS_RAW_PATH = RAW_DATA_DIR / "sms.tsv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_DATA_PATH = PROCESSED_DATA_DIR / "val.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"
OBFUSCATED_TEST_PATH = PROCESSED_DATA_DIR / "obfuscated_test.csv"
FULL_DATASET_PATH = PROCESSED_DATA_DIR / "full_dataset.csv"

# Model Artifact Paths
VECTORIZER_PATH = MODELS_DIR / "hybrid_vectorizer.joblib"
RULE_EXTRACTOR_PATH = MODELS_DIR / "rule_extractor.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"

# Reproducibility
RANDOM_STATE = 42

# Train / Val / Test Split Ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Class Labels
LABEL_SAFE = 0
LABEL_SCAM = 1
LABEL_NAMES = {0: "Safe", 1: "Scam"}

# Rule-based Keywords & Regex Patterns for Indian Financial Context
URGENCY_KEYWORDS = [
    "urgent", "urgently", "immediately", "immediate", "suspended", "blocked",
    "block", "terminate", "disconnect", "expire", "expires", "expiring", "24 hours",
    "12 hours", "today", "tonight", "warning", "final notice", "action required",
    "deactivate", "deactivated", "deadline", "turant", "jaldi", "khatre", "dhyan de"
]

FINANCIAL_LURE_KEYWORDS = [
    "lottery", "winner", "prize", "won", "crore", "lakh", "lakhs", "cashback",
    "reward", "bonus", "claim", "free", "approved", "refund", "credited",
    "part time", "earn", "daily income", "investment", "work from home",
    "badhai", "inaam", "paise", "kamaye"
]

SENSITIVE_REQUEST_KEYWORDS = [
    "otp", "pin", "cvv", "password", "netbanking", "mpin", "upi pin",
    "card number", "expiry", "aadhaar", "pan card", "pan update",
    "kyc update", "verify kyc", "complete kyc", "biometric"
]

URL_SHORTENERS = [
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "is.gd", "buff.ly",
    "ow.ly", "cutt.ly", "rb.gy", "shorturl.at", "wa.me", "t.me"
]

# Obfuscation Leetspeak Map
LEET_MAP = {
    'a': ['@', '4', '/\\'],
    'e': ['3', '€'],
    'i': ['1', '!', '|'],
    'l': ['1', '|', 'I'],
    'o': ['0', '()', 'Q'],
    's': ['$', '5'],
    't': ['7', '+'],
    'b': ['8', '6'],
    'c': ['(', '<']
}
