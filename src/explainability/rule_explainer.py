"""
Rule-Based Threat Diagnostics and Heuristic Explainer.
Analyzes specific financial scam patterns, suspicious URLs,
and sensitive credential requests.
"""

from typing import List, Dict, Any
import re
from src.config import (
    URGENCY_KEYWORDS,
    FINANCIAL_LURE_KEYWORDS,
    SENSITIVE_REQUEST_KEYWORDS,
    URL_SHORTENERS
)
from src.preprocessor import URL_PATTERN, PHONE_PATTERN, CURRENCY_PATTERN, LEET_CHAR_PATTERN, DOTTED_WORD_PATTERN


def diagnose_threats(text: str) -> List[Dict[str, Any]]:
    """
    Returns an actionable list of detected threat indicators with severity and reason.
    """
    findings = []
    lower_text = text.lower()

    # 1. URL Shortener Detection
    found_shorteners = [s for s in URL_SHORTENERS if s in lower_text]
    if found_shorteners:
        findings.append({
            "category": "Suspicious Link",
            "severity": "HIGH",
            "title": "Shortened or Redirect URL Detected",
            "description": f"Scammers frequently use URL shorteners ({', '.join(found_shorteners)}) to mask malicious phishing domains.",
            "color": "#e74c3c"
        })
    elif URL_PATTERN.search(text):
        findings.append({
            "category": "Link Presence",
            "severity": "MEDIUM",
            "title": "Embedded Hyperlink Found",
            "description": "Message contains a URL. Legitimate banks rarely send direct action links in unsolicited SMS.",
            "color": "#f39c12"
        })

    # 2. Sensitive Credential Inquiries (OTP, PIN, Password, NetBanking)
    matched_creds = [k for k in SENSITIVE_REQUEST_KEYWORDS if k in lower_text]
    if matched_creds:
        findings.append({
            "category": "Credential Threat",
            "severity": "CRITICAL",
            "title": "Request for Sensitive Credentials",
            "description": f"Contains prompts for sensitive security items ({', '.join(matched_creds)}). No bank or genuine authority asks for PIN/OTP.",
            "color": "#c0392b"
        })

    # 3. Urgency / Coercion Keywords
    matched_urgency = [k for k in URGENCY_KEYWORDS if k in lower_text]
    if matched_urgency:
        findings.append({
            "category": "Psychological Coercion",
            "severity": "HIGH",
            "title": "Urgent Action Required",
            "description": f"Message induces panic or urgency using terms: '{', '.join(matched_urgency)}'.",
            "color": "#e67e22"
        })

    # 4. Financial / Lottery Lures
    matched_lures = [k for k in FINANCIAL_LURE_KEYWORDS if k in lower_text]
    if matched_lures:
        findings.append({
            "category": "Financial Lure",
            "severity": "MEDIUM",
            "title": "Unrealistic Reward / Prize",
            "description": f"Prompts unexpected financial gain, lottery, or refund: '{', '.join(matched_lures)}'.",
            "color": "#d35400"
        })

    # 5. Obfuscation & Character Manipulation
    leet_matches = LEET_CHAR_PATTERN.findall(text)
    dotted_matches = DOTTED_WORD_PATTERN.findall(text)
    if len(leet_matches) > 3 or dotted_matches:
        findings.append({
            "category": "Evasion Technique",
            "severity": "HIGH",
            "title": "Adversarial Text Obfuscation",
            "description": "Detected leetspeak character substitutions (e.g., 0 for o, @ for a) or dotted words (e.g. K.Y.C) attempting to evade filters.",
            "color": "#8e44ad"
        })

    return findings
