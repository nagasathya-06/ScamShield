"""
Naive Bayes Feature Log-Likelihood Ratio Explainer.
Calculates token-level weights: log(P(token | Scam) / P(token | Safe))
"""

from typing import List, Tuple, Dict
import numpy as np
import re


class NaiveBayesWordExplainer:
    """
    Computes exact word-level log-odds contributions under the Naive Bayes assumption.
    """

    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer
        
        # Determine feature names and log probabilities
        if hasattr(vectorizer, 'get_feature_names'):
            self.feature_names = np.array(vectorizer.get_feature_names())
        elif hasattr(vectorizer, 'get_feature_names_out'):
            self.feature_names = np.array(vectorizer.get_feature_names_out())
        else:
            self.feature_names = np.array([])

        if hasattr(model, 'feature_log_prob_'):
            # log(P(w|scam)) - log(P(w|safe))
            self.log_ratio = model.feature_log_prob_[1] - model.feature_log_prob_[0]
        else:
            self.log_ratio = None

        # Build token lookup map for quick access
        self.token_to_idx = {name: i for i, name in enumerate(self.feature_names)}

    def explain_text(self, text: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Extracts tokens present in the text, looks up their log-ratio,
        and returns the top influential words for Scam and Safe.
        """
        words = re.findall(r'\b[a-zA-Z0-9_]+\b', text.lower())
        token_scores = []

        seen = set()
        for word in words:
            if word in seen:
                continue
            seen.add(word)

            # Check direct word token or prefixed token
            candidate_keys = [word, f"word__{word}"]
            found_idx = None
            for key in candidate_keys:
                if key in self.token_to_idx:
                    found_idx = self.token_to_idx[key]
                    break

            if found_idx is not None and self.log_ratio is not None:
                score = float(self.log_ratio[found_idx])
                token_scores.append((word, score))

        # Sort by score: positive = Scam indicator, negative = Safe indicator
        token_scores.sort(key=lambda x: x[1], reverse=True)
        
        scam_words = [t for t in token_scores if t[1] > 0][:top_k]
        safe_words = [t for t in reversed(token_scores) if t[1] < 0][:top_k]

        return {
            "all_tokens": token_scores,
            "scam_indicators": scam_words,
            "safe_indicators": safe_words
        }
