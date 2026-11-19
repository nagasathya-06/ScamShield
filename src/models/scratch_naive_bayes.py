"""
From-Scratch Implementation of Multinomial Naive Bayes using NumPy.
Meets the requirements of Stage 3: Baseline Model Development.
"""

import numpy as np
from scipy.sparse import issparse


class ScratchNaiveBayes:
    """
    Multinomial Naive Bayes Classifier implemented from scratch using NumPy.
    Supports Laplace (additive) smoothing, sparse feature inputs,
    and numerically stable log-space calculations using Log-Sum-Exp.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.classes_ = None
        self.class_count_ = None
        self.class_log_prior_ = None
        self.feature_count_ = None
        self.feature_log_prob_ = None
        self.n_features_in_ = None

    def fit(self, X, y):
        """
        Fits the Naive Bayes model according to X and y.
        
        Parameters:
        -----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Training vectors (e.g. TF-IDF counts or frequencies)
        y : array-like of shape (n_samples,)
            Target class labels (0 or 1)
        """
        y = np.asarray(y)
        self.classes_, class_indices = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        n_samples, self.n_features_in_ = X.shape

        self.class_count_ = np.bincount(class_indices, minlength=n_classes)
        # Class log priors: log(N_c / N)
        self.class_log_prior_ = np.log(self.class_count_ / n_samples)

        # Compute feature counts per class: sum of token counts for samples in each class
        self.feature_count_ = np.zeros((n_classes, self.n_features_in_), dtype=np.float64)

        if issparse(X):
            for idx, c in enumerate(self.classes_):
                mask = (y == c)
                # Matrix multiplication on subset
                self.feature_count_[idx, :] = np.asarray(X[mask].sum(axis=0)).ravel()
        else:
            for idx, c in enumerate(self.classes_):
                mask = (y == c)
                self.feature_count_[idx, :] = X[mask].sum(axis=0)

        # Laplace (additive) smoothing:
        # P(x_i | c) = (count(x_i, c) + alpha) / (total_tokens_c + alpha * n_features)
        smoothed_fc = self.feature_count_ + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1, keepdims=True)
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc)

        return self

    def _joint_log_likelihood(self, X) -> np.ndarray:
        """
        Calculates the joint log-likelihood:
        log P(c) + sum_i [ x_i * log P(x_i | c) ]
        """
        if issparse(X):
            # X is (n_samples, n_features), feature_log_prob_.T is (n_features, n_classes)
            joint_log_lik = X.dot(self.feature_log_prob_.T) + self.class_log_prior_
            return np.asarray(joint_log_lik)
        else:
            return np.dot(X, self.feature_log_prob_.T) + self.class_log_prior_

    def predict_log_proba(self, X) -> np.ndarray:
        """
        Computes normalized log-probabilities using the Log-Sum-Exp trick:
        log P(c | X) = joint_ll(c) - logsumexp(joint_ll)
        """
        joint_ll = self._joint_log_likelihood(X)
        # Log-Sum-Exp trick for numerical stability:
        max_ll = np.max(joint_ll, axis=1, keepdims=True)
        log_prob_norm = joint_ll - (max_ll + np.log(np.sum(np.exp(joint_ll - max_ll), axis=1, keepdims=True)))
        return log_prob_norm

    def predict_proba(self, X) -> np.ndarray:
        """
        Returns probability estimates for each class.
        """
        return np.exp(self.predict_log_proba(X))

    def predict(self, X) -> np.ndarray:
        """
        Perform classification on an array of test vectors X.
        """
        joint_ll = self._joint_log_likelihood(X)
        best_class_indices = np.argmax(joint_ll, axis=1)
        return self.classes_[best_class_indices]

    def get_feature_importance_ratio(self) -> np.ndarray:
        """
        Returns the log-likelihood ratio for binary classification:
        log(P(x_i | Scam) / P(x_i | Safe)) = log P(x_i | 1) - log P(x_i | 0)
        Positive values indicate words strongly indicative of Scam.
        Negative values indicate words strongly indicative of Safe.
        """
        if len(self.classes_) == 2:
            return self.feature_log_prob_[1] - self.feature_log_prob_[0]
        return None
