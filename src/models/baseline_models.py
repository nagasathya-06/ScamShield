"""
Baseline Machine Learning Models using Scikit-Learn.
Provides benchmark implementations for Naive Bayes, Logistic Regression, and kNN.
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from src.config import RANDOM_STATE


def build_sklearn_naive_bayes(alpha: float = 1.0) -> MultinomialNB:
    """Creates a Scikit-Learn Multinomial Naive Bayes classifier."""
    return MultinomialNB(alpha=alpha)


def build_sklearn_logistic_regression(
    C: float = 1.0,
    max_iter: int = 1000,
    class_weight: str = 'balanced'
) -> LogisticRegression:
    """Creates a Scikit-Learn Logistic Regression classifier with balanced class weighting."""
    return LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        solver='lbfgs'
    )


def build_knn_classifier(n_neighbors: int = 5, metric: str = 'cosine') -> KNeighborsClassifier:
    """Creates a k-Nearest Neighbours classifier using cosine distance for text features."""
    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric=metric
    )
