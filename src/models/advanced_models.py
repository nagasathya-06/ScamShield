"""
Advanced Machine Learning Models for Scam Detection.
Implements Random Forest and Gradient Boosting (XGBoost).
Addresses class imbalance and hyperparameter tuning.
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from src.config import RANDOM_STATE


def build_random_forest(
    n_estimators: int = 150,
    max_depth: int = 30,
    class_weight: str = 'balanced',
    random_state: int = RANDOM_STATE
) -> RandomForestClassifier:
    """Creates a Random Forest classifier optimized for text spam/scam classification."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1
    )


def build_xgboost_classifier(
    n_estimators: int = 150,
    learning_rate: float = 0.1,
    max_depth: int = 5,
    scale_pos_weight: float = 3.0,
    random_state: int = RANDOM_STATE
) -> XGBClassifier:
    """
    Creates an XGBoost Gradient Boosted Trees classifier.
    scale_pos_weight addresses class imbalance by upweighting the positive (scam) class.
    """
    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=random_state,
        n_jobs=-1
    )


def build_gradient_boosting(
    n_estimators: int = 150,
    learning_rate: float = 0.1,
    max_depth: int = 5,
    random_state: int = RANDOM_STATE
) -> GradientBoostingClassifier:
    """Fallback Sklearn Gradient Boosting Classifier."""
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state
    )
