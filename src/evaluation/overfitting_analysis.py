"""
Overfitting Analysis, Learning Curves, and Cross-Validation Module.
Diagnoses bias vs variance and monitors train-val generalization gap.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, StratifiedKFold, cross_validate
from src.config import RANDOM_STATE, PLOTS_DIR


def evaluate_overfitting_gap(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Computes Train vs Validation metrics and quantifies the Generalization Gap:
    Gap = Train_F1 - Val_F1. A large gap (> 0.05) indicates overfitting.
    """
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    from sklearn.metrics import accuracy_score, f1_score, recall_score

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)

    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred, zero_division=0)

    train_recall = recall_score(y_train, y_train_pred, zero_division=0)
    val_recall = recall_score(y_val, y_val_pred, zero_division=0)

    gap = train_f1 - val_f1

    diagnosis = "Optimal Generalization"
    if gap > 0.08:
        diagnosis = "High Variance (Overfitting)"
    elif val_f1 < 0.80 and train_f1 < 0.85:
        diagnosis = "High Bias (Underfitting)"

    return {
        "model_name": model_name,
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "train_f1": float(train_f1),
        "val_f1": float(val_f1),
        "train_recall": float(train_recall),
        "val_recall": float(val_recall),
        "generalization_gap": float(gap),
        "diagnosis": diagnosis
    }


def compute_learning_curve(
    estimator,
    X,
    y,
    cv: int = 5,
    train_sizes: np.ndarray = np.linspace(0.15, 1.0, 6)
) -> Dict[str, np.ndarray]:
    """
    Computes learning curve data across training sizes using StratifiedKFold.
    """
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    
    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv_splitter,
        train_sizes=train_sizes,
        scoring='f1',
        n_jobs=1,
        random_state=RANDOM_STATE
    )

    return {
        "train_sizes": train_sizes_abs,
        "train_scores_mean": np.mean(train_scores, axis=1),
        "train_scores_std": np.std(train_scores, axis=1),
        "val_scores_mean": np.mean(test_scores, axis=1),
        "val_scores_std": np.std(test_scores, axis=1)
    }


def plot_learning_curve(
    lc_data: Dict[str, np.ndarray],
    model_name: str,
    save_path: str = None
) -> plt.Figure:
    """
    Generates and styles a learning curve plot.
    """
    train_sizes = lc_data["train_sizes"]
    train_mean = lc_data["train_scores_mean"]
    train_std = lc_data["train_scores_std"]
    val_mean = lc_data["val_scores_mean"]
    val_std = lc_data["val_scores_std"]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.plot(train_sizes, train_mean, 'o-', color='#1f77b4', label='Training F1-Score', lw=2)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#1f77b4')

    ax.plot(train_sizes, val_mean, 's-', color='#2ca02c', label='Validation F1-Score (CV)', lw=2)
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='#2ca02c')

    ax.set_title(f"Learning Curve: {model_name}", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Training Examples", fontweight='bold')
    ax.set_ylabel("F1 Score", fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig
