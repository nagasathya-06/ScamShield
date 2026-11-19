"""
Evaluation Metrics and Reporting Module.
Computes comprehensive classification metrics with particular focus on Scam Recall,
and generates publication-quality visualization figures.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)
from src.config import PLOTS_DIR


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Computes standard and fraud-specific performance metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Specificity (True Negative Rate for Safe class)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Scam Recall": float(rec),
        "Specificity": float(specificity),
        "F1-Score": float(f1),
        "False Negatives (Missed Scams)": int(fn),
        "False Positives (False Alarms)": int(fp),
        "True Positives (Caught Scams)": int(tp),
        "True Negatives (Correct Safe)": int(tn)
    }

    if y_proba is not None:
        try:
            if y_proba.ndim == 2:
                prob_pos = y_proba[:, 1]
            else:
                prob_pos = y_proba
            metrics["ROC-AUC"] = float(roc_auc_score(y_true, prob_pos))
        except Exception:
            metrics["ROC-AUC"] = 0.0

    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots a styled confusion matrix with annotations.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["Safe (0)", "Scam (1)"]
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax
    )
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Predicted Label", fontweight='bold')
    ax.set_ylabel("True Label", fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_roc_curves(
    roc_dict: Dict[str, Any],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots multi-model ROC Curves for side-by-side comparison.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.50)')

    for name, data in roc_dict.items():
        fpr, tpr, auc_val = data['fpr'], data['tpr'], data['auc']
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", lw=2)

    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=12, fontweight='bold')
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Scam Recall)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig
