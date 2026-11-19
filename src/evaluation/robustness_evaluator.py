"""
Adversarial Robustness Evaluation Module.
Evaluates model vulnerability against character substitutions, leetspeak,
and delimiter insertions (e.g. 'K.Y.C', 'Bl0cked', 'cl1ck h3re').
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from src.config import PLOTS_DIR


def evaluate_model_robustness(
    model,
    clean_texts: List[str],
    obfuscated_texts: List[str],
    vectorizer,
    model_name: str
) -> Dict[str, Any]:
    """
    Compares clean vs obfuscated scam detection recall for a given model.
    Note: Both clean and obfuscated inputs are ground-truth SCAM (label 1).
    """
    X_clean = vectorizer.transform(clean_texts)
    X_obf = vectorizer.transform(obfuscated_texts)

    y_clean_pred = model.predict(X_clean)
    y_obf_pred = model.predict(X_obf)

    # True labels are all 1 (Scam)
    y_true = np.ones(len(clean_texts), dtype=int)

    clean_recall = recall_score(y_true, y_clean_pred, zero_division=0)
    obf_recall = recall_score(y_true, y_obf_pred, zero_division=0)

    recall_drop = clean_recall - obf_recall
    robustness_retention = (obf_recall / clean_recall * 100.0) if clean_recall > 0 else 0.0

    return {
        "model_name": model_name,
        "clean_scam_recall": float(clean_recall),
        "obfuscated_scam_recall": float(obf_recall),
        "recall_drop": float(recall_drop),
        "retention_rate_pct": float(robustness_retention),
        "clean_missed_count": int(np.sum(y_clean_pred == 0)),
        "obfuscated_missed_count": int(np.sum(y_obf_pred == 0)),
        "total_test_samples": len(clean_texts)
    }


def plot_robustness_comparison(
    robustness_results: List[Dict[str, Any]],
    save_path: str = None
) -> plt.Figure:
    """
    Plots grouped bar chart of Clean Recall vs Obfuscated Recall across models.
    """
    df = pd.DataFrame(robustness_results)
    fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.arange(len(df))
    width = 0.35

    rects1 = ax.bar(x - width/2, df['clean_scam_recall'] * 100, width, label='Clean Messages Recall (%)', color='#2b5c8f')
    rects2 = ax.bar(x + width/2, df['obfuscated_scam_recall'] * 100, width, label='Obfuscated Messages Recall (%)', color='#e05d44')

    ax.set_ylabel('Scam Detection Recall (%)', fontweight='bold')
    ax.set_title('Robustness Evaluation: Clean vs Obfuscated Adversarial Attacks', fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(df['model_name'], rotation=15, ha='right', fontweight='bold')
    ax.set_ylim(0, 110)
    ax.legend(loc='lower left')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig
