"""
Comprehensive Training, Validation, and Evaluation Pipeline for Scam Shield.
Trains baseline and advanced models, validates from-scratch implementations,
conducts overfitting analysis, evaluates adversarial robustness,
and exports all serialized artifacts.
"""

import sys
import json
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

from src.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    OBFUSCATED_TEST_PATH,
    MODELS_DIR,
    PLOTS_DIR,
    RANDOM_STATE
)
from src.feature_engineering import HybridFeaturePipeline
from src.models.scratch_naive_bayes import ScratchNaiveBayes
from src.models.scratch_logistic_regression import ScratchLogisticRegression
from src.models.baseline_models import (
    build_sklearn_naive_bayes,
    build_sklearn_logistic_regression,
    build_knn_classifier
)
from src.models.advanced_models import (
    build_random_forest,
    build_xgboost_classifier
)
from src.models.neural_model import build_neural_mlp
from src.evaluation.metrics import (
    compute_all_metrics,
    plot_confusion_matrix,
    plot_roc_curves
)
from src.evaluation.overfitting_analysis import (
    evaluate_overfitting_gap,
    compute_learning_curve,
    plot_learning_curve
)
from src.evaluation.robustness_evaluator import (
    evaluate_model_robustness,
    plot_robustness_comparison
)


def run_pipeline():
    print("=" * 70)
    print("SCAM SHIELD: ML TRAINING, VALIDATION & EVALUATION PIPELINE")
    print("=" * 70)

    # 1. Load Data
    print("\n[Stage 1] Loading Datasets...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    val_df = pd.read_csv(VAL_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    obf_df = pd.read_csv(OBFUSCATED_TEST_PATH)

    X_train_raw = train_df['message_text'].tolist()
    y_train = train_df['label'].values

    X_val_raw = val_df['message_text'].tolist()
    y_val = val_df['label'].values

    X_test_raw = test_df['message_text'].tolist()
    y_test = test_df['label'].values

    X_obf_raw = obf_df['message_text'].tolist()
    y_obf = obf_df['label'].values

    print(f"Loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}, Obfuscated={len(obf_df)}")

    # 2. Feature Extraction (Hybrid Word + Char + Rule Features)
    print("\n[Stage 2] Fitting Hybrid Feature Pipeline (Word TF-IDF + Char N-Grams + Rule Flags)...")
    feature_pipeline = HybridFeaturePipeline(
        word_max_features=2500,
        char_max_features=2500,
        word_ngram_range=(1, 2),
        char_ngram_range=(3, 5),
        include_rules=True
    )
    feature_pipeline.fit(X_train_raw)

    X_train = feature_pipeline.transform(X_train_raw)
    X_val = feature_pipeline.transform(X_val_raw)
    X_test = feature_pipeline.transform(X_test_raw)
    X_obf = feature_pipeline.transform(X_obf_raw)

    print(f"Feature Matrix Shape: Train={X_train.shape}, Total Features={X_train.shape[1]}")

    # Also build Word-only pipeline to demonstrate ablation in robustness
    word_pipeline = HybridFeaturePipeline(word_max_features=3000, char_max_features=0, include_rules=False)
    # Custom fit for word only:
    from sklearn.feature_extraction.text import TfidfVectorizer
    word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=3000)
    X_train_word = word_vec.fit_transform(X_train_raw)
    X_test_word = word_vec.transform(X_test_raw)
    X_obf_word = word_vec.transform(X_obf_raw)

    # 3. Model Training
    print("\n[Stage 3 & 4] Training Classifiers...")
    models = {}

    # (A) From-Scratch Naive Bayes
    print(" -> Training [From-Scratch] Multinomial Naive Bayes (NumPy)...")
    t0 = time.time()
    nb_scratch = ScratchNaiveBayes(alpha=1.0)
    nb_scratch.fit(X_train, y_train)
    models["Scratch Naive Bayes"] = nb_scratch
    print(f"    Done in {time.time() - t0:.2f}s")

    # (B) Scikit-Learn Naive Bayes
    print(" -> Training [Scikit-Learn] Multinomial Naive Bayes...")
    t0 = time.time()
    nb_sklearn = build_sklearn_naive_bayes(alpha=1.0)
    nb_sklearn.fit(X_train, y_train)
    models["Sklearn Naive Bayes"] = nb_sklearn
    print(f"    Done in {time.time() - t0:.2f}s")

    # (C) From-Scratch Logistic Regression
    print(" -> Training [From-Scratch] Logistic Regression (NumPy + Adam Optimizer)...")
    t0 = time.time()
    lr_scratch = ScratchLogisticRegression(learning_rate=0.08, n_iters=350, l2_reg=0.05, optimizer='adam')
    lr_scratch.fit(X_train, y_train)
    models["Scratch Logistic Regression"] = lr_scratch
    print(f"    Done in {time.time() - t0:.2f}s, Final Loss: {lr_scratch.loss_history_[-1]:.4f}")

    # (D) Scikit-Learn Logistic Regression
    print(" -> Training [Scikit-Learn] Logistic Regression (Balanced Weights)...")
    t0 = time.time()
    lr_sklearn = build_sklearn_logistic_regression(C=1.0, class_weight='balanced')
    lr_sklearn.fit(X_train, y_train)
    models["Sklearn Logistic Regression"] = lr_sklearn
    print(f"    Done in {time.time() - t0:.2f}s")

    # (E) Baseline k-Nearest Neighbours
    print(" -> Training [Baseline] k-Nearest Neighbours (k=5, Cosine Distance)...")
    t0 = time.time()
    knn = build_knn_classifier(n_neighbors=5, metric='cosine')
    knn.fit(X_train, y_train)
    models["k-Nearest Neighbours"] = knn
    print(f"    Done in {time.time() - t0:.2f}s")

    # (F) Advanced: Random Forest
    print(" -> Training [Advanced] Random Forest (150 Trees, Balanced Weights)...")
    t0 = time.time()
    rf = build_random_forest(n_estimators=150, max_depth=30, class_weight='balanced')
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf
    print(f"    Done in {time.time() - t0:.2f}s")

    # (G) Advanced: XGBoost Gradient Boosted Trees
    print(" -> Training [Advanced] XGBoost Gradient Boosting (scale_pos_weight=3.0)...")
    t0 = time.time()
    xgb = build_xgboost_classifier(n_estimators=150, learning_rate=0.1, max_depth=5, scale_pos_weight=3.0)
    xgb.fit(X_train, y_train)
    models["XGBoost"] = xgb
    print(f"    Done in {time.time() - t0:.2f}s")

    # (H) Advanced Extension: Neural MLP Text Classifier
    print(" -> Training [Neural] Multi-Layer Perceptron (128x64 Dense Architecture)...")
    t0 = time.time()
    mlp = build_neural_mlp(hidden_layer_sizes=(128, 64), max_iter=150)
    mlp.fit(X_train, y_train)
    models["Neural MLP"] = mlp
    print(f"    Done in {time.time() - t0:.2f}s")

    # 4. Scratch vs Sklearn Validation Check
    print("\n" + "=" * 70)
    print("[Validation Stage] Scikit-Learn vs From-Scratch Mathematical Verification")
    print("=" * 70)
    
    # NB Comparison
    nb_scratch_preds = nb_scratch.predict(X_test)
    nb_sklearn_preds = nb_sklearn.predict(X_test)
    nb_agreement = np.mean(nb_scratch_preds == nb_sklearn_preds) * 100.0
    print(f"Naive Bayes Agreement (Scratch vs Sklearn on Test Set): {nb_agreement:.2f}%")

    # LR Comparison
    lr_scratch_preds = lr_scratch.predict(X_test)
    lr_sklearn_preds = lr_sklearn.predict(X_test)
    lr_agreement = np.mean(lr_scratch_preds == lr_sklearn_preds) * 100.0
    print(f"Logistic Regression Agreement (Scratch vs Sklearn on Test Set): {lr_agreement:.2f}%")

    validation_verification = {
        "nb_agreement_pct": float(nb_agreement),
        "lr_agreement_pct": float(lr_agreement),
        "scratch_lr_final_loss": float(lr_scratch.loss_history_[-1]),
        "scratch_lr_initial_loss": float(lr_scratch.loss_history_[0])
    }

    # 5. Model Evaluation on Test Set
    print("\n" + "=" * 70)
    print("[Stage 5] Evaluating All Models on Independent Test Set...")
    print("=" * 70)

    test_metrics = {}
    roc_curves_data = {}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        t_start = time.time()
        y_pred = model.predict(X_test)
        latency_ms = (time.time() - t_start) / len(y_test) * 1000.0

        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            if y_proba.ndim == 2:
                prob_pos = y_proba[:, 1]
            else:
                prob_pos = y_proba
            fpr, tpr, _ = roc_curve(y_test, prob_pos)
            roc_auc = auc(fpr, tpr)
            roc_curves_data[name] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}

        metrics = compute_all_metrics(y_test, y_pred, y_proba)
        metrics["Inference Latency (ms/msg)"] = round(latency_ms, 3)
        test_metrics[name] = metrics

        # Generate and save individual confusion matrices
        cm = confusion_matrix(y_test, y_pred)
        safe_name = name.lower().replace(' ', '_').replace('-', '_')
        plot_confusion_matrix(cm, name, save_path=str(PLOTS_DIR / f"cm_{safe_name}.png"))
        plt.close()

    metrics_df = pd.DataFrame(test_metrics).T
    cols_display = ["Accuracy", "Precision", "Scam Recall", "Specificity", "F1-Score", "ROC-AUC", "False Negatives (Missed Scams)"]
    print(metrics_df[cols_display].to_string())

    # Save Multi-Model ROC Comparison Curve
    plot_roc_curves(roc_curves_data, save_path=str(PLOTS_DIR / "roc_comparison.png"))
    plt.close()

    # 6. Overfitting Analysis (Train vs Validation Gap)
    print("\n" + "=" * 70)
    print("[Overfitting Analysis] Train vs Validation Performance & Generalization Gap")
    print("=" * 70)

    overfitting_results = []
    for name, model in models.items():
        gap_info = evaluate_overfitting_gap(model, X_train, y_train, X_val, y_val, model_name=name)
        overfitting_results.append(gap_info)

    overfitting_df = pd.DataFrame(overfitting_results)
    print(overfitting_df[["model_name", "train_f1", "val_f1", "generalization_gap", "diagnosis"]].to_string(index=False))

    # Generate Learning Curve for Random Forest, XGBoost and Scratch Logistic Regression
    print("\n -> Generating Learning Curves...")
    lc_rf = compute_learning_curve(build_random_forest(n_estimators=60, max_depth=20), X_train, y_train, cv=4)
    plot_learning_curve(lc_rf, "Random Forest", save_path=str(PLOTS_DIR / "learning_curve_rf.png"))
    plt.close()

    lc_lr = compute_learning_curve(build_sklearn_logistic_regression(C=1.0), X_train, y_train, cv=4)
    plot_learning_curve(lc_lr, "Logistic Regression", save_path=str(PLOTS_DIR / "learning_curve_lr.png"))
    plt.close()

    # 7. Robustness Testing against Deliberate Obfuscation
    print("\n" + "=" * 70)
    print("[Robustness Testing] Performance on Deliberately Obfuscated Test Set")
    print("=" * 70)

    # Compare Word TF-IDF vs Hybrid Pipeline on Sklearn Naive Bayes
    nb_word_only = build_sklearn_naive_bayes(alpha=1.0)
    nb_word_only.fit(X_train_word, y_train)
    clean_test_scams = test_df[test_df['label'] == 1]['message_text'].tolist()
    # Match count with obf_df
    min_len = min(len(clean_test_scams), len(X_obf_raw))
    clean_sample_texts = clean_test_scams[:min_len]
    obf_sample_texts = X_obf_raw[:min_len]

    robustness_records = []
    
    # Evaluate Word-Only NB
    X_clean_w = word_vec.transform(clean_sample_texts)
    X_obf_w = word_vec.transform(obf_sample_texts)
    y_true_scam = np.ones(min_len, dtype=int)
    rec_clean_w = np.mean(nb_word_only.predict(X_clean_w) == 1)
    rec_obf_w = np.mean(nb_word_only.predict(X_obf_w) == 1)
    robustness_records.append({
        "model_name": "Word-TF-IDF Only (Naive Bayes)",
        "clean_scam_recall": float(rec_clean_w),
        "obfuscated_scam_recall": float(rec_obf_w),
        "recall_drop": float(rec_clean_w - rec_obf_w),
        "retention_rate_pct": float(rec_obf_w / rec_clean_w * 100.0)
    })

    # Evaluate Hybrid Models
    for eval_name in ["Scratch Naive Bayes", "Scratch Logistic Regression", "Random Forest", "XGBoost", "Neural MLP"]:
        m = models[eval_name]
        rob_data = evaluate_model_robustness(m, clean_sample_texts, obf_sample_texts, feature_pipeline, f"Hybrid ({eval_name})")
        robustness_records.append(rob_data)

    robustness_df = pd.DataFrame(robustness_records)
    print(robustness_df[["model_name", "clean_scam_recall", "obfuscated_scam_recall", "recall_drop", "retention_rate_pct"]].to_string(index=False))

    plot_robustness_comparison(robustness_records, save_path=str(PLOTS_DIR / "robustness_comparison.png"))
    plt.close()

    # 8. Plot Scratch Logistic Regression Loss Convergence
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lr_scratch.loss_history_, color='#d9534f', lw=2)
    ax.set_title("From-Scratch Logistic Regression: Adam Loss Convergence", fontsize=11, fontweight='bold')
    ax.set_xlabel("Iteration Number")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(PLOTS_DIR / "scratch_lr_loss_curve.png"), dpi=300)
    plt.close()

    # 9. Save Serialized Artifacts
    print("\n" + "=" * 70)
    print("[Stage 6] Serializing and Exporting Artifacts...")
    print("=" * 70)

    # Save Feature Pipeline
    joblib.dump(feature_pipeline, MODELS_DIR / "hybrid_vectorizer.joblib")
    print(f" -> Saved Feature Pipeline: {MODELS_DIR / 'hybrid_vectorizer.joblib'}")

    # Save Models
    for name, model in models.items():
        safe_name = name.lower().replace(' ', '_').replace('-', '_')
        model_path = MODELS_DIR / f"{safe_name}.joblib"
        joblib.dump(model, model_path)
        print(f" -> Saved {name} -> {model_path}")

    # Export Metrics Summary CSV and JSON
    metrics_export_path = MODELS_DIR.parent / "metrics_summary.json"
    full_summary = {
        "test_metrics": test_metrics,
        "scratch_vs_sklearn_validation": validation_verification,
        "overfitting_analysis": overfitting_results,
        "robustness_analysis": robustness_records
    }
    with open(metrics_export_path, "w") as f:
        json.dump(full_summary, f, indent=4)

    metrics_df.to_csv(MODELS_DIR.parent / "metrics_summary.csv")
    print(f" -> Saved Metrics Summary: {metrics_export_path}")

    print("\n" + "=" * 70)
    print("TRAINING & EVALUATION PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
