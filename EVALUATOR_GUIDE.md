# 🛡️ Scam Shield: Evaluator & Technical Assessment Guide
**Course:** 5th Semester AIML Experiential Learning (AIML_EL)  
**Institution:** Department of Artificial Intelligence & Machine Learning, RV College of Engineering, Bengaluru  
**Project:** Scam Shield – Explainable ML Text Classification for Indian Financial Fraud Detection  
**Repository:** [https://github.com/nagasathya-06/ScamShield](https://github.com/nagasathya-06/ScamShield)  

---

## 📋 Executive Assessment Summary
This document serves as an academic and technical dossier for evaluators assessing **Scam Shield**. It maps every project component directly to the approved project synopsis and syllabus outcomes.

| Syllabus Component | Requirement in Synopsis | Project Implementation | Verification Artifact |
| :--- | :--- | :--- | :--- |
| **Data Collection & Preparation** | Public SMS dataset + Indian UPI/KYC scams + genuine bank alerts | 5,716 samples (English & Hinglish) + 170 dedicated obfuscated test samples | `data/processed/*.csv`, `scripts/prepare_data.py` |
| **Feature Extraction** | Word TF-IDF + Character N-grams + Domain Rules | 5,014-dimensional Hybrid Pipeline | `src/feature_engineering.py`, `src/preprocessor.py` |
| **From-Scratch ML** | Implement Naive Bayes & Logistic Regression in NumPy | Laplace-smoothed MultinomialNB & Adam-optimized Logistic Regression | `src/models/scratch_naive_bayes.py`, `scratch_logistic_regression.py` |
| **Algorithm Validation** | Verify scratch models against Scikit-Learn equivalents | NB Agreement: **100.00%**, LR Agreement: **99.65%** | `tests/test_scratch_models.py`, `scripts/train_and_evaluate.py` |
| **Model Benchmarking** | Baseline vs Advanced (RF, XGBoost, kNN, Neural MLP) | 8 models evaluated on Accuracy, Precision, Scam Recall, F1, ROC-AUC | `artifacts/metrics_summary.json`, `Tab 2 of App` |
| **Overfitting Analysis** | Train vs Val gap analysis & Cross-Validation learning curves | Generalization gap quantified; Stratified 5-Fold learning curves | `src/evaluation/overfitting_analysis.py`, `artifacts/plots/*.png` |
| **Robustness Testing** | Evaluate performance on deliberate obfuscated text | Adversarial test suite; Hybrid pipeline retains 94%–99% recall vs 71% in Word TF-IDF | `src/evaluation/robustness_evaluator.py`, `artifacts/plots/robustness_comparison.png` |
| **Explainability** | Token-level log-odds & LIME local surrogate explanations | Word log-ratio explainer + LIME integration + Domain rule heuristics | `src/explainability/*.py`, `Tab 1 of App` |
| **Web Interface** | User-friendly Streamlit web application | 7-tab interactive UI with live detector, model comparison, and adversarial simulator | `app/app.py`, `run_app.bat` |

---

## 📐 Mathematical Formulations & Implementations

### 1. From-Scratch Multinomial Naive Bayes (`src/models/scratch_naive_bayes.py`)
Multinomial Naive Bayes models text as a sequence of token counts/weights under class-conditional independence:

#### Class Log-Priors:
$$\log P(y=c) = \log\left(\frac{N_c}{N}\right)$$
where $N_c$ is the number of training samples in class $c \in \{0, 1\}$ and $N$ is total training samples.

#### Feature Likelihoods with Additive (Laplace) Smoothing:
$$\theta_{ci} = P(x_i | y=c) = \frac{\sum_{j \in C} X_{ji} + \alpha}{\sum_{k=1}^D \left(\sum_{j \in C} X_{jk}\right) + \alpha \cdot D}$$
where $D = 5,014$ is the feature dimension and $\alpha = 1.0$ is the smoothing parameter.

#### Log-Space Joint Likelihood & Log-Sum-Exp Normalization:
$$\log P(c | X) = \log P(y=c) + \sum_{i=1}^D X_i \log \theta_{ci}$$
To avoid floating-point underflow when normalizing to probabilities:
$$m = \max_{k} \log P(k | X)$$
$$P(y=c | X) = \frac{\exp(\log P(c | X) - m)}{\sum_{k} \exp(\log P(k | X) - m)}$$

---

### 2. From-Scratch Logistic Regression with Adam (`src/models/scratch_logistic_regression.py`)
Binary Logistic Regression maps feature vectors to class probabilities using the sigmoid activation.

#### Forward Hypothesis:
$$z = X w + b$$
$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-\text{clip}(z, -30, 30)}}$$

#### Regularized Binary Cross-Entropy Loss:
$$J(w, b) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(\hat{y}^{(i)} + \epsilon) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)} + \epsilon) \right] + \frac{\lambda}{2m} \|w\|_2^2$$

#### Analytical Gradients:
$$\frac{\partial J}{\partial w} = \frac{1}{m} X^T (\hat{y} - y) + \frac{\lambda}{m} w, \qquad \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})$$

#### Adam Optimization Algorithm (NumPy):
At iteration $t$:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla_w$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla_w \odot \nabla_w)$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$w_t = w_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
*(where $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta = 0.08$)*.

---

## 📊 Experimental Results & Empirical Verification

### 1. Test Set Performance Comparison (858 Samples)
Evaluated on unseen test partition: 719 Safe (0), 139 Scam (1).

| Model | Accuracy | Precision | Scam Recall | Specificity | F1-Score | ROC-AUC | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scratch Naive Bayes** | 97.79% | 94.78% | **91.37%** | 99.03% | 0.9304 | 0.9898 | 0.01 ms |
| **Sklearn Naive Bayes** | 97.79% | 94.78% | **91.37%** | 99.03% | 0.9304 | 0.9898 | 0.01 ms |
| **Scratch Logistic Regression** | 98.72% | 98.48% | **93.53%** | 99.72% | 0.9594 | 0.9929 | 0.02 ms |
| **Sklearn Logistic Regression** | 98.83% | 99.24% | **93.53%** | 99.86% | 0.9630 | 0.9971 | 0.02 ms |
| **Random Forest** | 98.02% | 99.19% | 88.49% | 99.86% | 0.9354 | 0.9947 | 0.12 ms |
| **XGBoost Gradient Boosting** | 98.48% | 98.46% | **92.09%** | 99.72% | 0.9517 | 0.9967 | 0.08 ms |
| **k-Nearest Neighbours (k=5)**| 97.79% | 97.62% | 88.49% | 99.58% | 0.9283 | 0.9762 | 0.95 ms |
| **Neural MLP (128x64)** | 98.48% | 97.01% | **93.53%** | 99.44% | 0.9524 | 0.9888 | 0.05 ms |

---

### 2. Verification of Mathematical Equivalence
- **Naive Bayes**: $\text{Agreement} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_{\text{scratch}} == \hat{y}_{\text{sklearn}}) = \mathbf{100.00\%}$.
- **Logistic Regression**: $\text{Agreement} = \mathbf{99.65\%}$ (only 3 marginal boundary decisions differ out of 858 samples due to L-BFGS vs Adam convergence tolerances).
- **Automated Unit Tests**: All unit tests in `tests/test_scratch_models.py` pass cleanly.

---

### 3. Adversarial Obfuscation Robustness Analysis
Evaluated against 170 deliberately obfuscated test messages containing leetspeak substitutions (`0` for `o`, `@` for `a`), dotted tokens (`K.Y.C`, `S.B.I`), and spaced keywords (`u r g e n t`):

| Model Architecture | Clean Scam Recall | Obfuscated Scam Recall | Recall Drop | Retention Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Word TF-IDF Only (Naive Bayes)** | 84.89% | 60.43% | **-24.46%** | **71.2% (Vulnerable)** |
| **Hybrid (Scratch Naive Bayes)** | 91.37% | 86.33% | -5.04% | **94.5% (Resilient)** |
| **Hybrid (Scratch Logistic Regression)**| 93.53% | 89.93% | -3.60% | **96.2% (Resilient)** |
| **Hybrid (XGBoost)** | 92.09% | 99.28% | +7.19% | **107.8% (Highly Resilient)**|
| **Hybrid (Neural MLP)** | 93.53% | 88.49% | -5.04% | **94.6% (Resilient)** |

---

## 🔍 How to Run the Demonstration

### 1. Launching the Web Interface
Double click `run_app.bat` or run:
```bash
streamlit run app/app.py
```

### 2. Running Automated Unit Tests
```bash
python -m unittest tests/test_scratch_models.py
python -m unittest tests/test_pipeline.py
```

### 3. Re-training and Generating Artifacts from Scratch
```bash
python scripts/prepare_data.py
python scripts/train_and_evaluate.py
```
