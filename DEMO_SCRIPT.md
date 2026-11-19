# 🛡️ Scam Shield: Project Demonstration & Viva Script
**Course:** AIML Experiential Learning (5th Semester)  
**Institution:** RV College of Engineering (RVCE), Bengaluru  
**Presenter:** Naga Sathya & Team  
**Repository:** [https://github.com/nagasathya-06/ScamShield](https://github.com/nagasathya-06/ScamShield)  

---

## 🎬 Demonstration Overview
This script is structured step-by-step so you can deliver a smooth, confident, and academically rigorous 8–10 minute demonstration to your evaluator.

```
Total Time: ~8 to 10 Minutes
├─ 00:00 - 01:30 | Introduction & Indian Threat Landscape
├─ 01:30 - 03:00 | System Architecture & Hybrid Feature Engineering
├─ 03:00 - 04:30 | From-Scratch Algorithm Verification (Core AIML Requirement)
├─ 04:30 - 06:30 | Live Interactive Detection, Explainability & Obfuscation Attack
├─ 06:30 - 08:00 | Benchmark Results, Overfitting Analysis & Robustness
└─ 08:00 - 10:00 | Conclusion & Evaluator Q&A Defense
```

---

## ⏱️ Section 1: Introduction & Problem Context (00:00 – 01:30)

### 🗣️ What to Say to the Evaluator:
> *"Respected evaluators, good morning. Today, we present **Scam Shield**, an explainable Machine Learning classification system specifically engineered to detect Indian financial scams and differentiate them from legitimate banking and service notifications.*
>
> *With the rapid rise of digital payments like UPI, PhonePe, and Google Pay in India, digital fraud has surged. Scammers do not send generic English spam; they send highly targeted Indian financial lures: fake KYC suspensions, unauthorized electricity disconnection threats, UPI cashback rewards asking for a PIN, and lottery scams in mixed English and Hinglish.*
>
> *Standard spam filters fail because scammers deliberately obfuscate keywords—using leetspeak like `'Bl0cked'`, `'K.Y.C'`, or `'cl1ck h3re'`. Furthermore, traditional black-box models do not explain why a message was flagged.*
>
> ***Scam Shield** addresses this through three core contributions:*
> 1. *Implementing foundational classification algorithms—**Multinomial Naive Bayes and Logistic Regression—completely from scratch using NumPy**, verified against Scikit-Learn.*
> 2. *Designing a **Hybrid Feature Pipeline** (Word TF-IDF + Character N-Grams + Handcrafted Domain Rules) that provides high resilience against adversarial obfuscation.*
> 3. *Providing **token-level explainability (Log-Odds & LIME)** and an interactive Streamlit application for end users and evaluators."*

---

## ⏱️ Section 2: Hybrid Feature Engineering & Dataset (01:30 – 03:00)

### 🖥️ Action on Screen:
*Open the Streamlit app to **Tab 6: Dataset Explorer**.*

### 🗣️ What to Say:
> *"For our data foundation, we combined the benchmark UCI SMS dataset (5,572 messages) with a curated corpus of authentic Indian financial scams (UPI cashback, electricity disconnection, SBI/HDFC KYC alerts) and genuine bank transaction alerts and OTPs, creating a comprehensive dataset of 5,716 samples.*
>
> *To convert raw text into numerical features, we built a **5,014-dimensional Hybrid Feature Pipeline**:*
> - ***Word-level TF-IDF (1-2 grams)**: Captures semantic financial terms like `'refund'`, `'blocked'`, `'cashback'`, and `'otp'`.*
> - ***Character N-gram TF-IDF (3-5 grams)**: Captures sub-word morphology. When a scammer writes `'Bl0cked'` or `'K.Y.C'`, character n-grams still match patterns with `'blocked'` and `'kyc'`.*
> - ***Domain Heuristic Rules**: Extracts 14 handcrafted flags, including URL shortener presence (`bit.ly`, `tinyurl`), Indian phone numbers, urgency indicators, and credential extraction requests (PIN/OTP).*
>
> *All feature values are scaled using `MinMaxScaler` into $[0, 1]$ to maintain non-negativity for Naive Bayes while stabilizing gradient descent."*

---

## ⏱️ Section 3: From-Scratch Algorithms Verification (03:00 – 04:30)

### 🖥️ Action on Screen:
*Switch to **Tab 3: From-Scratch Algorithms**.*

### 🗣️ What to Say:
> *"To fulfill the fundamental machine learning syllabus objectives, we implemented both **Multinomial Naive Bayes** and **Logistic Regression** entirely from scratch using pure NumPy.*
>
> *For **Naive Bayes**:*
> - *We implemented Laplace additive smoothing ($\alpha=1.0$) to prevent zero-probability errors on unseen tokens.*
> - *To prevent arithmetic underflow when multiplying thousands of small probabilities, we formulate calculations entirely in log-space and use the **Log-Sum-Exp numerical trick** for probability normalization.*
> - *As shown on screen, our scratch implementation achieves **100.00% exact prediction agreement** with `sklearn.naive_bayes.MultinomialNB` on the test dataset.*
>
> *For **Logistic Regression**:*
> - *We formulated the analytical gradient of the L2-regularized Binary Cross-Entropy loss function:*
>   $$\nabla_w J = \frac{1}{m} X^T (\hat{y} - y) + \frac{\lambda}{m} w$$
> - *Instead of simple batch gradient descent, we implemented the **Adam optimizer (Adaptive Moment Estimation)** from scratch in NumPy, tracking running first and second moment vectors.*
> - *Our scratch Logistic Regression achieves **99.65% agreement** with Scikit-learn, with the empirical training loss smoothly converging from $0.693$ down to $0.020$."*

---

## ⏱️ Section 4: Live Interactive Detection & Obfuscation Attack (04:30 – 06:30)

### 🖥️ Action on Screen:
*Switch to **Tab 1: Live Scam Detector**.*

### 🗣️ Demo Steps to Execute Live:

#### Step 4.1: Test an Indian KYC Scam
1. Select preset: `[SCAM] SBI YONO Account Suspended (KYC Phishing)`
2. Click **🚀 Analyze Message**
3. **Show Evaluator:**
   - Classification: **🚨 SCAM / FRAUDULENT MESSAGE**
   - Scam Probability: **`99.8%`**
   - High-risk Rule Triggers: *Shortened URL (`bit.ly`), Urgency keyword (`suspended`, `blocked`), Sensitive request (`KYC`)*
   - Token-level highlights: `'blocked'`, `'kyc'`, `'yono'` highlighted in red.

#### Step 4.2: Demonstrate the Adversarial Obfuscation Attack!
1. Keep the same message or select `[SCAM - OBFUSCATED] Leetspeak & Dotted Keyword Evader`.
2. Click **⚡ Simulate Obfuscation (Evader)**.
3. The message instantly mutates to:  
   `"Urgent: Y0ur S.B.I @ccount has been bl0cked... Cl1ck h3re to v3rify: b1t.ly/sbi-upd8"`
4. Click **🚀 Analyze Message**.
5. **Show Evaluator:**  
   > *"Notice that despite leetspeak substitution (`0` for `o`, `@` for `a`, dotted `S.B.I` and `K.Y.C`), our model **still catches it with over 98% confidence**! This proves the real-world value of our character n-gram and heuristic hybrid pipeline."*

#### Step 4.3: Test a Genuine Bank Alert (False Positive Check)
1. Select preset: `[SAFE] Genuine HDFC Bank UPI Debit Alert`
2. Click **🚀 Analyze Message**
3. **Show Evaluator:**
   - Classification: **✅ SAFE / LEGITIMATE MESSAGE**
   - Legitimacy Confidence: **`98.5%`**
   - Zero malicious rule triggers; transaction tokens recognized as safe.

---

## ⏱️ Section 5: Model Benchmark, Overfitting & Robustness (06:30 – 08:00)

### 🖥️ Action on Screen:
*Switch to **Tab 2: Model Benchmark** and **Tab 4: Overfitting & Curves**.*

### 🗣️ What to Say:
> *"We trained and benchmarked 8 models on identical stratified splits:*
> - *Baseline: Scratch Naive Bayes, Sklearn Naive Bayes, Scratch Logistic Regression, Sklearn Logistic Regression, kNN (k=5).*
> - *Advanced: Random Forest (150 trees, balanced weights), XGBoost Gradient Boosting, and a Multi-Layer Perceptron (128x64).*
>
> *All models achieve over **97.8% Accuracy** and **> 0.989 ROC-AUC**. Logistic Regression and XGBoost lead with **98.8% Accuracy and 93.5% Scam Recall**.*
>
> *In **Tab 4 (Overfitting Analysis)**, we quantified the generalization gap:  
> $\text{Gap} = F1_{\text{train}} - F1_{\text{val}}$.  
> Across all models, the gap remains small ($< 0.04$), confirming **Optimal Generalization** without high variance or overfitting. The learning curves confirm both training and cross-validation scores converge cleanly as data increases.*
>
> *In **Tab 5 (Robustness)**, we compared a baseline Word-only model vs our Hybrid model on 170 adversarial messages. The Word-only model suffered a **24.5% drop in recall**, whereas our Hybrid models retained **94% to 99%** of their detection power."*

---

## ⏱️ Section 6: Evaluator Viva Defense & Anticipated Questions (08:00 – 10:00)

### ❓ Question 1: Why did you implement Naive Bayes and Logistic Regression from scratch?
**Answer:**
> *"Implementing from scratch demonstrates mastery of core Machine Learning mathematics. It allows us to explicitly handle numerical stability issues such as Laplace smoothing ($\alpha=1$) to prevent zero frequencies, log-space arithmetic to prevent floating-point underflow, and implementing Adam gradient updates with moment vectors ($m_t$ and $v_t$) to optimize non-convex/convex losses efficiently."*

### ❓ Question 2: Why is Scam Recall more important than overall Accuracy?
**Answer:**
> *"In fraud detection, classes are imbalanced (more safe messages than scam messages). An algorithm could achieve 90% accuracy by simply predicting 'Safe' every time, but missing a scam (False Negative) leads to catastrophic financial loss for the user. A False Positive only causes a user to double-check their bank app. Therefore, we optimize for **High Recall on the Scam class** while preserving precision."*

### ❓ Question 3: How does your system detect obfuscated text like "Bl0cked" or "K.Y.C"?
**Answer:**
> *"Traditional word tokenizers treat `'Bl0cked'` as an unknown token ($OOV$) because it is not in the dictionary. Our system uses **Character N-Grams (range 3 to 5)**. The character sequence `'l0c'`, `'0ck'`, `'ck'` shares sub-word n-gram overlaps with `'loc'`, `'ock'`, `'ck'`. Combined with our domain rule extractor detecting the leetspeak ratio and URL shorteners, the system identifies the message as fraudulent even when keywords are disguised."*

### ❓ Question 4: How does LIME work for local explainability here?
**Answer:**
> *"LIME (Local Interpretable Model-agnostic Explanations) generates perturbations of the input message by randomly dropping words, passes these perturbed variants through the trained classifier, and fits an interpretable linear surrogate model weighted by exponential cosine distance to the original message. The coefficients of this local model reveal which words most strongly pushed the decision towards 'Scam' versus 'Safe'."*

### ❓ Question 5: What are the future enhancements you plan for Scam Shield?
**Answer:**
> *"As detailed in our synopsis, our future roadmap includes:
> 1. **Multilingual Regional Support**: Extending coverage to native Indian scripts (Kannada, Hindi, Tamil, Telugu).
> 2. **Pretrained Multilingual Transformers**: Fine-tuning IndicBERT or MuRIL for complex code-mixed Roman-script semantics.
> 3. **Phishing URL Deep Inspection**: Real-time HTTP header and domain age/WHOIS validation of shortened links.
> 4. **Mobile & WhatsApp End-Point Integration**: Deploying Scam Shield as a lightweight Android SMS filter or WhatsApp verification bot."*
