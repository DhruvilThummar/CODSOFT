<div align="center">

# 🤖 Machine Learning

### A portfolio of 5 end-to-end Machine Learning projects — built from scratch with real data, real models, and real results.

<br/>

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

<br/>

> *Every project includes a Jupyter notebook walkthrough, a standalone training script, evaluation plots, and saved model artifacts.*

</div>

---

## 📋 Table of Contents

- [Task 1 — Movie Genre Classification](#-task-1--movie-genre-classification)
- [Task 2 — Credit Card Fraud Detection](#-task-2--credit-card-fraud-detection)
- [Task 3 — Customer Churn Prediction](#-task-3--customer-churn-prediction)
- [Task 4 — SMS Spam Detection](#-task-4--sms-spam-detection)
- [Task 5 — Handwritten Text Generation](#-task-5--handwritten-text-generation)
- [Tech Stack](#-tech-stack)
- [Repository Structure](#-repository-structure)

---

## 🎬 Task 1 — Movie Genre Classification

> **Classify movies into genres based purely on their plot descriptions using NLP + ML.**

**Problem:** Given a movie plot summary (raw text), predict its genre from 27 possible categories — a challenging multi-class NLP problem.

**Approach:**
- TF-IDF vectorization of movie plot descriptions
- Multi-class classification with Naive Bayes, Logistic Regression, and Linear SVC
- Handled severe class imbalance (documentary & drama dominate)

### 📊 Demo — Model Comparison

![Model Comparison](task%201/plots/model_comparison.png)

| Model | Test Accuracy | Best Genre F1 |
|---|---|---|
| Multinomial Naive Bayes | 41.47% | Documentary (0.65) |
| Logistic Regression | 42.42% | Documentary (0.65) |
| **Linear SVC ✅** | **48.58%** | **Documentary (0.65)** |

> 🏆 **Best Model:** Linear Support Vector Classifier — 48.6% accuracy across 27 genres on text-only features, with strong performance on high-support classes like drama and documentary.

**📁 Files:** [`task 1/movie_genre_classification.ipynb`](task%201/movie_genre_classification.ipynb) · [`task 1/train_and_evaluate.py`](task%201/train_and_evaluate.py)

---

## 💳 Task 2 — Credit Card Fraud Detection

> **Detect fraudulent transactions in a massively imbalanced dataset (0.17% fraud rate).**

**Problem:** Real-world fraud detection where fraudulent transactions are extremely rare — making recall on the minority class critical and standard accuracy metrics deceptive.

**Approach:**
- Explored 3 class-imbalance strategies: **Class Weights**, **Random Under-Sampling (RUS)**, and **SMOTE**
- Compared Logistic Regression, Decision Tree, and Random Forest across all strategies
- Prioritized **PR-AUC** and **Recall** over raw accuracy (misleading with 99.8% majority class)

### 📊 Demo — Precision-Recall Curves (Test Set)

![PR Curves](task%202/plots/pr_curves_test.png)

| Model | Test Accuracy | Recall (Fraud) | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (SMOTE) | 97.0% | 87.4% | 0.688 | 0.958 |
| Decision Tree (SMOTE) | 95.5% | 85.3% | 0.508 | 0.933 |
| **Random Forest (SMOTE) ✅** | **99.9%** | **78.9%** | **0.772** | **0.975** |

> 🏆 **Best Model:** Random Forest with SMOTE — highest PR-AUC (0.77) and near-perfect overall accuracy while catching ~79% of all fraud cases.

**📁 Files:** [`task 2/credit_card_fraud_detection.ipynb`](task%202/credit_card_fraud_detection.ipynb) · [`task 2/train_and_evaluate.py`](task%202/train_and_evaluate.py)

> **📥 Dataset Note:** `creditcard.csv` (143 MB) is **not included** in this repo due to GitHub's 100 MB file limit.
> Download it automatically by running:
> ```bash
> cd "task 2"
> python download_data.py
> ```
> Or download manually from [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` inside `task 2/Credit_Card_Fraud_Dataset/`.

---

## 📉 Task 3 — Customer Churn Prediction

> **Predict which bank customers are likely to leave using demographic and account data.**

**Problem:** Subscription-based businesses lose revenue silently through churn. Predict churn probability from customer demographics and behavior before it happens.

**Approach:**
- Feature engineering on 10,000 customer records (geography, age, balance, activity)
- Compared Logistic Regression, Random Forest, and XGBoost
- Class balancing via SMOTE and class weights
- Feature importance analysis to identify key churn drivers

### 📊 Demo — ROC Curves (Test Set)

![ROC Curves](task%203/plots/roc_curves_test.png)

| Model | Test Accuracy | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Logistic Regression | 71.2% | 0.499 | 0.778 | 0.469 |
| Random Forest | 83.7% | 0.608 | 0.861 | 0.684 |
| **XGBoost ✅** | **80.0%** | **0.597** | **0.866** | **0.711** |

> 🏆 **Best Model:** XGBoost achieves the highest ROC-AUC (0.866) and PR-AUC (0.711), making it best at ranking customers by churn risk — critical for business use.

**📁 Files:** [`task 3/customer_churn_prediction.ipynb`](task%203/customer_churn_prediction.ipynb) · [`task 3/train_and_evaluate.py`](task%203/train_and_evaluate.py)

---

## 📱 Task 4 — SMS Spam Detection

> **Classify SMS messages as spam or legitimate (ham) with near-perfect accuracy.**

**Problem:** SMS spam filtering is a classic NLP classification task — but doing it with high precision AND recall simultaneously (avoiding both false spam blocks and missed spam) is the real challenge.

**Approach:**
- TF-IDF text vectorization with n-gram features
- Compared Naive Bayes, Logistic Regression, and Linear SVM
- Visualized top spam-indicative words
- Saved best model + vectorizer as `.joblib` for deployment

### 📊 Demo — ROC Curves (Test Set)

![ROC Curves](task%204/plots/roc_curves_test.png)

| Model | Test Accuracy | F1-Score (Spam) | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Naive Bayes | 96.4% | 0.836 | 0.986 | 0.964 |
| Logistic Regression | 97.6% | 0.906 | 0.990 | 0.960 |
| **SVM Linear ✅** | **98.7%** | **0.944** | **0.993** | **0.975** |

> 🏆 **Best Model:** Linear SVM — 98.7% accuracy, 99.2% precision on spam, and 0.993 ROC-AUC. Near-perfect spam detection with minimal false positives.

**📁 Files:** [`task 4/spam_sms_detection.ipynb`](task%204/spam_sms_detection.ipynb) · [`task 4/train_and_evaluate.py`](task%204/train_and_evaluate.py)

---

## ✍️ Task 5 — Handwritten Text Generation

> **Train a character-level LSTM on Shakespeare, then render the output as a handwritten image.**

**Problem:** Generative sequence modeling — teach a neural network to write plausible English text character-by-character, then synthesize the output visually as realistic handwriting.

**Approach:**
- Character-level LSTM trained on 100,000 characters of Shakespearean play scripts
- 2-layer LSTM with dropout (embedding dim=128, hidden dim=256)
- Temperature-based sampling to control creativity vs. coherence
- Rendered output using Pillow + *Caveat* TTF font on a cream notebook canvas

### 📊 Demo — Generated Handwriting Output

> *Seed: `ROMEO:\nShall I speak ` — Temperature: 0.6 — 500 characters generated*

![Generated Handwriting](task%205/plots/generated_handwriting.png)

### 📉 Training Loss Curve

![Loss History](task%205/plots/loss_history.png)

| Metric | Value |
|---|---|
| Training Corpus | 100,000 characters |
| Vocabulary Size | 61 unique characters |
| Initial Loss | ~2.33 |
| **Final Loss (Epoch 12)** | **1.03 (56% improvement)** |

> 🏆 **Highlight:** The model learns Shakespearean dialogue structure — speaker names, line breaks, capitalization, and archaic vocabulary — purely from character sequences with no tokenizer.

**📁 Files:** [`task 5/handwritten_text_generation.ipynb`](task%205/handwritten_text_generation.ipynb) · [`task 5/train_and_evaluate.py`](task%205/train_and_evaluate.py)

---

## 🧰 Tech Stack

| Category | Libraries |
|---|---|
| **Deep Learning** | PyTorch, torch.nn, LSTM |
| **Classical ML** | scikit-learn (SVM, RF, LR, NB), XGBoost |
| **NLP** | TF-IDF, n-grams, character-level sequences |
| **Imbalance Handling** | SMOTE (imbalanced-learn), class weights, RUS |
| **Image Synthesis** | Pillow (PIL), TrueType Fonts |
| **Visualization** | Matplotlib, Seaborn |
| **Evaluation** | ROC-AUC, PR-AUC, F1, confusion matrices |
| **Workflow** | Jupyter Notebooks, standalone `.py` scripts |
| **Persistence** | `joblib` (sklearn models), `.pth` (PyTorch) |

---

## 📁 Repository Structure

```
CODSOFT/
│
├── task 1/                              # Movie Genre Classification
│   ├── movie_genre_classification.ipynb
│   ├── train_and_evaluate.py
│   ├── download_data.py
│   ├── requirements.txt
│   └── plots/
│       ├── model_comparison.png
│       ├── confusion_matrix.png
│       ├── genre_distribution.png
│       └── evaluation_report.txt
│
├── task 2/                              # Credit Card Fraud Detection
│   ├── credit_card_fraud_detection.ipynb
│   ├── train_and_evaluate.py
│   ├── download_data.py                 ← Run this to fetch creditcard.csv
│   ├── metrics.txt
│   ├── requirements.txt
│   └── plots/
│       ├── pr_curves_test.png
│       ├── pr_curves_validation.png
│       ├── class_distribution.png
│       └── confusion_matrix_*.png
│   # ⚠️  creditcard.csv (143 MB) is NOT committed — run download_data.py
│
├── task 3/                              # Customer Churn Prediction
│   ├── customer_churn_prediction.ipynb
│   ├── train_and_evaluate.py
│   ├── best_churn_model.joblib          ← Saved model
│   ├── preprocessor.joblib             ← Saved pipeline
│   ├── metrics.txt
│   ├── requirements.txt
│   └── plots/
│       ├── roc_curves_test.png
│       ├── feature_importances.png
│       └── confusion_matrix_*.png
│
├── task 4/                              # SMS Spam Detection
│   ├── spam_sms_detection.ipynb
│   ├── train_and_evaluate.py
│   ├── best_spam_model.joblib          ← Saved SVM model
│   ├── tfidf_vectorizer.joblib         ← Saved vectorizer
│   ├── metrics.txt
│   ├── requirements.txt
│   └── plots/
│       ├── roc_curves_test.png
│       ├── top_spam_words.png
│       └── confusion_matrix_*.png
│
└── task 5/                              # Handwritten Text Generation
    ├── handwritten_text_generation.ipynb
    ├── train_and_evaluate.py
    ├── best_rnn_model.pth              ← Trained LSTM checkpoint
    ├── Caveat-Regular.ttf              ← Handwriting font
    ├── handwriting_transcripts.txt     ← Shakespeare corpus
    ├── metrics.txt
    ├── requirements.txt
    └── plots/
        ├── generated_handwriting.png   ← Final visual output
        └── loss_history.png
```

---

## 🚀 Running Any Task

Each task is self-contained. To run:

```bash
# 1. Navigate to the task
cd "task X"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset (if needed)
python download_data.py

# 4a. Run the training script
python train_and_evaluate.py

# 4b. OR open the interactive notebook
jupyter notebook
```

---

<div align="center">


*5 projects · 5 real-world ML problems · End-to-end implementations*

</div>