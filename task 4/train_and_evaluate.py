import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve, classification_report
)

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove URL links
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove special characters and punctuation (keep letters and spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    # Setup directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "SMS_Spam_Dataset", "spam.csv")
    plots_dir = os.path.join(current_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print("Loading dataset...")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please run download_data.py first.")
        
    # Load raw CSV with latin-1 encoding
    df = pd.read_csv(dataset_path, encoding='latin-1')
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 1. Clean extra columns and rename
    # The dataset has extra columns 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'
    cols_to_keep = ['v1', 'v2']
    df = df[cols_to_keep]
    df.columns = ['label', 'text']
    print(f"Cleaned extra columns. Shape: {df.shape}")
    
    # Check nulls
    null_counts = df.isnull().sum().sum()
    print(f"Total null values: {null_counts}")
    
    # Check duplicates
    duplicate_count = df.duplicated().sum()
    print(f"Duplicate rows found: {duplicate_count}")
    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Dropped duplicates. New shape: {df.shape}")
        
    # Map target variable (ham = 0, spam = 1)
    df['target'] = df['label'].map({'ham': 0, 'spam': 1})
    print("Class mapping: 'ham' -> 0, 'spam' -> 1")
    
    # Target distribution
    target_counts = df['target'].value_counts()
    ham_count = target_counts.get(0, 0)
    spam_count = target_counts.get(1, 0)
    spam_percentage = (spam_count / len(df)) * 100
    print(f"Class distribution: Ham (Legit) = {ham_count}, Spam = {spam_count}")
    print(f"Spam percentage: {spam_percentage:.2f}%")
    
    # Save a distribution plot
    plt.figure(figsize=(6, 5))
    sns.countplot(x='label', hue='label', data=df, palette={'ham': '#1f77b4', 'spam': '#d62728'}, legend=False)
    plt.title('SMS Classification Target Distribution')
    plt.xlabel('Category')
    plt.ylabel('Count')
    for i, count in enumerate([ham_count, spam_count]):
        plt.text(i, count + 50, f"{count}\n({count/len(df)*100:.1f}%)", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'class_distribution.png'), dpi=300)
    plt.close()
    
    # 2. Text Preprocessing
    print("\n--- Preprocessing Text Content ---")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Drop rows that ended up empty after cleaning (just in case)
    df = df[df['cleaned_text'] != ''].reset_index(drop=True)
    print(f"Shape after removing empty cleaned texts: {df.shape}")
    
    # 3. Train / Validation / Test Splitting (60% Train, 20% Val, 20% Test)
    print("\n--- Splitting Data ---")
    X = df['cleaned_text']
    y = df['target']
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    
    print(f"Train set size: {X_train.shape[0]} ({len(X_train)/len(df)*100:.1f}%)")
    print(f"Validation set size: {X_val.shape[0]} ({len(X_val)/len(df)*100:.1f}%)")
    print(f"Test set size: {X_test.shape[0]} ({len(X_test)/len(df)*100:.1f}%)")
    
    # 4. TF-IDF Feature Extraction (Fit only on training data to prevent leakage)
    print("\n--- Extracting TF-IDF Features ---")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"TF-IDF shape (Train): {X_train_tfidf.shape}")
    
    # 5. Model Configurations
    # We will try:
    # - Naive Bayes (MultinomialNB) - standard baseline for text classification
    # - Logistic Regression
    # - SVM (SVC with linear kernel and probability calibration)
    models_config = {
        'Naive Bayes (MultinomialNB)': MultinomialNB(alpha=1.0),
        'Logistic Regression (Base)': LogisticRegression(random_state=42, max_iter=1000),
        'Logistic Regression (Balanced)': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
        'Support Vector Machine (Linear)': SVC(kernel='linear', probability=True, random_state=42)
    }
    
    # Naive baseline (Predict majority class 0 - ham)
    y_val_naive = np.zeros_like(y_val)
    results = {
        'Naive Baseline': {
            'Accuracy': accuracy_score(y_val, y_val_naive),
            'Precision': precision_score(y_val, y_val_naive, zero_division=0),
            'Recall': recall_score(y_val, y_val_naive),
            'F1-Score': f1_score(y_val, y_val_naive),
            'ROC-AUC': 0.5,
            'PR-AUC': average_precision_score(y_val, np.zeros_like(y_val) + spam_percentage/100),
            'model_obj': None
        }
    }
    
    print("\n--- Training and Evaluating Models on Validation Set ---")
    for name, model in models_config.items():
        print(f"Training {name}...")
        model.fit(X_train_tfidf, y_train)
        
        y_val_pred = model.predict(X_val_tfidf)
        y_val_proba = model.predict_proba(X_val_tfidf)[:, 1]
        
        results[name] = {
            'Accuracy': accuracy_score(y_val, y_val_pred),
            'Precision': precision_score(y_val, y_val_pred, zero_division=0),
            'Recall': recall_score(y_val, y_val_pred),
            'F1-Score': f1_score(y_val, y_val_pred),
            'ROC-AUC': roc_auc_score(y_val, y_val_proba),
            'PR-AUC': average_precision_score(y_val, y_val_proba),
            'model_obj': model
        }
        
    df_results = pd.DataFrame(results).T.drop(columns=['model_obj'])
    print("\n=== Validation Set Performance ===")
    print(df_results.round(4).to_string())
    
    metrics_path = os.path.join(current_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("=== Validation Set Performance Comparison ===\n")
        f.write(df_results.round(4).to_string())
        f.write("\n\n")
        
    # Plot PR Curves for all models on Validation Set
    plt.figure(figsize=(10, 8))
    for name in results.keys():
        if name == 'Naive Baseline':
            continue
        model = results[name]['model_obj']
        y_val_proba = model.predict_proba(X_val_tfidf)[:, 1]
        precision, recall, _ = precision_recall_curve(y_val, y_val_proba)
        pr_auc = results[name]['PR-AUC']
        plt.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.4f})")
        
    plt.title('Precision-Recall Curves (Validation Set)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'pr_curves_validation.png'), dpi=300)
    plt.close()
    
    # 6. Select the Best Models and Evaluate on Test Set
    # We will pick the three standard algorithm classes for final evaluation:
    # - Naive Bayes (MultinomialNB)
    # - Logistic Regression (Balanced)
    # - Support Vector Machine (Linear)
    selected_configs = {
        'Naive Bayes': results['Naive Bayes (MultinomialNB)']['model_obj'],
        'Logistic Regression': results['Logistic Regression (Balanced)']['model_obj'],
        'SVM (Linear)': results['Support Vector Machine (Linear)']['model_obj']
    }
    
    test_results = {}
    print("\n--- Evaluating Selected Models on Test Set ---")
    
    with open(metrics_path, "a") as f:
        f.write("=== Test Set Performance of Selected Models ===\n")
        
    # Plot ROC and PR Curves for Test Set
    fig_roc, ax_roc = plt.subplots(figsize=(8, 7))
    fig_pr, ax_pr = plt.subplots(figsize=(8, 7))
    
    best_model_name = None
    best_f1 = 0
    best_model_obj = None
    
    for name, model in selected_configs.items():
        y_test_pred = model.predict(X_test_tfidf)
        y_test_proba = model.predict_proba(X_test_tfidf)[:, 1]
        
        acc = accuracy_score(y_test, y_test_pred)
        prec = precision_score(y_test, y_test_pred, zero_division=0)
        rec = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        roc_auc = roc_auc_score(y_test, y_test_proba)
        pr_auc = average_precision_score(y_test, y_test_proba)
        
        test_results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'PR-AUC': pr_auc
        }
        
        # Track the best model based on F1-score on test set
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = model
            
        # Write report to console and file
        report = classification_report(y_test, y_test_pred)
        print(f"\n{name} Test Set Classification Report:")
        print(report)
        
        with open(metrics_path, "a") as file_out:
            file_out.write(f"\n{name} Test Set Classification Report:\n")
            file_out.write(report)
            file_out.write(f"ROC-AUC: {roc_auc:.4f}\n")
            file_out.write(f"PR-AUC: {pr_auc:.4f}\n")
            file_out.write("-" * 40 + "\n")
            
        # Save Confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Ham', 'Spam'], 
                    yticklabels=['Ham', 'Spam'])
        plt.title(f'Confusion Matrix - {name} (Test Set)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'confusion_matrix_{name.lower().replace(" ", "_").replace("(", "").replace(")", "")}.png'), dpi=300)
        plt.close()
        
        # Add to ROC plot
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        ax_roc.plot(fpr, tpr, label=f"{name} (ROC-AUC = {roc_auc:.4f})")
        
        # Add to PR plot
        precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
        ax_pr.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.4f})")
        
    # Finalize ROC Plot
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax_roc.set_title('ROC Curves (Test Set)')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend(loc='lower right')
    ax_roc.grid(True, linestyle='--', alpha=0.6)
    fig_roc.tight_layout()
    fig_roc.savefig(os.path.join(plots_dir, 'roc_curves_test.png'), dpi=300)
    plt.close(fig_roc)
    
    # Finalize PR Plot
    ax_pr.plot([0, 1], [spam_percentage/100, spam_percentage/100], 'k--', alpha=0.5, label='No Skill')
    ax_pr.set_title('Precision-Recall Curves (Test Set)')
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.legend(loc='lower left')
    ax_pr.grid(True, linestyle='--', alpha=0.6)
    fig_pr.tight_layout()
    fig_pr.savefig(os.path.join(plots_dir, 'pr_curves_test.png'), dpi=300)
    plt.close(fig_pr)
    
    # Print and save final comparison table
    df_test_results = pd.DataFrame(test_results).T
    print("\n=== Final Test Set Results ===")
    print(df_test_results.round(4).to_string())
    
    with open(metrics_path, "a") as f:
        f.write("\n=== Summary Test Metrics ===\n")
        f.write(df_test_results.round(4).to_string())
        
    print(f"\nResults successfully written to {metrics_path}")
    
    # 7. Coefficient/Word Importance Analysis
    print("\n--- Key Words Driving Spam Decisions (Word Coefficients) ---")
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    # Logistic Regression coefficients
    lr_model = selected_configs['Logistic Regression']
    lr_coefs = lr_model.coef_[0]
    
    # SVM coefficients
    svm_model = selected_configs['SVM (Linear)']
    svm_coefs = svm_model.coef_.toarray()[0]
    
    # Sort indexes
    top_lr_spam_idx = np.argsort(lr_coefs)[-10:]
    top_svm_spam_idx = np.argsort(svm_coefs)[-10:]
    
    # Plot side-by-side bar plots of word importances
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # LR top words
    sns.barplot(
        ax=axes[0], 
        x=lr_coefs[top_lr_spam_idx][::-1], 
        y=feature_names[top_lr_spam_idx][::-1], 
        palette="viridis",
        hue=feature_names[top_lr_spam_idx][::-1],
        legend=False
    )
    axes[0].set_title('Logistic Regression Top Spam Words')
    axes[0].set_xlabel('Coefficient Value')
    
    # SVM top words
    sns.barplot(
        ax=axes[1], 
        x=svm_coefs[top_svm_spam_idx][::-1], 
        y=feature_names[top_svm_spam_idx][::-1], 
        palette="magma",
        hue=feature_names[top_svm_spam_idx][::-1],
        legend=False
    )
    axes[1].set_title('SVM Top Spam Words')
    axes[1].set_xlabel('Coefficient Value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'top_spam_words.png'), dpi=300)
    plt.close()
    
    # 8. Save preprocessor and best model objects
    best_model_path = os.path.join(current_dir, "best_spam_model.joblib")
    vectorizer_path = os.path.join(current_dir, "tfidf_vectorizer.joblib")
    
    joblib.dump(best_model_obj, best_model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Saved vectorizer to {vectorizer_path}")
    print(f"Saved best model ({best_model_name}) to {best_model_path}")

if __name__ == "__main__":
    main()
