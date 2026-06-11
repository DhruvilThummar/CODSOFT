import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve, classification_report
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def main():
    # Setup directories
    current_dir = os.path.abspath(os.getcwd())
    dataset_path = os.path.join(current_dir, "Credit_Card_Fraud_Dataset", "creditcard.csv")
    plots_dir = os.path.join(current_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print("Loading dataset...")
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 1. Basic Data Integrity and Cleaning
    print("\n--- Data Cleaning & Integrity Check ---")
    null_counts = df.isnull().sum().sum()
    print(f"Total null values: {null_counts}")
    
    duplicate_count = df.duplicated().sum()
    print(f"Duplicate rows found: {duplicate_count}")
    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Dropped duplicates. New shape: {df.shape}")
        
    # Class distribution
    class_counts = df['Class'].value_counts()
    legit_count = class_counts.get(0, 0)
    fraud_count = class_counts.get(1, 0)
    fraud_percentage = (fraud_count / len(df)) * 100
    print(f"Class distribution: Legitimate (0) = {legit_count}, Fraudulent (1) = {fraud_count}")
    print(f"Fraud percentage: {fraud_percentage:.4f}%")
    
    # Save a distribution plot
    plt.figure(figsize=(6, 5))
    sns.countplot(x='Class', hue='Class', data=df, palette={0: '#1f77b4', 1: '#d62728'}, legend=False)
    plt.title('Transaction Class Distribution')
    plt.xlabel('Class (0: Legitimate, 1: Fraudulent)')
    plt.ylabel('Count')
    plt.yscale('log') # Log scale to see the minority class clearly
    plt.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'class_distribution.png'))
    plt.close()
    
    # 2. Train / Validation / Test Splitting
    # Split: 60% Train, 20% Val, 20% Test
    print("\n--- Splitting Data ---")
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    
    print(f"Train set size: {X_train.shape[0]} ({len(X_train)/len(df)*100:.1f}%)")
    print(f"Validation set size: {X_val.shape[0]} ({len(X_val)/len(df)*100:.1f}%)")
    print(f"Test set size: {X_test.shape[0]} ({len(X_test)/len(df)*100:.1f}%)")
    
    # 3. Scaling Features (Time and Amount)
    print("\n--- Scaling Time and Amount ---")
    # RobustScaler is less sensitive to outliers, which are common in transaction amounts
    scaler = RobustScaler()
    
    # Fit only on training data and transform train/val/test
    # Time and Amount are at columns 0 and 29 in the original dataset
    # We will scale them and update the datasets
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
    X_val_scaled[['Time', 'Amount']] = scaler.transform(X_val[['Time', 'Amount']])
    X_test_scaled[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])
    
    # 4. Resampling Training Data (SMOTE and Under-sampling)
    print("\n--- Creating Resampled Training Datasets ---")
    # SMOTE over-sampling
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    print(f"SMOTE Training Set: Legitimate = {np.sum(y_train_smote==0)}, Fraudulent = {np.sum(y_train_smote==1)}")
    
    # Random Under-sampling
    rus = RandomUnderSampler(random_state=42)
    X_train_rus, y_train_rus = rus.fit_resample(X_train_scaled, y_train)
    print(f"Under-sampled Training Set: Legitimate = {np.sum(y_train_rus==0)}, Fraudulent = {np.sum(y_train_rus==1)}")
    
    # 5. Model Training & Evaluation Setup
    # We will train Logistic Regression, Decision Tree, and Random Forest
    # on:
    # 1. Original (Imbalanced but using class_weight='balanced')
    # 2. Under-sampled (RUS)
    # 3. Over-sampled (SMOTE)
    # We'll evaluate on the validation set to find the best configuration,
    # then report performance of the selected models on the test set.
    
    models_config = {
        'Logistic Regression (Balanced)': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Logistic Regression (RUS)': LogisticRegression(max_iter=1000, random_state=42),
        'Logistic Regression (SMOTE)': LogisticRegression(max_iter=1000, random_state=42),
        
        'Decision Tree (Balanced)': DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42),
        'Decision Tree (RUS)': DecisionTreeClassifier(max_depth=6, random_state=42),
        'Decision Tree (SMOTE)': DecisionTreeClassifier(max_depth=6, random_state=42),
        
        'Random Forest (Balanced)': RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1),
        'Random Forest (RUS)': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'Random Forest (SMOTE)': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    }
    
    results = {}
    
    print("\n--- Training Models ---")
    for name, model in models_config.items():
        print(f"Training {name}...")
        if 'SMOTE' in name:
            model.fit(X_train_smote, y_train_smote)
        elif 'RUS' in name:
            model.fit(X_train_rus, y_train_rus)
        else:
            # Original imbalanced training set
            model.fit(X_train_scaled, y_train)
            
        # Predict on validation set
        y_val_pred = model.predict(X_val_scaled)
        # Check if model has predict_proba
        if hasattr(model, "predict_proba"):
            y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
        else:
            y_val_proba = model.decision_function(X_val_scaled)
            
        # Evaluate
        acc = accuracy_score(y_val, y_val_pred)
        prec = precision_score(y_val, y_val_pred, zero_division=0)
        rec = recall_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred)
        roc_auc = roc_auc_score(y_val, y_val_proba)
        pr_auc = average_precision_score(y_val, y_val_proba)
        
        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'PR-AUC (AP)': pr_auc,
            'model_obj': model
        }
    
    # 6. Save results to table
    df_results = pd.DataFrame(results).T.drop(columns=['model_obj'])
    print("\n--- Validation Set Results ---")
    print(df_results.to_string())
    
    # Save validation metrics to text file
    metrics_path = os.path.join(current_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("=== Validation Set Performance Comparison ===\n")
        f.write(df_results.to_string())
        f.write("\n\n")
        
    # Plot PR Curves for all models on Validation Set
    plt.figure(figsize=(10, 8))
    for name in results.keys():
        model = results[name]['model_obj']
        if hasattr(model, "predict_proba"):
            y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
        else:
            y_val_proba = model.decision_function(X_val_scaled)
        
        precision, recall, _ = precision_recall_curve(y_val, y_val_proba)
        ap = results[name]['PR-AUC (AP)']
        plt.plot(recall, precision, label=f"{name} (AP = {ap:.4f})")
        
    plt.title('Precision-Recall Curves (Validation Set)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'pr_curves_validation.png'))
    plt.close()
    
    # 7. Select and Evaluate on Test Set
    # We will pick the best Logistic Regression, Decision Tree, and Random Forest model based on PR-AUC / F1-Score
    # Usually:
    # - Logistic Regression (SMOTE or Balanced) performs well on recall.
    # - Random Forest (SMOTE or Balanced) performs best overall (high PR-AUC and F1).
    # Let's evaluate the three "Balanced" or "SMOTE" variants on the Test Set.
    
    selected_models = {
        'Logistic Regression (SMOTE)': results['Logistic Regression (SMOTE)']['model_obj'],
        'Decision Tree (SMOTE)': results['Decision Tree (SMOTE)']['model_obj'],
        'Random Forest (SMOTE)': results['Random Forest (SMOTE)']['model_obj']
    }
    
    test_results = {}
    print("\n--- Evaluating Selected Models on Test Set ---")
    
    with open(metrics_path, "a") as f:
        f.write("=== Test Set Performance of SMOTE Models ===\n")
    
    plt.figure(figsize=(10, 8))
    
    for name, model in selected_models.items():
        y_test_pred = model.predict(X_test_scaled)
        if hasattr(model, "predict_proba"):
            y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_test_proba = model.decision_function(X_test_scaled)
            
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
            'PR-AUC (AP)': pr_auc
        }
        
        # Save Confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Legit', 'Fraud'], 
                    yticklabels=['Legit', 'Fraud'])
        plt.title(f'Confusion Matrix - {name} (Test Set)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, f'confusion_matrix_{name.lower().replace(" ", "_").replace("(", "").replace(")", "")}.png'))
        plt.close()
        
        # Plot PR Curve for test
        precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
        plt.plot(recall, precision, label=f"{name} (AP = {pr_auc:.4f})")
        
        # Write to log and metrics file
        report = classification_report(y_test, y_test_pred)
        print(f"\n{name} Test Set Classification Report:")
        print(report)
        
        with open(metrics_path, "a") as file_out:
            file_out.write(f"\n{name} Test Set Classification Report:\n")
            file_out.write(report)
            file_out.write(f"ROC-AUC: {roc_auc:.4f}\n")
            file_out.write(f"PR-AUC (AP): {pr_auc:.4f}\n")
            file_out.write("-" * 40 + "\n")
            
    # Finalize test PR Curves plot
    plt.title('Precision-Recall Curves (Test Set)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'pr_curves_test.png'))
    plt.close()
    
    df_test_results = pd.DataFrame(test_results).T
    print("\n=== Final Test Set Results ===")
    print(df_test_results.to_string())
    
    with open(metrics_path, "a") as f:
        f.write("\n=== Summary Test Metrics ===\n")
        f.write(df_test_results.to_string())
        
    print(f"\nResults successfully written to {metrics_path}")
    print(f"Plots successfully saved to {plots_dir}")

if __name__ == "__main__":
    main()
