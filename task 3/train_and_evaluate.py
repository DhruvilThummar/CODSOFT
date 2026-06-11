import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve, classification_report
)
from imblearn.over_sampling import SMOTE

def main():
    # Setup directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "Churn_Modelling_Dataset", "Churn_Modelling.csv")
    plots_dir = os.path.join(current_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print("Loading dataset...")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please run download_data.py first.")
        
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 1. Basic Data Integrity and Cleaning
    print("\n--- Data Cleaning & Integrity Check ---")
    # Missing values
    null_counts = df.isnull().sum().sum()
    print(f"Total null values: {null_counts}")
    
    # Duplicates
    duplicate_count = df.duplicated().sum()
    print(f"Duplicate rows found: {duplicate_count}")
    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Dropped duplicates. New shape: {df.shape}")
        
    # Drop identifier columns that are irrelevant for modeling
    cols_to_drop = ['RowNumber', 'CustomerId', 'Surname']
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped identifier columns: {cols_to_drop}. Shape: {df.shape}")
    
    # Class distribution (Exited is the target: 1 = Churn, 0 = Retained)
    class_counts = df['Exited'].value_counts()
    retained_count = class_counts.get(0, 0)
    churned_count = class_counts.get(1, 0)
    churn_percentage = (churned_count / len(df)) * 100
    print(f"Class distribution: Retained (0) = {retained_count}, Churned (1) = {churned_count}")
    print(f"Churn percentage: {churn_percentage:.2f}%")
    
    # Save a distribution plot
    plt.figure(figsize=(6, 5))
    sns.countplot(x='Exited', hue='Exited', data=df, palette={0: '#1f77b4', 1: '#d62728'}, legend=False)
    plt.title('Customer Churn Class Distribution')
    plt.xlabel('Class (0: Retained, 1: Churned)')
    plt.ylabel('Count')
    for i, count in enumerate([retained_count, churned_count]):
        plt.text(i, count + 100, f"{count}\n({count/len(df)*100:.1f}%)", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'class_distribution.png'), dpi=300)
    plt.close()
    
    # 2. Train / Validation / Test Splitting (60% Train, 20% Val, 20% Test)
    print("\n--- Splitting Data ---")
    X = df.drop(columns=['Exited'])
    y = df['Exited']
    
    # Stratified split to preserve target ratio
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    
    print(f"Train set size: {X_train.shape[0]} ({len(X_train)/len(df)*100:.1f}%)")
    print(f"Validation set size: {X_val.shape[0]} ({len(X_val)/len(df)*100:.1f}%)")
    print(f"Test set size: {X_test.shape[0]} ({len(X_test)/len(df)*100:.1f}%)")
    
    # 3. Preprocessing (Fit on Train, transform on Val and Test to avoid data leakage)
    print("\n--- Preprocessing Features ---")
    numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    categorical_cols = ['Geography', 'Gender']
    # HasCrCard and IsActiveMember are binary and already 0/1, we leave them untreated by scaling
    binary_cols = ['HasCrCard', 'IsActiveMember']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
        ],
        remainder='passthrough' # Leave binary columns as is
    )
    
    # Fit preprocessor on training data
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)
    
    # Reconstruct column names for analysis & feature importance
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_cols = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    feature_names = numerical_cols + encoded_cat_cols + binary_cols
    
    print(f"Processed feature names: {feature_names}")
    
    # 4. Handle Class Imbalance with SMOTE
    print("\n--- Handling Class Imbalance (SMOTE) ---")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_proc, y_train)
    print(f"Original Training Target Count: Retained = {np.sum(y_train==0)}, Churned = {np.sum(y_train==1)}")
    print(f"SMOTE Training Target Count: Retained = {np.sum(y_train_smote==0)}, Churned = {np.sum(y_train_smote==1)}")
    
    # Calculate scale_pos_weight for XGBoost (Ratio of negative to positive cases)
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
    print(f"XGBoost scale_pos_weight: {scale_pos_weight:.4f}")
    
    # 5. Model Configurations
    # We will try:
    # - Logistic Regression (Base, Balanced, SMOTE)
    # - Random Forest (Base, Balanced, SMOTE)
    # - XGBoost (Base, Weighted, SMOTE)
    models_config = {
        'Logistic Regression (Base)': LogisticRegression(max_iter=1000, random_state=42),
        'Logistic Regression (Balanced)': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Logistic Regression (SMOTE)': LogisticRegression(max_iter=1000, random_state=42),
        
        'Random Forest (Base)': RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1),
        'Random Forest (Balanced)': RandomForestClassifier(n_estimators=150, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1),
        'Random Forest (SMOTE)': RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1),
        
        'XGBoost (Base)': XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1, eval_metric='logloss'),
        'XGBoost (Weighted)': XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1, eval_metric='logloss'),
        'XGBoost (SMOTE)': XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1, eval_metric='logloss')
    }
    
    # Naive baseline (Predict majority class 0)
    y_val_naive = np.zeros_like(y_val)
    results = {
        'Naive Baseline': {
            'Accuracy': accuracy_score(y_val, y_val_naive),
            'Precision': precision_score(y_val, y_val_naive, zero_division=0),
            'Recall': recall_score(y_val, y_val_naive),
            'F1-Score': f1_score(y_val, y_val_naive),
            'ROC-AUC': 0.5,
            'PR-AUC': average_precision_score(y_val, np.zeros_like(y_val) + churn_percentage/100),
            'model_obj': None
        }
    }
    
    print("\n--- Training and Evaluating Models on Validation Set ---")
    for name, model in models_config.items():
        print(f"Training {name}...")
        if 'SMOTE' in name:
            model.fit(X_train_smote, y_train_smote)
        else:
            model.fit(X_train_proc, y_train)
            
        y_val_pred = model.predict(X_val_proc)
        y_val_proba = model.predict_proba(X_val_proc)[:, 1]
        
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
        y_val_proba = model.predict_proba(X_val_proc)[:, 1]
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
    
    # 6. Select the Best Model Configurations
    # We will choose:
    # - Logistic Regression (Balanced)
    # - Random Forest (SMOTE)
    # - XGBoost (Weighted)
    # These represent the most robust configurations for each algorithm family.
    selected_configs = {
        'Logistic Regression': results['Logistic Regression (Balanced)']['model_obj'],
        'Random Forest': results['Random Forest (SMOTE)']['model_obj'],
        'XGBoost': results['XGBoost (Weighted)']['model_obj']
    }
    
    test_results = {}
    print("\n--- Evaluating Best Model Configurations on Test Set ---")
    
    with open(metrics_path, "a") as f:
        f.write("=== Test Set Performance of Selected Models ===\n")
        
    # Plot ROC and PR Curves for Test Set
    fig_roc, ax_roc = plt.subplots(figsize=(8, 7))
    fig_pr, ax_pr = plt.subplots(figsize=(8, 7))
    
    best_model_name = None
    best_f1 = 0
    best_model_obj = None
    
    for name, model in selected_configs.items():
        y_test_pred = model.predict(X_test_proc)
        y_test_proba = model.predict_proba(X_test_proc)[:, 1]
        
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
                    xticklabels=['Retained', 'Churned'], 
                    yticklabels=['Retained', 'Churned'])
        plt.title(f'Confusion Matrix - {name} (Test Set)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'confusion_matrix_{name.lower().replace(" ", "_")}.png'), dpi=300)
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
    ax_pr.plot([0, 1], [churn_percentage/100, churn_percentage/100], 'k--', alpha=0.5, label='No Skill')
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
    
    # 7. Feature Importance Analysis
    print("\n--- Feature Importance Analysis ---")
    
    # Random Forest Importance
    rf_model = selected_configs['Random Forest']
    rf_importances = rf_model.feature_importances_
    
    # XGBoost Importance
    xgb_model = selected_configs['XGBoost']
    xgb_importances = xgb_model.feature_importances_
    
    # Plot feature importances side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Sort RF features
    indices_rf = np.argsort(rf_importances)[::-1]
    sorted_features_rf = [feature_names[i] for i in indices_rf]
    sns.barplot(ax=axes[0], x=rf_importances[indices_rf][:10], y=sorted_features_rf[:10], palette="viridis", hue=sorted_features_rf[:10], legend=False)
    axes[0].set_title('Random Forest Feature Importance (Top 10)')
    axes[0].set_xlabel('Relative Importance')
    
    # Sort XGB features
    indices_xgb = np.argsort(xgb_importances)[::-1]
    sorted_features_xgb = [feature_names[i] for i in indices_xgb]
    sns.barplot(ax=axes[1], x=xgb_importances[indices_xgb][:10], y=sorted_features_xgb[:10], palette="magma", hue=sorted_features_xgb[:10], legend=False)
    axes[1].set_title('XGBoost Feature Importance (Top 10)')
    axes[1].set_xlabel('Relative Importance')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_importances.png'), dpi=300)
    plt.close()
    
    # 8. Save preprocessor and best model objects
    best_model_path = os.path.join(current_dir, "best_churn_model.joblib")
    preprocessor_path = os.path.join(current_dir, "preprocessor.joblib")
    
    joblib.dump(best_model_obj, best_model_path)
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Saved preprocessor to {preprocessor_path}")
    print(f"Saved best model ({best_model_name}) to {best_model_path}")

if __name__ == "__main__":
    main()
