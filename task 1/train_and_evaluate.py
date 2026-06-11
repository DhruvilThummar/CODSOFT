import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import nltk
from nltk.corpus import stopwords
import time
from tqdm import tqdm

# Download common English stopwords (words like 'is', 'the', 'and' which don't help find the genre)
print("Checking/downloading NLTK stopwords...")
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# This function cleans the movie plot summary
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Convert all letters to lowercase (e.g., 'Movie' becomes 'movie')
    text = text.lower()
    # Remove punctuation, symbols, and numbers (keep only letters and spaces)
    text = re.sub(r'[^a-z\s]', '', text)
    # Split text into single words
    words = text.split()
    # Remove stopwords (words in the list we downloaded earlier)
    cleaned_words = [w for w in words if w not in stop_words]
    # Join the remaining words back into a single sentence
    return " ".join(cleaned_words)

# This function reads the text file and loads it into a table (Pandas DataFrame)
def load_data(filepath):
    print(f"Loading data from {filepath}...")
    # The columns in the file are separated by ' ::: '
    # The columns are: Movie ID, Movie Title, Genre, and Description
    df = pd.read_csv(
        filepath,
        sep=' ::: ',
        engine='python',
        names=['id', 'title', 'genre', 'description'],
        header=None
    )
    return df

def main():
    start_time = time.time()
    
    # Locate directories for files and plots
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Genre_Classification_Dataset")
    train_path = os.path.join(dataset_dir, "train_data.txt")
    test_sol_path = os.path.join(dataset_dir, "test_data_solution.txt")
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(plots_dir, exist_ok=True) # Create plots folder if it doesn't exist
    
    # 1. Load the Datasets
    train_df = load_data(train_path)
    test_df = load_data(test_sol_path)
    
    print(f"Train dataset shape: {train_df.shape}")
    print(f"Test dataset shape: {test_df.shape}")
    
    # 2. Check for missing values
    print("\nChecking for missing values in Train dataset:")
    print(train_df.isnull().sum())
    print("\nChecking for missing values in Test dataset:")
    print(test_df.isnull().sum())
    
    # Drop rows if they are missing description or genre (just to be safe)
    train_df = train_df.dropna(subset=['description', 'genre'])
    test_df = test_df.dropna(subset=['description', 'genre'])
    
    # 3. Explore Genre Distribution (How many movies of each genre do we have?)
    genre_counts = train_df['genre'].value_counts()
    print("\nGenre distribution in Train dataset:")
    print(genre_counts)
    
    # Create and save a bar chart showing the count of movies for each genre
    plt.figure(figsize=(12, 6))
    sns.barplot(x=genre_counts.values, y=genre_counts.index, palette='viridis')
    plt.title("Movie Genre Distribution (Training Set)")
    plt.xlabel("Number of Movies")
    plt.ylabel("Genre")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "genre_distribution.png"))
    plt.close()
    print("Saved genre distribution plot to plots/genre_distribution.png")
    
    # 4. Clean all movie descriptions
    print("\nPreprocessing movie descriptions (cleaning text)...")
    tqdm.pandas(desc="Preprocessing Train descriptions")
    train_df['clean_description'] = train_df['description'].progress_apply(preprocess_text)
    
    tqdm.pandas(desc="Preprocessing Test descriptions")
    test_df['clean_description'] = test_df['description'].progress_apply(preprocess_text)
    
    # 5. Split Train Data into Train and Validation sets (80% Train, 20% Validation)
    # This allows us to test our models on unseen data before doing final testing.
    # We use "stratify" so that both sets have the same percentage of each genre.
    print("\nSplitting training data into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        train_df['clean_description'],
        train_df['genre'],
        test_size=0.2,
        random_state=42,
        stratify=train_df['genre']
    )
    
    # 6. Convert Text to Numbers (TF-IDF Vectorization)
    # Computers understand numbers, not text. TF-IDF gives a score to words:
    # - Words that appear often in a movie but rarely in other movies get a high score.
    # - Words that appear in almost all movies get a low score.
    # We fit the vectorizer ONLY on the training set to prevent leakage.
    print("\nExtracting features using TF-IDF...")
    # ngram_range=(1, 2) looks at single words and pairs of words (e.g. "sci fi", "not good")
    # max_features=20000 keeps only the top 20,000 most common and useful word features
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    
    X_train_vec = tfidf.fit_transform(X_train) # Fit and transform train set
    X_val_vec = tfidf.transform(X_val)         # Transform validation set
    X_test_vec = tfidf.transform(test_df['clean_description']) # Transform test set
    
    print(f"TF-IDF representation shape: {X_train_vec.shape}")
    
    # 7. Train and compare 3 models
    # Models we compare:
    # 1. Multinomial Naive Bayes (a simple algorithm based on probability)
    # 2. Logistic Regression (a classic model that calculates word weights)
    # 3. Linear Support Vector Classifier (LinearSVC - finds the best boundary lines between genres)
    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Linear Support Vector Classifier": LinearSVC(random_state=42, C=1.0)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        m_start = time.time()
        model.fit(X_train_vec, y_train) # Train the model
        m_train_time = time.time() - m_start
        
        # Predict genres for validation set
        y_val_pred = model.predict(X_val_vec)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        print(f"Finished training in {m_train_time:.2f}s. Validation Accuracy: {val_acc:.4%}")
        results[name] = {
            "model": model,
            "val_accuracy": val_acc,
            "train_time": m_train_time
        }
        
    # Find the best model based on validation accuracy
    best_model_name = max(results, key=lambda k: results[k]['val_accuracy'])
    print(f"\nBest Model by Validation Accuracy: {best_model_name}")
    
    # Create and save a bar chart comparing validation accuracy of the three models
    plt.figure(figsize=(8, 4))
    acc_values = [results[name]['val_accuracy'] for name in results]
    sns.barplot(x=list(results.keys()), y=acc_values, palette='magma')
    plt.title("Model Validation Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    for i, v in enumerate(acc_values):
        plt.text(i, v + 0.02, f"{v:.2%}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "model_comparison.png"))
    plt.close()
    print("Saved model comparison plot to plots/model_comparison.png")
    
    # 8. Evaluate our best model on the independent test set
    best_model = results[best_model_name]['model']
    print(f"\nEvaluating the best model ({best_model_name}) on the final test set...")
    y_test = test_df['genre']
    y_test_pred = best_model.predict(X_test_vec)
    
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_acc:.4%}")
    
    # Print metrics report (shows Precision, Recall, and F1-score for each genre)
    print("\nTest Classification Report:")
    report = classification_report(y_test, y_test_pred, zero_division=0)
    print(report)
    
    # 9. Plot Confusion Matrix for the Best Model
    # A confusion matrix shows how many movies were classified correctly vs. incorrectly.
    # To keep the chart clean, we plot the matrix for the top 10 most common genres.
    top_genres = genre_counts.index[:10].tolist()
    
    # Filter test labels so we only plot the top 10 genres
    mask = y_test.isin(top_genres) & pd.Series(y_test_pred).isin(top_genres).values
    y_test_filtered = y_test[mask]
    y_test_pred_filtered = y_test_pred[mask]
    
    cm = confusion_matrix(y_test_filtered, y_test_pred_filtered, labels=top_genres)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        xticklabels=top_genres,
        yticklabels=top_genres,
        cmap='Blues'
    )
    plt.title(f"Confusion Matrix (Top 10 Genres) - {best_model_name}")
    plt.xlabel("Predicted Genre")
    plt.ylabel("True Genre")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"))
    plt.close()
    print("Saved confusion matrix plot to plots/confusion_matrix.png")
    
    # Save the text report of all metrics to a text file
    with open(os.path.join(plots_dir, "evaluation_report.txt"), "w") as f:
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"Validation Accuracy: {results[best_model_name]['val_accuracy']:.4%}\n")
        f.write(f"Test Accuracy: {test_acc:.4%}\n\n")
        f.write("Test Classification Report:\n")
        f.write(report)
    print("Saved evaluation report text to plots/evaluation_report.txt")
    
    total_time = time.time() - start_time
    print(f"\nPipeline finished successfully in {total_time:.2f}s!")

if __name__ == "__main__":
    main()
