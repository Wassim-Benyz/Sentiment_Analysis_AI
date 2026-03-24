import os
import sys
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.text_cleaning import clean_text


def find_processed_csv(base_dir="data"):
    """Find processed_reviews.csv in a few likely locations.

    This tries CWD first, then the project-relative data folder.
    """
    # check CWD/data
    candidate = os.path.join(os.getcwd(), base_dir, "processed_reviews.csv")
    if os.path.exists(candidate):
        return candidate
    # search under CWD/data
    for root, _, files in os.walk(os.path.join(os.getcwd(), base_dir)):
        if "processed_reviews.csv" in files:
            return os.path.join(root, "processed_reviews.csv")

    # check project-relative (script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    candidate2 = os.path.join(project_root, base_dir, "processed_reviews.csv")
    if os.path.exists(candidate2):
        return candidate2
    for root, _, files in os.walk(os.path.join(project_root, base_dir)):
        if "processed_reviews.csv" in files:
            return os.path.join(root, "processed_reviews.csv")

    return None


def main():
    """Train a TF-IDF + LogisticRegression model on processed_reviews.csv.

    Produces `models/tfidf_vectorizer.joblib` and `models/logreg_model.joblib`.
    """

    csv_path = find_processed_csv("data")
    if csv_path is None:
        raise FileNotFoundError("Could not find processed_reviews.csv under data/")

    df = pd.read_csv(csv_path)
    if not set(["Text", "sentiment"]).issubset(df.columns):
        raise KeyError("Expected columns 'Text' and 'sentiment' in processed_reviews.csv")

    # ensure no missing values for training
    df = df.dropna(subset=["Text", "sentiment"]).copy()
    X = df["Text"].astype(str).apply(clean_text)
    y = df["sentiment"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # vectorize text and train classifier
    vect = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    X_train_vec = vect.fit_transform(X_train)
    X_test_vec = vect.transform(X_test)

    # Use a single process to avoid joblib/loky semaphore permission issues
    # in constrained environments.
    clf = LogisticRegression(max_iter=1000, n_jobs=1, random_state=42)
    clf.fit(X_train_vec, y_train)

    preds = clf.predict(X_test_vec)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)

    print(f"Accuracy: {acc:.4f}")
    print("Classification report:\n")
    print(report)

    # Save artifacts to models/ inside the project
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    vect_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    model_path = os.path.join(models_dir, "logreg_model.joblib")
    joblib.dump(vect, vect_path)
    joblib.dump(clf, model_path)

    print(f"Saved vectorizer to: {vect_path}")
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
