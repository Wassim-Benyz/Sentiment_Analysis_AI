import os
import sys
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.svm import LinearSVC

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
    """Compare TF-IDF classifiers on processed_reviews.csv.

    Produces `results_model_comparison.txt`, `models/tfidf_vectorizer.joblib`,
    and the best classifier as `models/logreg_model.joblib` for compatibility.
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

    # Keep the same TF-IDF configuration used by the original baseline.
    vect = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    X_train_vec = vect.fit_transform(X_train)
    X_test_vec = vect.transform(X_test)

    models = {
        "Logistic Regression baseline": LogisticRegression(
            max_iter=1000, n_jobs=1, random_state=42
        ),
        'Logistic Regression balanced': LogisticRegression(
            max_iter=1000, n_jobs=1, random_state=42, class_weight="balanced"
        ),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(random_state=42, max_iter=5000),
    }

    results = []
    best_model_name = None
    best_model = None
    best_macro_f1 = -1.0

    for model_name, clf in models.items():
        print(f"\n=== {model_name} ===")

        clf.fit(X_train_vec, y_train)
        preds = clf.predict(X_test_vec)

        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro")
        weighted_f1 = f1_score(y_test, preds, average="weighted")
        report = classification_report(y_test, preds)

        print(f"Accuracy: {acc:.4f}")
        print(f"Macro F1-score: {macro_f1:.4f}")
        print(f"Weighted F1-score: {weighted_f1:.4f}")
        print("Classification report:\n")
        print(report)

        results.append(
            {
                "model_name": model_name,
                "accuracy": acc,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "classification_report": report,
            }
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_model_name = model_name
            best_model = clf

    # Save artifacts to models/ inside the project
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    results_path = os.path.join(project_root, "results_model_comparison.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("Model Comparison Results\n")
        f.write("========================\n\n")
        for result in results:
            f.write(f"Model: {result['model_name']}\n")
            f.write(f"Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"Macro F1-score: {result['macro_f1']:.4f}\n")
            f.write(f"Weighted F1-score: {result['weighted_f1']:.4f}\n")
            f.write("Classification report:\n\n")
            f.write(result["classification_report"])
            f.write("\n\n")
        f.write(f"Best model based on macro F1-score: {best_model_name}\n")
        f.write(f"Best macro F1-score: {best_macro_f1:.4f}\n")

    vect_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    model_path = os.path.join(models_dir, "logreg_model.joblib")
    joblib.dump(vect, vect_path)
    joblib.dump(best_model, model_path)

    print(f"\nBest model based on macro F1-score: {best_model_name}")
    print(f"Best macro F1-score: {best_macro_f1:.4f}")
    print(f"Saved comparison results to: {results_path}")
    print(f"Saved vectorizer to: {vect_path}")
    print(f"Saved best model to: {model_path}")


if __name__ == "__main__":
    main()
