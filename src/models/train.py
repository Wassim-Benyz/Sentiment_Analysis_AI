import os
import sys
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "sentiment_matplotlib")
)

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
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


def save_confusion_matrix(cm, class_names, model_name, output_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    threshold = cm.max() / 2
    for row_idx, row in enumerate(cm):
        for col_idx, value in enumerate(row):
            text_color = "white" if value > threshold else "black"
            ax.text(
                col_idx,
                row_idx,
                str(value),
                ha="center",
                va="center",
                color=text_color,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_error_analysis(original_reviews, y_true, y_pred, output_dir):
    error_df = pd.DataFrame(
        {
            "Original Review": list(original_reviews),
            "True Label": list(y_true),
            "Predicted Label": list(y_pred),
        }
    )
    error_df = error_df[error_df["True Label"] != error_df["Predicted Label"]]

    examples_path = os.path.join(output_dir, "misclassified_examples.csv")
    error_df.head(20).to_csv(examples_path, index=False, encoding="utf-8")

    class_counts = error_df["True Label"].value_counts()
    confusion_counts = (
        error_df.groupby(["True Label", "Predicted Label"])
        .size()
        .sort_values(ascending=False)
    )

    summary_path = os.path.join(output_dir, "error_analysis.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Error Analysis\n")
        f.write("==============\n\n")
        f.write(f"Total number of misclassified samples: {len(error_df)}\n\n")

        f.write("Number of mistakes for each true class:\n")
        for class_name in ["negative", "neutral", "positive"]:
            f.write(f"- {class_name}: {int(class_counts.get(class_name, 0))}\n")

        f.write("\nFive representative misclassified examples:\n")
        for idx, row in enumerate(error_df.head(5).itertuples(index=False), start=1):
            f.write(f"\nExample {idx}\n")
            f.write(f"Original Review: {row[0]}\n")
            f.write(f"True Label: {row[1]}\n")
            f.write(f"Predicted Label: {row[2]}\n")

        f.write("\nAutomatically generated summary:\n")
        if confusion_counts.empty:
            f.write("No misclassified samples were found in the test set.\n")
        else:
            most_common = confusion_counts.head(3)
            confusion_descriptions = [
                f"{true_label} reviews predicted as {predicted_label} ({count} cases)"
                for (true_label, predicted_label), count in most_common.items()
            ]
            f.write(
                "The most frequent confusions were "
                + "; ".join(confusion_descriptions)
                + ".\n"
            )

    return examples_path, summary_path


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
    original_reviews = df["Text"].astype(str)
    X = original_reviews.apply(clean_text)
    y = df["sentiment"].astype(str)

    (
        X_train,
        X_test,
        _,
        original_reviews_test,
        y_train,
        y_test,
    ) = train_test_split(
        X, original_reviews, y, test_size=0.2, random_state=42, stratify=y
    )

    # Keep the same TF-IDF configuration used by the original baseline.
    vect = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    X_train_vec = vect.fit_transform(X_train)
    X_test_vec = vect.transform(X_test)

    models = {
        "Logistic Regression baseline": (
            LogisticRegression(max_iter=1000, n_jobs=1, random_state=42),
            "confusion_matrix_logistic_regression.png",
        ),
        "Logistic Regression balanced": (
            LogisticRegression(
                max_iter=1000, n_jobs=1, random_state=42, class_weight="balanced"
            ),
            "confusion_matrix_logistic_regression_balanced.png",
        ),
        "Multinomial Naive Bayes": (
            MultinomialNB(),
            "confusion_matrix_naive_bayes.png",
        ),
        "Linear SVM": (
            LinearSVC(random_state=42, max_iter=5000),
            "confusion_matrix_linear_svm.png",
        ),
    }

    class_names = ["negative", "neutral", "positive"]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    evaluation_dir = os.path.join(project_root, "evaluation")
    os.makedirs(evaluation_dir, exist_ok=True)

    results = []
    best_model_name = None
    best_model = None
    best_preds = None
    best_macro_f1 = -1.0

    for model_name, (clf, confusion_matrix_filename) in models.items():
        print(f"\n=== {model_name} ===")

        clf.fit(X_train_vec, y_train)
        preds = clf.predict(X_test_vec)

        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro")
        weighted_f1 = f1_score(y_test, preds, average="weighted")
        report = classification_report(y_test, preds)
        cm = confusion_matrix(y_test, preds, labels=class_names)
        confusion_matrix_path = os.path.join(evaluation_dir, confusion_matrix_filename)
        save_confusion_matrix(cm, class_names, model_name, confusion_matrix_path)

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
            best_preds = preds

    error_examples_path, error_summary_path = save_error_analysis(
        original_reviews_test, y_test, best_preds, evaluation_dir
    )

    # Save artifacts to models/ inside the project
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
    print(f"Saved misclassified examples to: {error_examples_path}")
    print(f"Saved error analysis to: {error_summary_path}")
    print(f"Saved vectorizer to: {vect_path}")
    print(f"Saved best model to: {model_path}")


if __name__ == "__main__":
    main()
