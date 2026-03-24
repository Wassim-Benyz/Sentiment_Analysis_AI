import os
import sys
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.text_cleaning import clean_text


def find_models_dir():
    """Return the path to the project's `models/` directory.

    Tries the project layout relative to this file first, then CWD as a fallback.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    models_dir = os.path.join(project_root, "models")
    if os.path.exists(models_dir):
        return models_dir
    # fallback: relative to CWD
    cwd_models = os.path.join(os.getcwd(), "models")
    if os.path.exists(cwd_models):
        return cwd_models
    raise FileNotFoundError("Could not find models/ directory.")


def load_artifacts():
    """Load the saved TF-IDF vectorizer and classifier from models/."""
    models_dir = find_models_dir()
    vect_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    model_path = os.path.join(models_dir, "logreg_model.joblib")
    if not os.path.exists(vect_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Model or vectorizer not found in models/.")
    vectorizer = joblib.load(vect_path)
    classifier = joblib.load(model_path)
    return vectorizer, classifier


def predict_text(text: str, vectorizer, classifier):
    """Return predicted label and optional probabilities for a single text."""
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    pred = classifier.predict(X)[0]
    probs = None
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(X)[0]
    return pred, probs


def main():
    # accept CLI args or prompt interactively
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = input("Enter text to predict sentiment: \n")

    vectorizer, classifier = load_artifacts()
    pred, probs = predict_text(text, vectorizer, classifier)

    print(f"Predicted sentiment: {pred}")
    if probs is not None:
        labels = classifier.classes_
        prob_str = ", ".join([f"{lab}:{p:.3f}" for lab, p in zip(labels, probs)])
        print(f"Class probabilities: {prob_str}")


if __name__ == "__main__":
    main()
