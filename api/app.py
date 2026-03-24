import os
import sys
import joblib
from flask import Flask, request, jsonify

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.text_cleaning import clean_text


def find_models_dir():
    """Locate `models/` folder relative to the project or CWD.

    Returns path to models directory or raises FileNotFoundError.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    models_dir = os.path.join(project_root, "models")
    if os.path.exists(models_dir):
        return models_dir
    cwd_models = os.path.join(os.getcwd(), "models")
    if os.path.exists(cwd_models):
        return cwd_models
    raise FileNotFoundError("Could not find models/ directory.")


def load_artifacts():
    """Load the TF-IDF vectorizer and classifier from `models/`.

    Returns (vectorizer, model).
    """
    models_dir = find_models_dir()
    vect_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    model_path = os.path.join(models_dir, "logreg_model.joblib")
    if not os.path.exists(vect_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Model or vectorizer not found in models/.")
    vectorizer = joblib.load(vect_path)
    model = joblib.load(model_path)
    return vectorizer, model


app = Flask(__name__)
# load artifacts at import so the app is ready to serve requests
vectorizer, model = load_artifacts()


@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json()
    text = data.get("text")
    if text is None:
        return jsonify({"error": "JSON must contain 'text' field"}), 400
    # preprocess and predict
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]
    confidence = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        for lab, p in zip(model.classes_, probs):
            confidence[str(lab)] = float(p)

    return jsonify({"sentiment": str(pred), "confidence": confidence})


if __name__ == "__main__":
    # allow running directly for quick tests
    vectorizer, model = load_artifacts()
    app.run(host="0.0.0.0", port=5000)
