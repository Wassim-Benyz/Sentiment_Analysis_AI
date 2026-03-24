import os
import re
import pandas as pd


def find_reviews_csv(base_dir="data"):
    """Locate Reviews.csv in either the current working directory or the project.

    Search order:
    1) CWD/base_dir/Reviews.csv
    2) any child under CWD/base_dir
    3) project-relative (script location) base_dir
    """
    # try CWD first
    target = os.path.join(os.getcwd(), base_dir, "Reviews.csv")
    if os.path.exists(target):
        return target

    # search under CWD/base_dir
    cwd_base = os.path.join(os.getcwd(), base_dir)
    if os.path.exists(cwd_base):
        for root, _, files in os.walk(cwd_base):
            if "Reviews.csv" in files:
                return os.path.join(root, "Reviews.csv")

    # fallback: check relative to this script (typical project layout)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_data = os.path.abspath(os.path.join(script_dir, "..", "..", base_dir))
    target2 = os.path.join(project_data, "Reviews.csv")
    if os.path.exists(target2):
        return target2
    if os.path.exists(project_data):
        for root, _, files in os.walk(project_data):
            if "Reviews.csv" in files:
                return os.path.join(root, "Reviews.csv")

    return None


def clean_text(text: str) -> str:
    """Simple cleaning: lowercase, remove non-letters, collapse spaces.

    Returns the cleaned string, or empty string if input is missing.
    """
    if pd.isna(text):
        return ""
    s = str(text).lower()
    # keep letters and spaces only
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    """Load Reviews.csv, keep Text/Score, map scores to sentiment, clean text, save CSV."""
    csv_path = find_reviews_csv("data")
    if csv_path is None:
        raise FileNotFoundError("Could not find Reviews.csv under the data/ folder.")

    df = pd.read_csv(csv_path, low_memory=False)

    if not set(["Text", "Score"]).issubset(df.columns):
        raise KeyError("Expected columns 'Text' and 'Score' in Reviews.csv")

    # keep only needed columns and drop rows missing either
    df = df[["Text", "Score"]].copy()
    df.dropna(subset=["Text", "Score"], inplace=True)

    # map numeric score to a simple sentiment label
    mapping = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}
    df["sentiment"] = df["Score"].map(mapping)

    # clean the text column in-place
    df["Text"] = df["Text"].astype(str).apply(clean_text)

    data_dir = os.path.dirname(csv_path)
    out_path = os.path.join(data_dir, "processed_reviews.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")

    # show a small sample and the class balance
    print("First 5 rows:\n")
    print(df.head(5).to_string(index=False))
    print("\nClass counts:\n")
    print(df["sentiment"].value_counts())


if __name__ == "__main__":
    main()
import pandas as pd
import re
import string
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "Reviews.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed_reviews.csv"


def score_to_sentiment(score):
    if score in [1, 2]:
        return "negative"
    elif score == 3:
        return "neutral"
    else:
        return "positive"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    df = pd.read_csv(DATA_PATH)

    df = df[["Text", "Score"]].copy()
    df = df.dropna()

    df["sentiment"] = df["Score"].apply(score_to_sentiment)
    df["clean_text"] = df["Text"].apply(clean_text)

    df = df[df["clean_text"].str.len() > 0]

    df.to_csv(OUTPUT_PATH, index=False)

    print(df[["clean_text", "sentiment"]].head())
    print("\nClass distribution:")
    print(df["sentiment"].value_counts())


if __name__ == "__main__":
    main()