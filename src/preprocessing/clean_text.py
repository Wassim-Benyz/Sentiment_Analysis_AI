import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.text_cleaning import clean_text


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
