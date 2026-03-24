import re


def clean_text(text: str) -> str:
    """Normalize text by lowercasing and removing punctuation, digits, and extra spaces."""
    if text is None:
        return ""
    s = str(text).lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
