from __future__ import annotations

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def download_nltk_resources() -> None:
    for resource in ["punkt", "stopwords"]:
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource)


def get_stopwords() -> set[str]:
    download_nltk_resources()
    return set(stopwords.words("english"))


def preprocess_text(text: str, stop_words: set[str] | None = None) -> str:
    if not isinstance(text, str):
        return ""
    stop_words = stop_words or get_stopwords()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)


def add_cleaned_text(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    stop_words = get_stopwords()
    output = df.copy()
    output["cleaned_text"] = output[text_col].fillna("").apply(lambda value: preprocess_text(value, stop_words))
    return output
