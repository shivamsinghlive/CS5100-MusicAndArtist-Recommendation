from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRecommender:
    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.matrix = None
        self.df = None

    def fit(self, df: pd.DataFrame, text_col: str = "cleaned_text") -> "TfidfRecommender":
        self.df = df.reset_index(drop=True).copy()
        self.matrix = self.vectorizer.fit_transform(self.df[text_col].fillna(""))
        return self

    def recommend(self, song_name: str, top_n: int = 5) -> pd.DataFrame:
        if self.df is None or self.matrix is None:
            raise ValueError("Call fit() before recommend().")
        matches = self.df[self.df["song"].str.lower() == song_name.lower()]
        if matches.empty:
            raise ValueError("Song not found in dataset.")
        idx = matches.index[0]
        scores = cosine_similarity(self.matrix[idx], self.matrix).flatten()
        ranked = scores.argsort()[::-1]
        ranked = [i for i in ranked if i != idx][:top_n]
        result = self.df.loc[ranked, [col for col in ["artist", "song"] if col in self.df.columns]].copy()
        result["score"] = scores[ranked]
        return result.reset_index(drop=True)
