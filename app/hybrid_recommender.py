from __future__ import annotations

import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


class HybridRecommender:
    def __init__(self, text_max_features: int = 5000, audio_feature_cols: list[str] | None = None):
        self.text_vectorizer = TfidfVectorizer(max_features=text_max_features)
        self.audio_feature_cols = audio_feature_cols or ["tempo", "energy", "valence"]
        self.scaler = StandardScaler()
        self.matrix = None
        self.df = None

    def fit(self, df: pd.DataFrame, text_col: str = "cleaned_text") -> "HybridRecommender":
        self.df = df.reset_index(drop=True).copy()
        tfidf_matrix = self.text_vectorizer.fit_transform(self.df[text_col].fillna(""))
        audio_features = self.df[self.audio_feature_cols].fillna(0.0).values
        audio_scaled = self.scaler.fit_transform(audio_features)
        audio_sparse = sparse.csr_matrix(audio_scaled)
        self.matrix = sparse.hstack([tfidf_matrix, audio_sparse])
        return self

    def recommend(self, song_name: str, top_n: int = 5, offset: int = 0) -> pd.DataFrame:
        if self.df is None or self.matrix is None:
            raise ValueError("Call fit() before recommend().")
        matches = self.df[self.df["song"].str.lower() == song_name.lower()]
        if matches.empty:
            raise ValueError("Song not found in hybrid dataset.")
        idx = matches.index[0]
        scores = cosine_similarity(self.matrix[idx], self.matrix).flatten()
        ranked = scores.argsort()[::-1]
        ranked = [i for i in ranked if i != idx]
        start = max(0, int(offset))
        ranked = ranked[start:start + top_n]
        cols = [col for col in ["artist", "song", *self.audio_feature_cols] if col in self.df.columns]
        result = self.df.loc[ranked, cols].copy()
        result["score"] = scores[ranked]
        return result.reset_index(drop=True)


def apply_context_filters(df: pd.DataFrame, mood: str | None = None, min_energy: float | None = None, language: str | None = None) -> pd.DataFrame:
    filtered = df.copy()
    if mood and mood.lower() == "gym" and "energy" in filtered.columns:
        threshold = 0.7 if min_energy is None else min_energy
        filtered = filtered[filtered["energy"] > threshold]
    elif min_energy is not None and "energy" in filtered.columns:
        filtered = filtered[filtered["energy"] >= min_energy]
    if language and "language" in filtered.columns:
        filtered = filtered[filtered["language"].astype(str).str.lower() == language.lower()]
    return filtered.reset_index(drop=True)
