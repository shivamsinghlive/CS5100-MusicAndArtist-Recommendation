from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None

from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL_NAME


class EmbeddingRecommender:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.df = None
        self.index = None

    def fit(self, df: pd.DataFrame, text_col: str = "cleaned_text") -> "EmbeddingRecommender":
        self.df = df.reset_index(drop=True).copy()
        texts = self.df[text_col].fillna("").tolist()
        self.embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        self._build_index()
        return self

    def _build_index(self) -> None:
        if self.embeddings is None:
            raise ValueError("Embeddings are not available.")
        emb = np.asarray(self.embeddings, dtype=np.float32).copy()
        if faiss is not None:
            faiss.normalize_L2(emb)
            self.index = faiss.IndexFlatIP(emb.shape[1])
            self.index.add(emb)
        else:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            self.embeddings = emb / np.clip(norms, 1e-12, None)

    def save(self, embeddings_path: str | Path, metadata_path: str | Path) -> None:
        if self.df is None or self.embeddings is None:
            raise ValueError("Nothing to save. Call fit() first.")
        np.save(embeddings_path, self.embeddings)
        self.df.to_parquet(metadata_path, index=False)

    def recommend(self, song_name: str, top_n: int = 5) -> pd.DataFrame:
        if self.df is None or self.embeddings is None:
            raise ValueError("Call fit() before recommend().")
        matches = self.df[self.df["song"].str.lower() == song_name.lower()]
        if matches.empty:
            raise ValueError("Song not found in dataset.")
        idx = matches.index[0]
        if self.index is not None and faiss is not None:
            query = np.asarray(self.embeddings[idx:idx+1], dtype=np.float32).copy()
            faiss.normalize_L2(query)
            scores, indices = self.index.search(query, top_n + 1)
            ranked = [i for i in indices[0].tolist() if i != idx][:top_n]
            sim_scores = [float(s) for i, s in zip(indices[0].tolist(), scores[0].tolist()) if i != idx][:top_n]
        else:
            query = self.embeddings[idx]
            scores = self.embeddings @ query
            ranked = scores.argsort()[::-1]
            ranked = [i for i in ranked if i != idx][:top_n]
            sim_scores = [float(scores[i]) for i in ranked]
        result = self.df.loc[ranked, [col for col in ["artist", "song"] if col in self.df.columns]].copy()
        result["score"] = sim_scores
        return result.reset_index(drop=True)

    def semantic_search(self, prompt: str, top_n: int = 5) -> pd.DataFrame:
        if self.df is None or self.embeddings is None:
            raise ValueError("Call fit() before semantic_search().")
        query = self.model.encode([prompt], convert_to_numpy=True)
        query = np.asarray(query, dtype=np.float32)
        if self.index is not None and faiss is not None:
            faiss.normalize_L2(query)
            scores, indices = self.index.search(query, top_n)
            ranked = indices[0].tolist()
            sim_scores = scores[0].tolist()
        else:
            q = query[0] / max(np.linalg.norm(query[0]), 1e-12)
            scores = self.embeddings @ q
            ranked = scores.argsort()[::-1][:top_n]
            sim_scores = [float(scores[i]) for i in ranked]
        result = self.df.loc[ranked, [col for col in ["artist", "song"] if col in self.df.columns]].copy()
        result["score"] = sim_scores
        return result.reset_index(drop=True)
