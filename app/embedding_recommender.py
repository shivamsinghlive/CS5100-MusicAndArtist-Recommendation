from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

try:
    import faiss
except ImportError:
    faiss = None

from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL_NAME


class EmbeddingRecommender:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None  # lazy load
        self.embeddings: np.ndarray | None = None
        self.df: pd.DataFrame | None = None
        self.index = None

        # song name → df index lookup (built once in _build_index)
        self._song_index: dict[str, int] = {}

    # ── Lazy model load: only instantiate when actually encoding ──────────────
    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, df: pd.DataFrame, text_col: str = "cleaned_text") -> "EmbeddingRecommender":
        self.df = df.reset_index(drop=True).copy()
        texts = self.df[text_col].fillna("").tolist()
        self.embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=64,        # explicit batch size for speed
            normalize_embeddings=True,  # normalize here → skip extra step later
        )
        self.embeddings = np.asarray(self.embeddings, dtype=np.float32)
        self._build_index()
        return self

    def _build_index(self) -> None:
        if self.embeddings is None:
            raise ValueError("No embeddings available.")

        emb = self.embeddings.copy()

        if faiss is not None:
            # Already normalized if loaded from fit(); normalize again just in case
            faiss.normalize_L2(emb)
            self.index = faiss.IndexFlatIP(emb.shape[1])
            self.index.add(emb)
        else:
            # Fallback: keep normalized embeddings for dot-product similarity
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            self.embeddings = emb / np.clip(norms, 1e-12, None)

        # Build O(1) song-name lookup dict (case-insensitive)
        if self.df is not None:
            self._song_index = {
                name.lower(): idx
                for idx, name in enumerate(self.df["song"].fillna(""))
            }

        # Cache for semantic_search: prompt → encoded vector
        self._prompt_cache: dict[str, np.ndarray] = {}

    def save(self, embeddings_path: str | Path, metadata_path: str | Path) -> None:
        if self.df is None or self.embeddings is None:
            raise ValueError("Nothing to save — call fit() first.")
        np.save(embeddings_path, self.embeddings)
        self.df.to_parquet(metadata_path, index=False)

    # ── Internal: shared FAISS / numpy search ─────────────────────────────────
    def _search(
        self,
        query_vec: np.ndarray,
        top_n: int,
        exclude_idx: int | None = None,
        offset: int = 0,
    ):
        """
        query_vec: (1, D) float32, already normalized.
        Returns (ranked_indices, scores).
        """
        start = max(0, int(offset))
        fetch = top_n + start + (1 if exclude_idx is not None else 0)

        if self.index is not None and faiss is not None:
            faiss.normalize_L2(query_vec)
            scores, indices = self.index.search(query_vec, fetch)
            pairs = list(zip(indices[0].tolist(), scores[0].tolist()))
        else:
            q = query_vec[0]
            raw_scores = self.embeddings @ q
            top_idx = raw_scores.argsort()[::-1][:fetch]
            pairs = [(int(i), float(raw_scores[i])) for i in top_idx]

        if exclude_idx is not None:
            pairs = [(i, s) for i, s in pairs if i != exclude_idx]

        pairs = pairs[start:start + top_n]
        return [i for i, _ in pairs], [s for _, s in pairs]

    # ── Public API ─────────────────────────────────────────────────────────────
    def recommend(self, song_name: str, top_n: int = 5, offset: int = 0) -> pd.DataFrame:
        if self.df is None or self.embeddings is None:
            raise ValueError("Call fit() before recommend().")

        # O(1) lookup instead of full DataFrame scan
        idx = self._song_index.get(song_name.lower())
        if idx is None:
            raise ValueError(f"Song not found: '{song_name}'")

        query = np.asarray(self.embeddings[idx:idx+1], dtype=np.float32)
        ranked, sim_scores = self._search(query, top_n, exclude_idx=idx, offset=offset)

        result = self.df.loc[ranked, [c for c in ["artist", "song"] if c in self.df.columns]].copy()
        result["score"] = sim_scores
        return result.reset_index(drop=True)

    def semantic_search(self, prompt: str, top_n: int = 5, offset: int = 0) -> pd.DataFrame:
        if self.df is None or self.embeddings is None:
            raise ValueError("Call fit() before semantic_search().")

        # Return cached vector if same prompt was used before
        if prompt not in self._prompt_cache:
            self._prompt_cache[prompt] = self.model.encode(
                [prompt],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)

        query = self._prompt_cache[prompt]
        ranked, sim_scores = self._search(query, top_n, offset=offset)

        result = self.df.loc[ranked, [c for c in ["artist", "song"] if c in self.df.columns]].copy()
        result["score"] = sim_scores
        return result.reset_index(drop=True)