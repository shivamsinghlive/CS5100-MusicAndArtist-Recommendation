from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class HybridRecommenderArtifacts:
    tfidf_vectorizer: TfidfVectorizer
    tfidf_matrix: sparse.spmatrix
    hybrid_matrix: sparse.spmatrix


def build_hybrid_artifacts(
    combined_df: pd.DataFrame,
    *,
    text_col: str = "cleaned_text",
    audio_feature_cols: Optional[Sequence[str]] = None,
    max_tfidf_features: int = 5000,
) -> HybridRecommenderArtifacts:
    """
    Build reusable matrices for fast recommendation queries.

    Data contract (from `src.data_pipeline.load_and_clean_data`):
    - `combined_df` must contain at least:
      - `track_id` (recommended) OR a unique identifier column you will pass into `get_hybrid_recommendations`
      - `song` (display name)
      - `artist` (display name)
      - `cleaned_text` (string): preprocessed lyrics for TF-IDF
    - If you want hybrid recommendations, `combined_df` should also contain normalized numeric audio features
      (e.g. `tempo`, `energy`, `danceability`, ...). Those columns are passed via `audio_feature_cols`.

    Notes:
    - This function DOES NOT compute an NxN similarity matrix (which is huge). We compute query-to-all cosine
      similarity at request time using `cosine_similarity(query_vec, hybrid_matrix)`.
    """
    if text_col not in combined_df.columns:
        raise ValueError(f"combined_df must include `{text_col}` column")

    text_series = combined_df[text_col].fillna("").astype(str)
    tfidf_vectorizer = TfidfVectorizer(max_features=max_tfidf_features)
    tfidf_matrix = tfidf_vectorizer.fit_transform(text_series)

    if audio_feature_cols:
        missing = [c for c in audio_feature_cols if c not in combined_df.columns]
        if missing:
            raise ValueError(f"combined_df missing audio_feature_cols: {missing}")

        audio = combined_df.loc[:, list(audio_feature_cols)].astype(float)
        audio_sparse = sparse.csr_matrix(audio.values)
        hybrid_matrix = sparse.hstack([tfidf_matrix, audio_sparse], format="csr")
    else:
        hybrid_matrix = tfidf_matrix

    return HybridRecommenderArtifacts(
        tfidf_vectorizer=tfidf_vectorizer,
        tfidf_matrix=tfidf_matrix,
        hybrid_matrix=hybrid_matrix,
    )


def get_hybrid_recommendations(
    song_id: str,
    combined_df: pd.DataFrame,
    top_n: int = 10,
    *,
    id_col: str = "track_id",
    artifacts: Optional[HybridRecommenderArtifacts] = None,
    text_col: str = "cleaned_text",
    audio_feature_cols: Optional[Sequence[str]] = None,
    max_tfidf_features: int = 5000,
) -> pd.DataFrame:
    """
    Return top-N recommendations for `song_id`.

    **Contract**:
    - `combined_df` should contain:
      - `track_id` (or set `id_col` to your ID column)
      - `song`, `artist`
      - `cleaned_text` (preprocessed lyrics, string)
    - Optional hybrid mode:
      - pass `audio_feature_cols` listing numeric, normalized audio columns (tempo, energy, etc.)

    Output:
    - DataFrame with columns: `rank`, `track_id`, `artist`, `song`, `score`
    """
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if id_col not in combined_df.columns:
        raise ValueError(f"combined_df must include `{id_col}` column (or set id_col)")

    if artifacts is None:
        artifacts = build_hybrid_artifacts(
            combined_df,
            text_col=text_col,
            audio_feature_cols=audio_feature_cols,
            max_tfidf_features=max_tfidf_features,
        )

    match = combined_df.index[combined_df[id_col].astype(str) == str(song_id)]
    if len(match) == 0:
        raise KeyError(f"song_id `{song_id}` not found in combined_df[{id_col}]")

    query_row_pos = int(combined_df.index.get_indexer([match[0]])[0])
    query_vec = artifacts.hybrid_matrix[query_row_pos]

    sims = cosine_similarity(query_vec, artifacts.hybrid_matrix).ravel()
    if sims.shape[0] != len(combined_df):
        raise RuntimeError("similarity computation returned unexpected shape")

    # exclude self
    sims[query_row_pos] = -np.inf

    top_idx = np.argpartition(-sims, kth=min(top_n, len(sims) - 1) - 1)[:top_n]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    out = combined_df.iloc[top_idx].copy()
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out["score"] = sims[top_idx]

    # normalize output columns
    keep_cols = ["rank", id_col, "artist", "song", "score"]
    for col in ("artist", "song"):
        if col not in out.columns:
            out[col] = ""
    return out.loc[:, keep_cols].rename(columns={id_col: "track_id"}).reset_index(drop=True)


def make_mock_combined_df() -> pd.DataFrame:
    """
    A tiny fake dataset for Member B to test end-to-end before Member A finishes the pipeline.
    """
    return pd.DataFrame(
        {
            "track_id": ["track_1", "track_2", "track_3"],
            "artist": ["A", "B", "C"],
            "song": ["Love Song", "Sad Song", "Dance Song"],
            "cleaned_text": [
                "love love heart forever",
                "sad tears lonely night",
                "dance party move tonight",
            ],
            # Optional audio features (already normalized here)
            "tempo": [0.4, 0.2, 0.9],
            "energy": [0.5, 0.3, 0.95],
        }
    )

