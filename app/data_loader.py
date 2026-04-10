from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path.suffix}")


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def merge_lyrics_audio(lyrics_df: pd.DataFrame, audio_df: pd.DataFrame, join_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Inner-join lyrics with audio features on normalized artist + song (case-insensitive, stripped).
    Keeps artist/song text from the lyrics dataframe.
    """
    join_cols = join_cols or ["artist", "song"]
    a_col, s_col = join_cols[0], join_cols[1]
    L = lyrics_df.copy()
    A = audio_df.copy()
    key_a, key_s = "_join_artist", "_join_song"
    L[key_a] = L[a_col].astype(str).str.strip().str.lower()
    L[key_s] = L[s_col].astype(str).str.strip().str.lower()
    A[key_a] = A[a_col].astype(str).str.strip().str.lower()
    A[key_s] = A[s_col].astype(str).str.strip().str.lower()
    feature_cols = [c for c in A.columns if c not in (a_col, s_col, key_a, key_s)]
    A_feat = A[[key_a, key_s, *feature_cols]]
    merged = L.merge(A_feat, on=[key_a, key_s], how="inner")
    return merged.drop(columns=[key_a, key_s]).reset_index(drop=True)
