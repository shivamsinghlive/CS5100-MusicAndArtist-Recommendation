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
    join_cols = join_cols or ["artist", "song"]
    return pd.merge(lyrics_df, audio_df, on=join_cols, how="inner")
