from __future__ import annotations

from pathlib import Path

import pandas as pd

# Column names seen on Kaggle “Spotify / Million Song” companion CSVs
_ARTIST_ALIASES = ("artist", "artists", "Artist", "ARTIST", "artist_name", "artists_name")
_SONG_ALIASES = (
    "song",
    "track_name",
    "track name",
    "track",
    "name",
    "title",
    "song_title",
    "track_title",
)

# HybridRecommender defaults; keep any other numeric Spotify audio columns too
_PREFERRED_AUDIO_COLS = (
    "tempo",
    "energy",
    "valence",
    "danceability",
    "loudness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
    "key",
    "mode",
    "time_signature",
)


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lower_map = {c.lower().replace(" ", "_"): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "_")
        if key in lower_map:
            return lower_map[key]
        if cand in df.columns:
            return cand
    return None


def normalize_join_key(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def standardize_kaggle_audio_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map a Kaggle-style Spotify features table to columns: artist, song + numeric audio features.
    Deduplicates on normalized (artist, song).
    """
    artist_col = _pick_column(df, _ARTIST_ALIASES)
    song_col = _pick_column(df, _SONG_ALIASES)
    if not artist_col or not song_col:
        raise ValueError(
            f"Could not find artist/song columns. Columns present: {list(df.columns)!r}. "
            "Expected something like 'artists' + 'track_name' or 'artist' + 'song'."
        )

    out = df.rename(columns={artist_col: "artist", song_col: "song"}).copy()
    out["artist"] = out["artist"].astype(str).str.strip()
    out["song"] = out["song"].astype(str).str.strip()

    numeric_cols: list[str] = []
    for c in out.columns:
        if c in ("artist", "song"):
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            numeric_cols.append(c)
        else:
            # try coerce object columns that look numeric
            try:
                converted = pd.to_numeric(out[c], errors="coerce")
                if converted.notna().sum() > len(out) * 0.5:
                    out[c] = converted
                    numeric_cols.append(c)
            except Exception:
                pass

    # Prefer known Spotify feature names first, then any other numeric
    ordered = [c for c in _PREFERRED_AUDIO_COLS if c in numeric_cols]
    ordered += [c for c in numeric_cols if c not in ordered]
    if not ordered:
        raise ValueError("No numeric audio feature columns found after parsing.")

    use = out[["artist", "song"] + ordered].copy()
    use["_ka"] = normalize_join_key(use["artist"])
    use["_ks"] = normalize_join_key(use["song"])
    use = use.drop_duplicates(subset=["_ka", "_ks"], keep="first")
    use = use.drop(columns=["_ka", "_ks"])
    return use.reset_index(drop=True)


def _read_audio_csv_raw(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    drop_unnamed = [c for c in raw.columns if str(c).startswith("Unnamed")]
    if drop_unnamed:
        raw = raw.drop(columns=drop_unnamed)
    return raw


def load_standardized_audio_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio features CSV not found: {path}")
    return standardize_kaggle_audio_csv(_read_audio_csv_raw(path))


def load_merge_standardize_audio_csvs(paths: list[str | Path]) -> pd.DataFrame:
    """
    Load several Spotify feature CSVs (same schema), concatenate, then dedupe.
    If `track_id` exists, dedupe on it (keeps first file’s row — list newer archives first).
    Finally standardize to artist/song + numeric features.
    """
    paths = [Path(p) for p in paths]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Audio features CSV not found: {p}")
    frames = [_read_audio_csv_raw(p) for p in paths]
    combined = pd.concat(frames, ignore_index=True)
    if "track_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["track_id"], keep="first")
    return standardize_kaggle_audio_csv(combined)


def audio_lyrics_match_stats(lyrics_df: pd.DataFrame, audio_df: pd.DataFrame) -> tuple[int, int, float]:
    """Returns (matched_rows, lyrics_rows, match_rate). Match uses normalized artist+song."""
    L = lyrics_df.copy()
    A = audio_df.copy()
    L["_ka"] = normalize_join_key(L["artist"])
    L["_ks"] = normalize_join_key(L["song"])
    A["_ka"] = normalize_join_key(A["artist"])
    A["_ks"] = normalize_join_key(A["song"])
    keys_l = set(zip(L["_ka"], L["_ks"]))
    keys_a = set(zip(A["_ka"], A["_ks"]))
    matched = len(keys_l & keys_a)
    n = len(L)
    rate = (matched / n * 100.0) if n else 0.0
    return matched, n, rate
