"""
One-step: Kaggle-style Spotify audio CSV -> data/processed/spotify_audio_features.parquet

Does not re-run NLTK lyric cleaning. Use after you place the raw audio CSV under data/raw/
(see app.config.AUDIO_FILE) or pass --audio.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from app.audio_features import audio_lyrics_match_stats, load_merge_standardize_audio_csvs, load_standardized_audio_csv
from app.config import AUDIO_FILE, DEFAULT_LYRICS_PARQUET, DEFAULT_AUDIO_PARQUET
from app.data_loader import load_dataset, save_parquet


def main() -> None:
    p = argparse.ArgumentParser(description="Build spotify_audio_features.parquet from a Kaggle Spotify CSV.")
    p.add_argument(
        "--audio",
        type=Path,
        nargs="*",
        default=None,
        help=(
            f"One or more input CSVs (same columns). Merged in order; duplicate track_id keeps first file. "
            f"Default: single file {AUDIO_FILE}"
        ),
    )
    p.add_argument("--out", type=Path, default=None, help=f"Output parquet (default: {DEFAULT_AUDIO_PARQUET})")
    p.add_argument(
        "--lyrics-parquet",
        type=Path,
        default=None,
        help=f"Optional cleaned lyrics parquet for match-rate stats (default: {DEFAULT_LYRICS_PARQUET} if exists)",
    )
    args = p.parse_args()

    out_path = args.out or Path(DEFAULT_AUDIO_PARQUET)

    if args.audio is None or len(args.audio) == 0:
        audio_df = load_standardized_audio_csv(Path(AUDIO_FILE))
    elif len(args.audio) == 1:
        audio_df = load_standardized_audio_csv(args.audio[0])
    else:
        audio_df = load_merge_standardize_audio_csvs(args.audio)
    save_parquet(audio_df, out_path)
    print(f"Wrote {len(audio_df):,} rows to {out_path}")

    lyrics_pq = args.lyrics_parquet
    if lyrics_pq is None and Path(DEFAULT_LYRICS_PARQUET).exists():
        lyrics_pq = Path(DEFAULT_LYRICS_PARQUET)
    if lyrics_pq and Path(lyrics_pq).exists():
        lyrics_df = load_dataset(lyrics_pq)
        matched, n, rate = audio_lyrics_match_stats(lyrics_df, audio_df)
        print(f"Overlap with lyrics: {matched:,} / {n:,} ({rate:.1f}% of lyrics rows have a matching audio row).")


if __name__ == "__main__":
    main()
