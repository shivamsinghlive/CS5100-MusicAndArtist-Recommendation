from __future__ import annotations

import argparse
from pathlib import Path

from app.audio_features import audio_lyrics_match_stats, load_standardized_audio_csv
from app.config import AUDIO_FILE, DEFAULT_LYRICS_CSV, DEFAULT_LYRICS_PARQUET, DEFAULT_AUDIO_PARQUET
from app.data_loader import load_dataset, merge_lyrics_audio, save_parquet
from app.preprocess import add_cleaned_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build cleaned_lyrics.parquet from Million Song CSV; optionally build spotify_audio_features.parquet from Kaggle audio CSV.",
    )
    parser.add_argument(
        "--lyrics",
        type=Path,
        default=None,
        help=f"Path to spotify_millsongdata.csv (default: {DEFAULT_LYRICS_CSV})",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help=f"Path to Kaggle Spotify audio-features CSV (default: {AUDIO_FILE})",
    )
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Only build cleaned lyrics parquet; do not read audio CSV.",
    )
    args = parser.parse_args()

    lyrics_path = args.lyrics or Path(DEFAULT_LYRICS_CSV)
    if not lyrics_path.exists():
        raise FileNotFoundError(
            f"Lyrics CSV not found: {lyrics_path}\n"
            f"Copy spotify_millsongdata.csv there, e.g.:\n"
            f"  cp ~/Downloads/spotify_millsongdata.csv {DEFAULT_LYRICS_CSV.parent}/"
        )

    df = load_dataset(lyrics_path)
    if "link" in df.columns:
        df = df.drop(columns=["link"])
    df = add_cleaned_text(df)
    save_parquet(df, DEFAULT_LYRICS_PARQUET)
    print(f"Saved cleaned lyrics to {DEFAULT_LYRICS_PARQUET}")

    if args.skip_audio:
        print("Skipping audio (--skip-audio).")
        return

    audio_csv = args.audio or Path(AUDIO_FILE)
    if not audio_csv.exists():
        print(
            f"No audio CSV at {audio_csv}. Hybrid mode will stay in lyrics-only fallback until you add it.\n"
            "Suggested: download a Kaggle dataset with artists + track_name + energy/valence/tempo, save as:\n"
            f"  {AUDIO_FILE}\n"
            "Then re-run: python -m scripts.build_artifacts"
        )
        return

    audio_df = load_standardized_audio_csv(audio_csv)
    need_hybrid = {"energy", "valence", "tempo"}
    missing = sorted(need_hybrid - set(audio_df.columns))
    if missing:
        print(
            f"Warning: columns {missing} not in audio table; HybridRecommender defaults expect "
            "tempo, energy, valence. Use a fuller Spotify-features CSV or adjust audio_feature_cols."
        )
    save_parquet(audio_df, DEFAULT_AUDIO_PARQUET)
    print(f"Saved audio features for {len(audio_df):,} unique tracks to {DEFAULT_AUDIO_PARQUET}")

    matched, n, rate = audio_lyrics_match_stats(df, audio_df)
    print(
        f"Match vs lyrics (normalized artist+song): {matched:,} / {n:,} rows ({rate:.1f}% of lyrics rows have audio)."
    )

    hybrid_out = DEFAULT_AUDIO_PARQUET.parent / "hybrid_merged.parquet"
    merged = merge_lyrics_audio(df, audio_df)
    save_parquet(merged, hybrid_out)
    print(f"Saved inner-join preview to {hybrid_out} ({len(merged):,} rows).")


if __name__ == "__main__":
    main()
