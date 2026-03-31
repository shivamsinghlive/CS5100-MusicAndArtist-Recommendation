from __future__ import annotations

from pathlib import Path
from app.config import DEFAULT_LYRICS_CSV, DEFAULT_LYRICS_PARQUET, DEFAULT_AUDIO_PARQUET
from app.data_loader import load_dataset, save_parquet, merge_lyrics_audio
from app.preprocess import add_cleaned_text


def main() -> None:
    base_path = Path(DEFAULT_LYRICS_CSV)
    if not base_path.exists():
        raise FileNotFoundError(f"Place spotify_millsongdata.csv at {base_path}")

    df = load_dataset(base_path)
    if "link" in df.columns:
        df = df.drop(columns=["link"])
    df = add_cleaned_text(df)
    save_parquet(df, DEFAULT_LYRICS_PARQUET)
    print(f"Saved cleaned lyrics to {DEFAULT_LYRICS_PARQUET}")

    audio_path = Path(DEFAULT_AUDIO_PARQUET)
    if audio_path.exists():
        audio_df = load_dataset(audio_path)
        merged = merge_lyrics_audio(df, audio_df)
        output_path = DEFAULT_AUDIO_PARQUET.parent / "hybrid_merged.parquet"
        save_parquet(merged, output_path)
        print(f"Saved hybrid merged file to {output_path}")
    else:
        print(f"Audio parquet not found at {audio_path}. Skipping merge.")


if __name__ == "__main__":
    main()
