# 🎵 CS5100 Music Recommender

**Cross-lingual · TF-IDF · Multilingual Embeddings · Hybrid Audio+Lyrics · 3D Visualization**

A content-based music recommendation system built for CS 5100 (Foundations of AI). Supports three recommendation modes — TF-IDF lyric similarity, multilingual sentence embeddings (cross-lingual), and a hybrid lyrics + audio features model — with an interactive Streamlit interface featuring semantic search, explainable recommendations, and 3D/2D embedding visualizations.

## App Preview

![App Screenshot](docs/app_screenshot.png)

---

## Project Structure
- `app/preprocess.py`: cleans lyrics and builds `cleaned_text`
- `app/tfidf_recommender.py`: baseline TF-IDF recommender
- `app/embedding_recommender.py`: multilingual embedding recommender with FAISS
- `app/hybrid_recommender.py`: lyrics + audio hybrid recommender
- `app/explainability.py`: simple XAI explanation generator
- `app/visualization.py`: optional 3D embedding visualization with UMAP and Plotly
- `app/streamlit_app.py`: Streamlit UI
- `scripts/build_artifacts.py`: builds `data/processed/cleaned_lyrics.parquet` and (if present) `spotify_audio_features.parquet`
- `scripts/merge_audio_features.py`: rebuilds only the audio-features parquet from a Kaggle CSV (fast; no lyric NLP pass)

## Expected files
Place raw CSVs under **`data/raw/`** (see `app/config.py`):
- `data/raw/spotify_millsongdata.csv` — Million Song lyrics (`artist`, `song`, `text`, …)
- `data/raw/spotify_songs.csv` — optional; Spotify audio-features table (`artists`/`track_name` or `artist`/`song` + `energy`, `valence`, `tempo`, …). Common sources: [Kaggle mrmorj/spotify](https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify) or similar.

Processed outputs go to **`data/processed/`**:
- `cleaned_lyrics.parquet`
- `spotify_audio_features.parquet` (enables hybrid mode in the app)

## Install
```bash
pip install -r requirements.txt
```

## Prepare data
```bash
# Default: reads data/raw/spotify_millsongdata.csv; if data/raw/spotify_songs.csv exists, writes audio parquet too.
python -m scripts.build_artifacts

# Custom locations (e.g. file still in Downloads):
python -m scripts.build_artifacts --lyrics ~/Downloads/spotify_millsongdata.csv --audio ~/Downloads/spotify_songs.csv

# Audio CSV only (after lyrics parquet already exists):
python -m scripts.merge_audio_features --audio ~/Downloads/spotify_songs.csv

# Merge two Spotify chart CSVs (e.g. April2019 + Nov2018), April first so duplicate `track_id` keeps the newer file:
python -m scripts.merge_audio_features --audio \
  ~/Downloads/archive\ \(5\)/SpotifyAudioFeaturesApril2019.csv \
  ~/Downloads/archive\ \(5\)/SpotifyAudioFeaturesNov2018.csv
```

## Run the app (correct entry point)
```bash
streamlit run app/streamlit_app.py
```

> **Note:** `src/data_pipeline.py` and `main.py` are early scaffolding files and are **not runnable** (`load_and_clean_data` raises `NotImplementedError`). All functional code lives under `app/` and `scripts/`.

## Quantitative evaluation (for report Results section)
Run a quick baseline comparison between TF-IDF and Embedding recommenders:
```bash
python -m scripts.evaluate_recommenders --top-k 5 --n-queries 200
```
This writes a metrics table to `results/evaluation_metrics.csv` with:
- `precision@k`
- `recall@k`
- `ndcg@k`

## Notes
- Hybrid mode requires **`data/processed/spotify_audio_features.parquet`** (created by the scripts above). Merge keys are **`artist` + `song`**, matched case-insensitively after stripping whitespace.
- If audio features are missing, the app falls back to lyrics-only hybrid scoring and shows a warning in the UI.
- The `compute_umap_coords` function uses **PCA** (not UMAP) for 3D/2D projection. UMAP caused Numba/Streamlit runtime conflicts; PCA was adopted as a stable alternative. The function name is retained from an earlier iteration.
