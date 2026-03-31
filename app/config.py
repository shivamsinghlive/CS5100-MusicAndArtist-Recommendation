from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Raw datasets
SONGS_FILE = RAW_DATA_DIR / "spotify_millsongdata.csv"
AUDIO_FILE = RAW_DATA_DIR / "spotify_songs.csv"
UNIVERSAL_TOP_FILE = RAW_DATA_DIR / "universal_top_spotify_songs.csv"

# Backward-compatible names used elsewhere
DEFAULT_LYRICS_CSV = SONGS_FILE
DEFAULT_LYRICS_PARQUET = PROCESSED_DATA_DIR / "cleaned_lyrics.parquet"
DEFAULT_AUDIO_PARQUET = PROCESSED_DATA_DIR / "spotify_audio_features.parquet"

# Model / retrieval settings
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TFIDF_MAX_FEATURES = 5000
TOP_K = 10

# Artifact paths
EMBEDDINGS_FILE = PROCESSED_DATA_DIR / "multilingual_embeddings.npy"
FAISS_INDEX_FILE = PROCESSED_DATA_DIR / "faiss.index"
MERGED_DATA_FILE = PROCESSED_DATA_DIR / "merged_dataset.parquet"