from __future__ import annotations
import sys
from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
import pandas as pd
import streamlit as st

from app.config import DEFAULT_LYRICS_CSV, DEFAULT_LYRICS_PARQUET, DEFAULT_AUDIO_PARQUET
from app.data_loader import load_dataset, merge_lyrics_audio
from app.preprocess import add_cleaned_text
from app.tfidf_recommender import TfidfRecommender
from app.embedding_recommender import EmbeddingRecommender
from app.hybrid_recommender import HybridRecommender, apply_context_filters
from app.explainability import explain_recommendation

st.set_page_config(page_title="Music Recommendation System", layout="wide")
st.title("Hybrid Cross-lingual Music Recommendation System")

@st.cache_data
def load_lyrics() -> pd.DataFrame:
    if Path(DEFAULT_LYRICS_PARQUET).exists():
        return load_dataset(DEFAULT_LYRICS_PARQUET)
    if Path(DEFAULT_LYRICS_CSV).exists():
        df = load_dataset(DEFAULT_LYRICS_CSV)
        if "link" in df.columns:
            df = df.drop(columns=["link"])
        return add_cleaned_text(df)
    raise FileNotFoundError("No lyrics dataset found. Add spotify_millsongdata.csv or lyrics_clean.parquet.")


lyrics_df = load_lyrics()
base_df = lyrics_df.copy()

st.sidebar.header("Controls")
mode = st.sidebar.selectbox("Recommendation Mode", ["TF-IDF", "Multilingual Embeddings", "Hybrid"])
seed_song = st.sidebar.text_input("Seed song", value=str(base_df["song"].iloc[0] if not base_df.empty else ""))
top_n = st.sidebar.slider("Top recommendations", 3, 10, 5)
mood = st.sidebar.selectbox("Mood filter", ["None", "Gym"])
min_energy = st.sidebar.slider("Minimum energy", 0.0, 1.0, 0.0, 0.05)
semantic_prompt = st.sidebar.text_input("Semantic search prompt", value="")

filtered_df = apply_context_filters(base_df, mood=None if mood == "None" else mood, min_energy=min_energy if min_energy > 0 else None)

if mode == "TF-IDF":
    model = TfidfRecommender().fit(filtered_df)
    results = model.recommend(seed_song, top_n=top_n)
elif mode == "Multilingual Embeddings":
    model = EmbeddingRecommender().fit(filtered_df)
    results = model.semantic_search(semantic_prompt, top_n=top_n) if semantic_prompt.strip() else model.recommend(seed_song, top_n=top_n)
else:
    if Path(DEFAULT_AUDIO_PARQUET).exists():
        audio_df = load_dataset(DEFAULT_AUDIO_PARQUET)
        hybrid_df = merge_lyrics_audio(filtered_df, audio_df)
    else:
        st.warning("Audio features parquet not found. Hybrid mode is using the lyrics dataframe only where possible.")
        hybrid_df = filtered_df.copy()
        for col in ["tempo", "energy", "valence"]:
            if col not in hybrid_df.columns:
                hybrid_df[col] = 0.0
    model = HybridRecommender().fit(hybrid_df)
    results = model.recommend(seed_song, top_n=top_n)

st.subheader("Recommendations")
st.dataframe(results, use_container_width=True)

matches = filtered_df[filtered_df["song"].str.lower() == seed_song.lower()]
if not matches.empty and not results.empty:
    st.subheader("Why these songs?")
    seed = matches.iloc[0].to_dict()
    for _, row in results.iterrows():
        st.write(f"**{row.get('song', 'Unknown')}** by {row.get('artist', 'Unknown')}")
        st.write(explain_recommendation(seed, row.to_dict()))
