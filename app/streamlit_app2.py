from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from app.config import (
    DEFAULT_LYRICS_CSV, DEFAULT_LYRICS_PARQUET, DEFAULT_AUDIO_PARQUET,
    EMBEDDINGS_FILE, MERGED_DATA_FILE,
)
from app.data_loader import load_dataset, merge_lyrics_audio
from app.preprocess import add_cleaned_text
from app.tfidf_recommender import TfidfRecommender
from app.embedding_recommender import EmbeddingRecommender
from app.hybrid_recommender import HybridRecommender, apply_context_filters
from app.explainability import explain_recommendation

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎵 Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0f0f1a; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-right: 1px solid #2d2d5e;
}
[data-testid="stSidebar"] * { color: #e0e0f0 !important; }

.main-title {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.4rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}
.subtitle { color: #888; font-size: 0.95rem; margin-bottom: 1.5rem; }

.rec-card {
    background: linear-gradient(135deg, #1e1e3a 0%, #252545 100%);
    border: 1px solid #3a3a6a;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
}
.rec-card:hover { border-color: #667eea; }
.rec-rank { font-size: 0.75rem; color: #667eea; font-weight: 600; text-transform: uppercase; }
.rec-song { font-size: 1.1rem; font-weight: 700; color: #e8e8ff; margin: 0.15rem 0; }
.rec-artist { font-size: 0.88rem; color: #9090bb; }
.rec-score {
    float: right;
    background: rgba(102,126,234,0.15);
    color: #667eea;
    border-radius: 20px;
    padding: 0.15rem 0.6rem;
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: -2.2rem;
}
.rec-reason {
    margin-top: 0.5rem; font-size: 0.82rem; color: #7070a0;
    font-style: italic; border-top: 1px solid #2d2d55; padding-top: 0.4rem;
}
.section-header {
    font-size: 1.2rem; font-weight: 700; color: #c0c0f0;
    border-left: 3px solid #667eea; padding-left: 0.6rem; margin: 1.2rem 0 0.8rem;
}
</style>
""", unsafe_allow_html=True)


# ── Data & model loading (cached) ─────────────────────────────────────────────
@st.cache_data(show_spinner="📀 Loading lyrics…")
def load_lyrics() -> pd.DataFrame:
    if Path(DEFAULT_LYRICS_PARQUET).exists():
        return load_dataset(DEFAULT_LYRICS_PARQUET)
    if Path(DEFAULT_LYRICS_CSV).exists():
        df = load_dataset(DEFAULT_LYRICS_CSV)
        if "link" in df.columns:
            df = df.drop(columns=["link"])
        return add_cleaned_text(df)
    raise FileNotFoundError("No lyrics dataset found.")


@st.cache_resource(show_spinner="🤖 Building TF-IDF model…")
def get_tfidf_model(df: pd.DataFrame) -> TfidfRecommender:
    return TfidfRecommender().fit(df)


@st.cache_resource(show_spinner="🧠 Loading embedding model…")
def get_embedding_model() -> EmbeddingRecommender:
    model = EmbeddingRecommender()
    if Path(EMBEDDINGS_FILE).exists() and Path(MERGED_DATA_FILE).exists():
        model.df = pd.read_parquet(MERGED_DATA_FILE)
        model.embeddings = np.load(EMBEDDINGS_FILE)
        model._build_index()
        return model
    lyrics_df = load_lyrics()
    model.fit(lyrics_df)
    model.save(EMBEDDINGS_FILE, MERGED_DATA_FILE)
    return model


@st.cache_resource(show_spinner="🎛️ Building hybrid model…")
def get_hybrid_model(df: pd.DataFrame):
    if not Path(DEFAULT_AUDIO_PARQUET).exists():
        return None, None
    audio_df = load_dataset(DEFAULT_AUDIO_PARQUET)
    hybrid_df = merge_lyrics_audio(df, audio_df)
    return HybridRecommender().fit(hybrid_df), hybrid_df


# ── 3D Visualization (cached because UMAP is slow) ───────────────────────────
@st.cache_data(show_spinner="🌐 Computing 3D layout (UMAP)… this may take ~1 min the first time")
def compute_umap_coords(emb_key: str, n_samples: int = 2000) -> pd.DataFrame:
    """
    emb_key is a hash/string so Streamlit knows when to recompute.
    Returns a DataFrame with columns: song, artist, x, y, z
    """
    try:
        import umap as umap_lib
    except ImportError:
        st.error("Please install umap-learn: `pip install umap-learn`")
        return pd.DataFrame()

    emb_model = get_embedding_model()
    df_meta = emb_model.df
    embeddings = emb_model.embeddings

    # Sample for performance
    n = min(n_samples, len(df_meta))
    idx = np.random.default_rng(42).choice(len(df_meta), n, replace=False)
    sample_emb = embeddings[idx].astype(np.float32)
    sample_meta = df_meta.iloc[idx].reset_index(drop=True)

    reducer = umap_lib.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(sample_emb)

    result = sample_meta[["artist", "song"]].copy()
    result["x"] = coords[:, 0]
    result["y"] = coords[:, 1]
    result["z"] = coords[:, 2]
    result["is_highlight"] = False
    return result


def render_3d_plot(
    viz_df: pd.DataFrame,
    seed_song: str,
    rec_songs: list[str],
) -> str | None:
    """
    Plotly 3D scatter with click-to-recommend.
    Returns the clicked song name, or None if nothing was clicked.
    Requires: pip install streamlit-plotly-events
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.error("Please install plotly: `pip install plotly`")
        return None

    # Try to import plotly_events for click interaction
    try:
        from streamlit_plotly_events import plotly_events
        has_plotly_events = True
    except ImportError:
        has_plotly_events = False

    seed_lower = seed_song.lower()
    rec_lower = {s.lower() for s in rec_songs}

    def point_type(row):
        s = row["song"].lower()
        if s == seed_lower:   return "seed"
        if s in rec_lower:    return "recommended"
        return "other"

    viz_df = viz_df.copy()
    viz_df["type"] = viz_df.apply(point_type, axis=1)
    # Store original index so we can map click → song name
    viz_df = viz_df.reset_index(drop=True)

    style = {
        "other":       dict(color="rgba(100,100,180,0.22)", size=3,  symbol="circle"),
        "recommended": dict(color="#f59e0b",                size=10, symbol="diamond"),
        "seed":        dict(color="#ef4444",                size=14, symbol="cross"),
    }
    label = {"other": "Song", "recommended": "Recommended", "seed": "Seed"}

    fig = go.Figure()
    # We keep a single customdata array aligned to all points for click lookup
    all_indices: list[int] = []

    for ptype, sty in style.items():
        mask = viz_df["type"] == ptype
        sub = viz_df[mask]
        if sub.empty:
            continue
        all_indices.extend(sub.index.tolist())
        fig.add_trace(go.Scatter3d(
            x=sub["x"], y=sub["y"], z=sub["z"],
            mode="markers",
            name=label[ptype],
            marker=dict(
                size=sty["size"],
                color=sty["color"],
                symbol=sty["symbol"],
                line=dict(width=0),
            ),
            # customdata carries the DataFrame index → lets us look up song name on click
            customdata=sub.index.tolist(),
            text=sub["song"] + "<br>" + sub["artist"],
            hovertemplate="<b>%{text}</b><br><i>Click to recommend</i><extra></extra>",
            showlegend=True,
        ))

    fig.update_layout(
        height=620,
        paper_bgcolor="#0f0f1a",
        plot_bgcolor="#0f0f1a",
        font=dict(color="#c0c0f0"),
        legend=dict(
            bgcolor="rgba(30,30,60,0.85)",
            bordercolor="#3a3a6a",
            x=0.01, y=0.99,
        ),
        scene=dict(
            bgcolor="#0f0f1a",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(
            text="🌐 Song Embedding Space"
                 "<span style='font-size:11px;color:#555'>  "
                 "🔴 seed  🟡 recommended  🔵 others  · click any point to use it as seed</span>",
            font=dict(size=14, color="#c0c0f0"),
        ),
    )

    # ── Render with or without click support ──────────────────────────────────
    clicked_song = None

    if has_plotly_events:
        clicked = plotly_events(
            fig,
            click_event=True,
            hover_event=False,
            select_event=False,
            override_height=630,
            key="3d_plot",
        )
        if clicked:
            # clicked[0] = {"curveNumber": int, "pointIndex": int, "x":…, …}
            pt = clicked[0]
            curve_idx = pt.get("curveNumber", 0)
            point_idx = pt.get("pointIndex", 0)
            # Retrieve the DataFrame index stored in customdata
            traces = [t for t in fig.data]
            if curve_idx < len(traces):
                cd = traces[curve_idx].customdata
                if cd is not None and point_idx < len(cd):
                    df_idx = cd[point_idx]
                    clicked_song = viz_df.loc[df_idx, "song"]
    else:
        # Fallback: plain plotly chart, no click
        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 Install `streamlit-plotly-events` to enable click-to-recommend: "
                   "`pip install streamlit-plotly-events`")

    return clicked_song


# ── Sidebar ───────────────────────────────────────────────────────────────────
lyrics_df = load_lyrics()

with st.sidebar:
    st.markdown("## 🎛️ Controls")
    st.divider()

    mode = st.selectbox(
        "Recommendation Mode",
        ["TF-IDF", "Multilingual Embeddings", "Hybrid"],
        help="TF-IDF: keyword | Embeddings: semantic | Hybrid: lyrics + audio",
    )
    _default_seed = st.session_state.pop("seed_override", None) \
        or (str(lyrics_df["song"].iloc[0]) if not lyrics_df.empty else "")
    seed_song = st.text_input(
        "🎵 Seed song",
        value=_default_seed,
        placeholder="e.g. Imagine",
    )
    top_n = st.slider("Top N recommendations", 3, 10, 5)

    st.divider()
    st.markdown("**Filters**")
    mood = st.selectbox("Mood", ["None", "Gym", "Calm"])
    min_energy = st.slider("Min energy", 0.0, 1.0, 0.0, 0.05)

    if mode == "Multilingual Embeddings":
        st.divider()
        semantic_prompt = st.text_input(
            "💬 Semantic search",
            placeholder="e.g. sad rainy day ballad",
        )
    else:
        semantic_prompt = ""

    st.divider()
    show_3d = st.toggle("🌐 Show 3D Embedding Space", value=False,
                        help="Requires Multilingual Embeddings mode + umap-learn")
    if show_3d:
        viz_samples = st.slider("Points in 3D plot", 500, 3000, 1500, 500)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎵 Music Recommender</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Hybrid Cross-lingual · TF-IDF · Multilingual Embeddings · 3D Visualization</div>',
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
c1.metric("Songs", f"{len(lyrics_df):,}")
c2.metric("Artists", f"{lyrics_df['artist'].nunique():,}")
c3.metric("Mode", mode)
st.divider()

# ── Recommendation logic ──────────────────────────────────────────────────────
filtered_df = apply_context_filters(
    lyrics_df,
    mood=None if mood == "None" else mood.lower(),
    min_energy=min_energy if min_energy > 0 else None,
)

results, error_msg = None, None

with st.spinner("Finding similar songs…"):
    try:
        if mode == "TF-IDF":
            results = get_tfidf_model(filtered_df).recommend(seed_song, top_n=top_n)

        elif mode == "Multilingual Embeddings":
            emb_model = get_embedding_model()
            if semantic_prompt.strip():
                results = emb_model.semantic_search(semantic_prompt.strip(), top_n=top_n)
            else:
                results = emb_model.recommend(seed_song, top_n=top_n)

        else:
            hybrid_model, hybrid_df = get_hybrid_model(filtered_df)
            if hybrid_model is None:
                error_msg = "⚠️ Audio features parquet not found."
            else:
                results = hybrid_model.recommend(seed_song, top_n=top_n)

    except ValueError as e:
        error_msg = f"❌ {e}"

# ── Results ───────────────────────────────────────────────────────────────────
if error_msg:
    st.error(error_msg)

elif results is not None and not results.empty:
    seed_match = lyrics_df[lyrics_df["song"].str.lower() == seed_song.lower()]
    seed_info = seed_match.iloc[0].to_dict() if not seed_match.empty else {}
    rec_song_names = results["song"].tolist()

    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.markdown(f'<div class="section-header">Recommendations for "{seed_song}"</div>',
                    unsafe_allow_html=True)
        for i, row in results.iterrows():
            song_name = row.get("song", "Unknown")
            artist    = row.get("artist", "Unknown")
            score     = row.get("score", 0.0)
            reason    = explain_recommendation(seed_info, row.to_dict()) if seed_info else ""
            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-rank">#{i+1}</div>
                <div class="rec-score">{score:.3f}</div>
                <div class="rec-song">{song_name}</div>
                <div class="rec-artist">by {artist}</div>
                {"" if not reason else f'<div class="rec-reason">{reason}</div>'}
            </div>
            """, unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="section-header">Score Distribution</div>', unsafe_allow_html=True)
        chart_data = pd.DataFrame({
            "Song": [
                (str(r.get("song", "?"))[:22] + "…")
                if len(str(r.get("song", ""))) > 22 else r.get("song", "?")
                for _, r in results.iterrows()
            ],
            "Score": [r.get("score", 0.0) for _, r in results.iterrows()],
        })
        st.bar_chart(chart_data.set_index("Song"), color="#667eea")

        if seed_info:
            st.markdown('<div class="section-header">Seed Song Info</div>', unsafe_allow_html=True)
            display_fields = {k: v for k, v in seed_info.items()
                              if k in ("artist", "song", "language", "energy", "valence", "tempo")}
            if display_fields:
                st.json(display_fields)

    # ── 3D Visualization ──────────────────────────────────────────────────────
    if show_3d:
        if mode != "Multilingual Embeddings":
            st.info("💡 3D visualization uses multilingual embeddings. Switch mode to 'Multilingual Embeddings' for best results.")

        st.markdown('<div class="section-header">🌐 3D Embedding Space</div>', unsafe_allow_html=True)
        st.caption(
            "Each point is a song projected into 3D via UMAP. "
            "🔴 = seed song · 🟡 = recommended · 🔵 = others. "
            "Nearby songs share similar lyric semantics."
        )

        # Use embedding model regardless of current mode for the visualization
        try:
            emb_model = get_embedding_model()
            viz_df = compute_umap_coords(
                emb_key=f"umap_{len(emb_model.df)}_{viz_samples}",
                n_samples=viz_samples,
            )
            if not viz_df.empty:
                clicked_song = render_3d_plot(viz_df, seed_song, rec_song_names)
                if clicked_song and clicked_song.lower() != seed_song.lower():
                    st.info(f"🎯 Clicked: **{clicked_song}** — updating seed song and re-running…")
                    st.session_state["seed_override"] = clicked_song
                    st.rerun()
        except Exception as e:
            st.error(f"3D visualization failed: {e}")

elif results is not None and results.empty:
    st.warning("No results found. Try a different song name.")