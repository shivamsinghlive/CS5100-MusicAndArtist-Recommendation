from __future__ import annotations
import sys
from pathlib import Path
from textwrap import dedent

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA

from app.config import (
    DEFAULT_LYRICS_CSV, DEFAULT_LYRICS_PARQUET, DEFAULT_AUDIO_PARQUET,
    EMBEDDINGS_FILE, MERGED_DATA_FILE,
)
from app.data_loader import load_dataset, merge_lyrics_audio
from app.preprocess import add_cleaned_text
from app.tfidf_recommender import TfidfRecommender
from app.embedding_recommender import EmbeddingRecommender
from app.hybrid_recommender import HybridRecommender, apply_context_filters
from app.explainability import explain_recommendation, explain_tfidf_overlap

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
/* Force dark color scheme regardless of OS/browser preference */
:root { color-scheme: dark; }

.stApp { background-color: #0f0f1a; }
[data-testid="stAppViewContainer"] { background-color: #0f0f1a; color: #e0e0f0; }
[data-testid="stHeader"] { background: #0f0f1a; }
[data-testid="stToolbar"] { background: transparent; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-right: 1px solid #2d2d5e;
}
[data-testid="stSidebar"] * { color: #e0e0f0 !important; }

/* Inputs and controls */
.stTextInput > div > div > input,
.stTextArea textarea,
div[data-baseweb="select"] > div,
div[data-baseweb="select"] input {
    background-color: #1b1b34 !important;
    color: #e8e8ff !important;
    border-color: #3a3a6a !important;
}

/* DataFrame and chart containers */
[data-testid="stDataFrame"],
[data-testid="stPlotlyChart"],
[data-testid="stMetric"] {
    background-color: #131327 !important;
    border: 1px solid #2d2d55;
    border-radius: 10px;
    padding: 0.3rem;
}

/* Metric text readability on dark cards */
[data-testid="stMetric"] label,
[data-testid="stMetric"] [data-testid="stMetricLabel"],
[data-testid="stMetric"] [data-testid="stMetricLabel"] * {
    color: #8f92bd !important;
    opacity: 1 !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"],
[data-testid="stMetric"] [data-testid="stMetricValue"] * {
    color: #e8e8ff !important;
    opacity: 1 !important;
}
[data-testid="stMetric"] [data-testid="stMetricDelta"],
[data-testid="stMetric"] [data-testid="stMetricDelta"] * {
    color: #7dd3fc !important;
    opacity: 1 !important;
}

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

/* ── Nav buttons: dark ghost style ── */
.stButton > button {
    background: rgba(20, 20, 45, 0.7) !important;
    color: #8890cc !important;
    -webkit-text-fill-color: #8890cc !important;
    border: 1px solid #2e2e5a !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    min-height: 2.1rem !important;
    width: 100% !important;
    transition: all 0.15s ease !important;
}
.stButton > button p,
.stButton > button span {
    color: #8890cc !important;
    -webkit-text-fill-color: #8890cc !important;
}
.stButton > button:hover,
.stButton > button:focus {
    background: rgba(102, 126, 234, 0.12) !important;
    border-color: #667eea !important;
    color: #b8c0ff !important;
    -webkit-text-fill-color: #b8c0ff !important;
}
.stButton > button:hover p,
.stButton > button:hover span,
.stButton > button:focus p,
.stButton > button:focus span {
    color: #b8c0ff !important;
    -webkit-text-fill-color: #b8c0ff !important;
}
.stButton > button:disabled {
    opacity: 0.3 !important;
    cursor: not-allowed !important;
}

/* ── Page badge ── */
.page-badge {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 2.1rem;
    background: rgba(102, 126, 234, 0.06);
    border: 1px solid #2e2e5a;
    border-radius: 20px;
    font-size: 0.8rem;
    color: #5a6090;
    letter-spacing: 0.05em;
    font-variant-numeric: tabular-nums;
}

/* 3D tip line — white text on dark banner (avoid st.info default blue) */
.custom-info-banner {
    background: #132043;
    border: 1px solid #2b4ea1;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    margin: 0.25rem 0 0.75rem 0;
    font-size: 0.92rem;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
.custom-info-banner * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
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


def _row_indices_for_songs(df: pd.DataFrame, songs: list[str]) -> set[int]:
    """Map song titles (case-insensitive) to row positions in df (expects default RangeIndex)."""
    out: set[int] = set()
    for s in songs:
        if not s or not str(s).strip():
            continue
        m = df[df["song"].str.lower() == str(s).lower()]
        if not m.empty:
            out.add(int(m.index[0]))
    return out


def _safe_recommend(model, song_name: str, top_n: int, offset: int) -> pd.DataFrame:
    """Call recommend() with backward compatibility for older cached model instances."""
    try:
        return model.recommend(song_name, top_n=top_n, offset=offset)
    except TypeError:
        return model.recommend(song_name, top_n=top_n)


def _safe_semantic_search(model, prompt: str, top_n: int, offset: int) -> pd.DataFrame:
    """Call semantic_search() with backward compatibility for older cached instances."""
    try:
        return model.semantic_search(prompt, top_n=top_n, offset=offset)
    except TypeError:
        return model.semantic_search(prompt, top_n=top_n)


# ── 3D Visualization (cached; PCA is stable in Streamlit reruns) ─────────────
@st.cache_data(show_spinner="🌐 Computing 3D layout (PCA)…")
def compute_umap_coords(
    emb_key: str,
    n_samples: int = 2000,
    seed_song: str = "",
    rec_songs: tuple[str, ...] = (),
) -> pd.DataFrame:
    """
    emb_key is a hash/string so Streamlit knows when to recompute.
    Returns a DataFrame with columns: song, artist, x, y, z, df_idx (row index in embedding table).

    Always tries to include the current seed song and recommended songs in the sample.
    ``df_idx`` is stored so :func:`ensure_songs_in_viz` can transform any still-missing
    songs into the same PCA space as the fitted sample.
    """
    emb_model = get_embedding_model()
    df_meta = emb_model.df
    embeddings = emb_model.embeddings
    n_rows = len(df_meta)
    if n_rows < 2:
        return pd.DataFrame()

    rng = np.random.default_rng(42)
    must_have = _row_indices_for_songs(
        df_meta,
        [seed_song, *list(rec_songs)],
    )
    must_have = {i for i in must_have if 0 <= i < n_rows}

    n = min(n_samples, n_rows)
    pool = list(set(range(n_rows)) - must_have)
    need_extra = max(0, n - len(must_have))
    if need_extra > 0 and pool:
        take = min(need_extra, len(pool))
        extra = rng.choice(pool, take, replace=False)
        idx_set = must_have | set(int(x) for x in np.atleast_1d(extra))
    else:
        idx_set = must_have

    idx = np.array(sorted(idx_set), dtype=np.int64)
    sample_emb = embeddings[idx].astype(np.float32)
    sample_meta = df_meta.iloc[idx].reset_index(drop=True)

    reducer = PCA(n_components=3, random_state=42)
    coords = reducer.fit_transform(sample_emb)

    result = sample_meta[["artist", "song"]].copy()
    result["x"] = coords[:, 0]
    result["y"] = coords[:, 1]
    result["z"] = coords[:, 2]
    result["df_idx"] = idx.astype(np.int64)
    result["is_highlight"] = False
    return result


def ensure_songs_in_viz(viz_df: pd.DataFrame, must_include: list[str]) -> pd.DataFrame:
    """
    If seed / recommended songs are missing from ``viz_df`` (e.g. cache from an older run,
    or name mismatch), project their embeddings with ``PCA.transform`` after fitting on the
    same sample embeddings already shown — so 3D positions are consistent with the plot.
    """
    if viz_df.empty or "df_idx" not in viz_df.columns:
        return viz_df

    emb_model = get_embedding_model()
    df_meta = emb_model.df
    embeddings = emb_model.embeddings
    n_rows = len(df_meta)

    viz_lower = set(viz_df["song"].astype(str).str.lower())
    missing: list[str] = []
    for s in must_include:
        if not s or not str(s).strip():
            continue
        if str(s).lower() not in viz_lower:
            missing.append(str(s))

    if not missing:
        return viz_df

    rows: list[dict] = []
    extra_embs: list[np.ndarray] = []
    for song in missing:
        m = df_meta[df_meta["song"].astype(str).str.lower() == song.lower()]
        if m.empty:
            continue
        ridx = int(m.index[0])
        if ridx < 0 or ridx >= n_rows:
            continue
        rows.append(
            {
                "artist": df_meta.iloc[ridx]["artist"],
                "song": df_meta.iloc[ridx]["song"],
                "df_idx": ridx,
            }
        )
        extra_embs.append(np.asarray(embeddings[ridx], dtype=np.float32))

    if not extra_embs:
        return viz_df

    sample_idx = viz_df["df_idx"].values.astype(np.int64)
    sample_emb = embeddings[sample_idx].astype(np.float32)
    reducer = PCA(n_components=3, random_state=42)
    reducer.fit(sample_emb)

    extra_arr = np.stack(extra_embs, axis=0).astype(np.float32)
    try:
        extra_coords = reducer.transform(extra_arr)
    except Exception as exc:  # pragma: no cover
        st.warning(f"Could not transform extra songs into the same PCA space: {exc}")
        return viz_df

    extra_df = pd.DataFrame(rows)
    extra_df["x"] = extra_coords[:, 0]
    extra_df["y"] = extra_coords[:, 1]
    extra_df["z"] = extra_coords[:, 2]
    extra_df["is_highlight"] = False

    out = pd.concat([viz_df, extra_df], ignore_index=True)
    # One row per embedding row (avoid duplicate df_idx from concat)
    out = out.drop_duplicates(subset=["df_idx"], keep="first")
    return out


def render_3d_plot(
    viz_df: pd.DataFrame,
    seed_song: str,
    rec_songs: list[str],
) -> str | None:
    """
    Plotly 3D scatter. Uses native ``st.plotly_chart`` for rendering — ``streamlit-plotly-events``
    often shows a blank canvas on recent Streamlit versions.

    Optional: point selection via ``on_select="rerun"`` when supported; otherwise use sidebar seed.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.error("Please install plotly: `pip install plotly`")
        return None

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

    if viz_df.empty:
        st.warning("No points available for 3D plot.")
        return None

    # "other" must stay visible on dark plot_bg (#0f0f1a); old rgba(...,0.22) + size 3 looked empty
    style = {
        "other":       dict(color="rgba(140,140,220,0.65)", size=5,  symbol="circle"),
        "recommended": dict(color="#f59e0b",                size=10, symbol="diamond"),
        "seed":        dict(color="#ef4444",                size=14, symbol="circle"),
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
        template="plotly_dark",
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
            aspectmode="data",
            camera=dict(eye=dict(x=1.35, y=1.35, z=0.9)),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(
            text="🌐 Song Embedding Space"
                 "<span style='font-size:11px;color:#889'>  "
                 "🔴 seed  🟡 recommended  🔵 others  · select a point (lasso/box) if available</span>",
            font=dict(size=14, color="#c0c0f0"),
        ),
    )

    # ── Render: native Streamlit Plotly only (plotly_events often yields an empty div) ──
    clicked_song = None
    traces = list(fig.data)

    try:
        event = st.plotly_chart(
            fig,
            width="stretch",
            key="plotly_3d_embedding",
            on_select="rerun",
            selection_mode="points",
        )
    except TypeError:
        # Older Streamlit without on_select on plotly_chart
        st.plotly_chart(fig, width="stretch", key="plotly_3d_embedding")
        event = None

    if event is not None:
        try:
            sel = event.get("selection") if isinstance(event, dict) else getattr(event, "selection", None)
            if sel is not None:
                pts = sel.get("points", []) if isinstance(sel, dict) else getattr(sel, "points", None) or []
                if pts:
                    p0 = pts[0] if isinstance(pts, list) else pts
                    if isinstance(p0, dict):
                        curve_idx = int(p0.get("curve_number", p0.get("curveNumber", 0)))
                        point_idx = int(p0.get("point_index", p0.get("pointIndex", 0)))
                    else:
                        curve_idx = int(
                            getattr(p0, "curve_number", getattr(p0, "curveNumber", 0))
                        )
                        point_idx = int(
                            getattr(p0, "point_index", getattr(p0, "pointIndex", 0))
                        )
                    if curve_idx < len(traces):
                        cd = traces[curve_idx].customdata
                        if cd is not None and point_idx < len(cd):
                            df_idx = cd[point_idx]
                            clicked_song = str(viz_df.loc[int(df_idx), "song"])
        except Exception:
            pass

    st.caption(
        "💡 If the chart is slow to respond, reduce “Points in 3D plot” in the sidebar. "
        "Select a point on the chart (when lasso/box tools appear) to set a new seed, or change the seed in the sidebar."
    )

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

    st.caption("Mode guide: TF-IDF = lexical overlap, Embeddings = semantic retrieval, Hybrid = lyrics + audio features.")
    if mode == "Multilingual Embeddings":
        st.divider()
        semantic_prompt = st.text_input(
            "💬 Semantic search",
            placeholder="e.g. sad rainy day ballad",
            help="Only available in Multilingual Embeddings mode.",
        )
    else:
        st.caption("Semantic search input appears only in Multilingual Embeddings mode.")
        semantic_prompt = ""

    st.divider()
    show_3d = st.toggle("🌐 Show 3D Embedding Space", value=True,
                        help="3D projection uses PCA for stable rendering.")
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

# Reset recommendation pagination whenever query context changes.
rec_context = (
    mode,
    seed_song.strip().lower(),
    semantic_prompt.strip().lower(),
    mood,
    float(min_energy),
    int(top_n),
)
if st.session_state.get("_last_rec_context") != rec_context:
    st.session_state["rec_page"] = 0
    st.session_state["_last_rec_context"] = rec_context

offset = st.session_state.get("rec_page", 0) * top_n

results, error_msg = None, None
tfidf_model = None
hybrid_without_audio = False
emb_model = None
hybrid_model = None
hybrid_df = None

with st.spinner("Finding similar songs…"):
    try:
        if mode == "TF-IDF":
            tfidf_model = get_tfidf_model(filtered_df)
            results = tfidf_model.recommend(seed_song, top_n=top_n, offset=offset)

        elif mode == "Multilingual Embeddings":
            emb_model = get_embedding_model()
            if semantic_prompt.strip():
                results = _safe_semantic_search(emb_model, semantic_prompt.strip(), top_n=top_n, offset=offset)
            else:
                results = _safe_recommend(emb_model, seed_song, top_n=top_n, offset=offset)

        else:
            hybrid_model, hybrid_df = get_hybrid_model(filtered_df)
            if hybrid_model is None:
                hybrid_without_audio = True
                tfidf_model = get_tfidf_model(filtered_df)
                results = tfidf_model.recommend(seed_song, top_n=top_n, offset=offset)
            else:
                results = hybrid_model.recommend(seed_song, top_n=top_n, offset=offset)

    except ValueError as e:
        error_msg = f"❌ {e}"

# ── Results ───────────────────────────────────────────────────────────────────
if error_msg:
    st.error(error_msg)

elif results is not None and not results.empty:
    if hybrid_without_audio:
        st.warning(
            "Hybrid mode is currently using lyrics-only fallback because "
            "`data_parquet/spotify_audio_features.parquet` was not found. "
            "Add audio features to enable true hybrid scoring."
        )

    seed_match = lyrics_df[lyrics_df["song"].str.lower() == seed_song.lower()]
    seed_info = seed_match.iloc[0].to_dict() if not seed_match.empty else {}
    rec_song_names = results["song"].tolist()

    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.markdown(f'<div class="section-header">Recommendations for "{seed_song}"</div>',
                    unsafe_allow_html=True)
        for rank, (_, row) in enumerate(results.iterrows(), start=1):
            song_name = row.get("song", "Unknown")
            artist    = row.get("artist", "Unknown")
            score     = row.get("score", 0.0)
            if mode == "TF-IDF" or hybrid_without_audio:
                reason = (
                    explain_tfidf_overlap(
                        seed_song_name=seed_song,
                        rec_song_name=str(song_name),
                        df=filtered_df.reset_index(drop=True),
                        vectorizer=tfidf_model.vectorizer,
                        matrix=tfidf_model.matrix,
                    )
                    if tfidf_model is not None else ""
                )
            else:
                reason = explain_recommendation(seed_info, row.to_dict()) if seed_info else ""
            st.markdown(
                dedent(
                    f"""
                    <div class="rec-card">
                        <div class="rec-rank">#{offset + rank}</div>
                        <div class="rec-score">{score:.3f}</div>
                        <div class="rec-song">{song_name}</div>
                        <div class="rec-artist">by {artist}</div>
                        {"" if not reason else f'<div class="rec-reason">{reason}</div>'}
                    </div>
                    """
                ).strip(),
                unsafe_allow_html=True,
            )

        if mode == "TF-IDF" or hybrid_without_audio:
            total_candidates = max(0, len(filtered_df) - 1)
        elif mode == "Multilingual Embeddings" and emb_model is not None and emb_model.df is not None:
            total_candidates = max(0, len(emb_model.df) - 1)
        elif hybrid_df is not None:
            total_candidates = max(0, len(hybrid_df) - 1)
        else:
            total_candidates = max(0, len(filtered_df) - 1)

        max_page = (total_candidates - 1) // top_n if total_candidates > 0 else 0
        current_page = st.session_state.get("rec_page", 0)

        st.markdown('<div class="section-header">Explore More Recommendations</div>', unsafe_allow_html=True)
        nav1, nav2, nav3, nav4 = st.columns([1.1, 0.85, 1.1, 1.1])

        with nav1:
            if st.button("← Previous", key="rec_prev_batch",
                         disabled=(current_page == 0),
                         use_container_width=True):
                st.session_state["rec_page"] = current_page - 1
                st.rerun()

        with nav2:
            st.markdown(
                f'<div class="page-badge">{current_page + 1} / {max_page + 1}</div>',
                unsafe_allow_html=True,
            )

        with nav3:
            if st.button("Next →", key="rec_next_batch",
                         disabled=(current_page >= max_page or len(results) < top_n),
                         use_container_width=True):
                st.session_state["rec_page"] = current_page + 1
                st.rerun()

        with nav4:
            if st.button("🔀 Shuffle", key="rec_random_batch",
                         use_container_width=True):
                candidates = [p for p in range(max_page + 1) if p != current_page]
                if candidates:
                    st.session_state["rec_page"] = int(np.random.choice(candidates))
                    st.rerun()

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
        try:
            import plotly.express as px

            fig = px.bar(
                chart_data,
                x="Song",
                y="Score",
                template="plotly_dark",
                color_discrete_sequence=["#667eea"],
            )
            fig.update_layout(
                paper_bgcolor="#0f0f1a",
                plot_bgcolor="#131327",
                font_color="#c0c0f0",
                margin=dict(l=20, r=20, t=10, b=20),
                xaxis_title="",
                yaxis_title="Score",
            )
            st.plotly_chart(fig, width="stretch", key="score_distribution_dark")
        except Exception:
            # Fallback keeps app functional if Plotly import fails unexpectedly.
            st.bar_chart(chart_data.set_index("Song"), color="#667eea")

        if seed_info:
            st.markdown('<div class="section-header">Seed Song</div>', unsafe_allow_html=True)

            artist_name  = seed_info.get("artist", "Unknown")
            artist_count = len(lyrics_df[lyrics_df["artist"] == artist_name])

            # Build optional tags (dedent so Markdown does not treat lines as code blocks)
            tags = dedent(
                f"""
                <span style="background:rgba(102,126,234,0.15);color:#667eea;border-radius:20px;padding:0.2rem 0.7rem;font-size:0.78rem">🎤 {artist_count} songs in dataset</span>
                """
            ).strip()
            if seed_info.get("language"):
                tags += dedent(
                    f"""
                    <span style="background:rgba(102,126,234,0.15);color:#667eea;border-radius:20px;padding:0.2rem 0.7rem;font-size:0.78rem">🌐 {seed_info["language"]}</span>
                    """
                ).strip()
            if seed_info.get("energy"):
                tags += dedent(
                    f"""
                    <span style="background:rgba(245,158,11,0.15);color:#f59e0b;border-radius:20px;padding:0.2rem 0.7rem;font-size:0.78rem">⚡ energy {float(seed_info["energy"]):.2f}</span>
                    """
                ).strip()
            if seed_info.get("valence"):
                tags += dedent(
                    f"""
                    <span style="background:rgba(52,211,153,0.15);color:#34d399;border-radius:20px;padding:0.2rem 0.7rem;font-size:0.78rem">😊 valence {float(seed_info["valence"]):.2f}</span>
                    """
                ).strip()

            st.markdown(
                dedent(
                    f"""
                    <div class="rec-card" style="margin-top:0.5rem">
                        <div class="rec-song">{seed_info.get("song", "—")}</div>
                        <div class="rec-artist">by {artist_name}</div>
                        <div style="margin-top:0.8rem;display:flex;gap:0.6rem;flex-wrap:wrap;align-items:center">
                            {tags}
                        </div>
                    </div>
                    """
                ).strip(),
                unsafe_allow_html=True,
            )

    # ── 3D Visualization ──────────────────────────────────────────────────────
    if show_3d:
        if mode != "Multilingual Embeddings":
            st.markdown(
                '<div class="custom-info-banner">💡 3D visualization uses multilingual embeddings. '
                "Switch mode to 'Multilingual Embeddings' for best results.</div>",
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-header">🌐 3D Embedding Space</div>', unsafe_allow_html=True)
        st.caption(
            "Each point is a song projected into 3D via PCA. "
            "🔴 = seed song · 🟡 = recommended · 🔵 = others. "
            "Seed and recommendations are kept in the same PCA space (fit on the sample, "
            "then transform for any extra titles). Nearby songs share similar lyric semantics."
        )

        # Use embedding model regardless of current mode for the visualization
        try:
            emb_model = get_embedding_model()
            rec_key = "|".join(sorted(str(s) for s in rec_song_names))
            viz_df = compute_umap_coords(
                emb_key=f"umap_{len(emb_model.df)}_{viz_samples}_{seed_song}_{rec_key}",
                n_samples=viz_samples,
                seed_song=seed_song,
                rec_songs=tuple(rec_song_names),
            )
            if not viz_df.empty:
                must_3d = [seed_song, *rec_song_names]
                viz_df = ensure_songs_in_viz(viz_df, must_3d)
                clicked_song = render_3d_plot(viz_df, seed_song, rec_song_names)
                if clicked_song and clicked_song.lower() != seed_song.lower():
                    st.info(f"🎯 Clicked: **{clicked_song}** — updating seed song and re-running…")
                    st.session_state["seed_override"] = clicked_song
                    st.rerun()
            else:
                st.warning(
                    "3D layout is empty. Check that embedding files load correctly."
                )
        except Exception as e:
            st.error(f"3D visualization failed: {e}")

elif results is not None and results.empty:
    st.warning("No results found. Try a different song name.")