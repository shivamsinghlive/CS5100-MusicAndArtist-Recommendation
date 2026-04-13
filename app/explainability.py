from __future__ import annotations

import numpy as np
import pandas as pd


def explain_recommendation(seed_song: dict, rec_song: dict) -> str:
    reasons: list[str] = []
    for feature in ["energy", "danceability", "valence", "tempo"]:
        if feature in seed_song and feature in rec_song:
            try:
                if abs(float(seed_song[feature]) - float(rec_song[feature])) < 0.1:
                    reasons.append(f"similar {feature}")
            except Exception:
                pass
    if seed_song.get("artist") == rec_song.get("artist"):
        reasons.append("the same artist")
    if seed_song.get("language") and seed_song.get("language") == rec_song.get("language"):
        reasons.append("the same language")
    if not reasons:
        return "Recommended due to overall similarity across lyrics and available audio features."
    if len(reasons) == 1:
        return f"Recommended because it has {reasons[0]}."
    return "Recommended because it has " + ", ".join(reasons[:-1]) + f", and {reasons[-1]}."


def explain_tfidf_overlap(
    seed_song_name: str,
    rec_song_name: str,
    df: pd.DataFrame,
    vectorizer,
    matrix,
    top_k: int = 6,
) -> str:
    """Explain TF-IDF recommendations using overlapping high-weight lyric terms."""
    if not seed_song_name or not rec_song_name or df.empty:
        return "Recommended due to lyrical similarity."

    seed_match = df[df["song"].astype(str).str.lower() == seed_song_name.lower()]
    rec_match = df[df["song"].astype(str).str.lower() == rec_song_name.lower()]
    if seed_match.empty or rec_match.empty:
        return "Recommended due to lyrical similarity."

    seed_idx = int(seed_match.index[0])
    rec_idx = int(rec_match.index[0])

    seed_vec = matrix[seed_idx].toarray().ravel()
    rec_vec = matrix[rec_idx].toarray().ravel()
    overlap = seed_vec * rec_vec
    if not np.any(overlap > 0):
        return "Recommended due to lyrical similarity."

    top_idx = overlap.argsort()[::-1]
    top_idx = [i for i in top_idx if overlap[i] > 0][:top_k]
    if not top_idx:
        return "Recommended due to lyrical similarity."

    try:
        terms = vectorizer.get_feature_names_out()
    except Exception:
        return "Recommended due to lyrical similarity."

    keywords = [str(terms[i]) for i in top_idx]
    return f"Shared lyric themes: {', '.join(keywords)}."
