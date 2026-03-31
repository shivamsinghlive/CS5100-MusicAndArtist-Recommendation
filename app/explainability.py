from __future__ import annotations


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
