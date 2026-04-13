from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd

from app.config import DEFAULT_LYRICS_CSV, DEFAULT_LYRICS_PARQUET
from app.data_loader import load_dataset
from app.preprocess import add_cleaned_text
from app.tfidf_recommender import TfidfRecommender
from app.embedding_recommender import EmbeddingRecommender


def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = recommended[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for song in top_k if song in relevant)
    return hits / float(k)


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for song in top_k if song in relevant)
    return hits / float(len(relevant))


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    import math

    top_k = recommended[:k]
    if not top_k:
        return 0.0
    dcg = 0.0
    for i, song in enumerate(top_k, start=1):
        rel = 1.0 if song in relevant else 0.0
        dcg += rel / math.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def load_lyrics() -> pd.DataFrame:
    if Path(DEFAULT_LYRICS_PARQUET).exists():
        return load_dataset(DEFAULT_LYRICS_PARQUET)
    if Path(DEFAULT_LYRICS_CSV).exists():
        df = load_dataset(DEFAULT_LYRICS_CSV)
        if "link" in df.columns:
            df = df.drop(columns=["link"])
        return add_cleaned_text(df)
    raise FileNotFoundError("No lyrics dataset found.")


def evaluate(df: pd.DataFrame, top_k: int, n_queries: int) -> pd.DataFrame:
    if "artist" not in df.columns or "song" not in df.columns:
        raise ValueError("Dataset must include `artist` and `song` columns.")
    if "cleaned_text" not in df.columns:
        raise ValueError("Dataset must include `cleaned_text` column.")

    df = df.dropna(subset=["artist", "song"]).reset_index(drop=True)
    artist_counts = df["artist"].value_counts()
    candidate_artists = artist_counts[artist_counts >= 2].index
    eval_df = df[df["artist"].isin(candidate_artists)].copy()
    if eval_df.empty:
        raise ValueError("Not enough artists with >=2 songs for evaluation.")

    tfidf = TfidfRecommender().fit(eval_df)
    emb = EmbeddingRecommender().fit(eval_df)

    queries = eval_df.sample(min(n_queries, len(eval_df)), random_state=42)["song"].tolist()
    scores = {
        "TF-IDF": {"p": [], "r": [], "n": []},
        "Embedding": {"p": [], "r": [], "n": []},
    }

    for query_song in queries:
        query_rows = eval_df[eval_df["song"].str.lower() == str(query_song).lower()]
        if query_rows.empty:
            continue
        query_artist = str(query_rows.iloc[0]["artist"])
        relevant = set(
            eval_df[
                (eval_df["artist"] == query_artist)
                & (eval_df["song"].str.lower() != str(query_song).lower())
            ]["song"].tolist()
        )
        if not relevant:
            continue

        tfidf_rec = tfidf.recommend(query_song, top_n=top_k)["song"].tolist()
        emb_rec = emb.recommend(query_song, top_n=top_k)["song"].tolist()

        for name, rec_list in [("TF-IDF", tfidf_rec), ("Embedding", emb_rec)]:
            scores[name]["p"].append(precision_at_k(rec_list, relevant, top_k))
            scores[name]["r"].append(recall_at_k(rec_list, relevant, top_k))
            scores[name]["n"].append(ndcg_at_k(rec_list, relevant, top_k))

    rows = []
    for model_name, vals in scores.items():
        if vals["p"]:
            rows.append(
                {
                    "model": model_name,
                    "precision@k": sum(vals["p"]) / len(vals["p"]),
                    "recall@k": sum(vals["r"]) / len(vals["r"]),
                    "ndcg@k": sum(vals["n"]) / len(vals["n"]),
                    "queries_evaluated": len(vals["p"]),
                }
            )
    return pd.DataFrame(rows).sort_values("ndcg@k", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TF-IDF vs Embedding recommenders.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K recommendations to evaluate.")
    parser.add_argument("--n-queries", type=int, default=200, help="Number of seed songs to sample.")
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation_metrics.csv",
        help="CSV output path for metrics table.",
    )
    args = parser.parse_args()

    df = load_lyrics()
    metrics = evaluate(df, top_k=args.top_k, n_queries=args.n_queries)
    if metrics.empty:
        raise RuntimeError("Evaluation produced no metrics. Check dataset columns and content.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)
    print(metrics.to_string(index=False))
    print(f"\nSaved evaluation metrics to: {output_path}")


if __name__ == "__main__":
    main()
