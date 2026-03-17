from src.data_pipeline import load_and_clean_data
from src.recommender import get_hybrid_recommendations


def main() -> None:
    # 1) Load data (Member A)
    df = load_and_clean_data("data/processed/combined.parquet")

    # 2) Recommend (Member B)
    results = get_hybrid_recommendations(
        "track_123",
        df,
        top_n=10,
        audio_feature_cols=["tempo", "energy"],
    )
    print(results)


if __name__ == "__main__":
    main()

