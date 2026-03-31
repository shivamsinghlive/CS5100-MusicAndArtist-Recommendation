# Music Recommendation Project

This project converts your notebook into a modular Python app.

## Included modules
- `app/preprocess.py`: cleans lyrics and builds `cleaned_text`
- `app/tfidf_recommender.py`: baseline TF-IDF recommender
- `app/embedding_recommender.py`: multilingual embedding recommender with FAISS
- `app/hybrid_recommender.py`: lyrics + audio hybrid recommender
- `app/explainability.py`: simple XAI explanation generator
- `app/visualization.py`: optional 3D embedding visualization with UMAP and Plotly
- `app/streamlit_app.py`: Streamlit UI
- `scripts/build_artifacts.py`: prepares parquet files from the raw CSV

## Expected files
Place these under the project root or `data_parquet/`:
- `spotify_millsongdata.csv`
- `data_parquet/spotify_audio_features.parquet` (optional for hybrid mode)

## Install
```bash
pip install -r requirements.txt
```

## Prepare data
```bash
python -m scripts.build_artifacts
```

## Run Streamlit
```bash
streamlit run app/streamlit_app.py
```
