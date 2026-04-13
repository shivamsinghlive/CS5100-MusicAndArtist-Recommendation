# CS5100 Music and Artist Recommendation

This repository contains a modular music recommendation project built from a notebook workflow and split into reusable Python modules. The current `main` branch is centered around:

- a notebook for exploration and model prototyping
- a `src/` package that defines the data pipeline contract and a hybrid recommender implementation
- a small `main.py` entry point showing the intended end-to-end flow

## Project goal

The project recommends songs using lyric-based similarity, with optional audio features added for a hybrid recommendation setup.

At a high level, the system is designed to:

1. load and clean a combined music dataset
2. create text features from lyrics
3. optionally combine those text features with normalized audio features
4. return the most similar songs for a given seed track

## Repository structure

```text
.
|-- src/
|   |-- data_pipeline.py     # Data loading / cleaning contract
|   |-- recommender.py       # Hybrid recommender implementation
|   `-- __init__.py
|-- notebooks/
|   `-- music_recommendation_system.ipynb   # Exploration and prototype workflow
|-- data/
|   `-- .gitkeep
|-- main.py                  # Example top-level usage
|-- requirements.txt
`-- LICENSE
```

## Main flow

The intended application flow is defined by `main.py`:

1. `load_and_clean_data()` loads a prepared dataset
2. `get_hybrid_recommendations()` computes the top similar songs
3. the ranked recommendations are printed as a dataframe

Current example:

```python
df = load_and_clean_data("data/processed/combined.parquet")

results = get_hybrid_recommendations(
    "track_123",
    df,
    top_n=10,
    audio_feature_cols=["tempo", "energy"],
)
```

This makes the architecture clear even though the data pipeline module is still a scaffold.

## How the recommender works

The recommendation logic lives in `src/recommender.py`.

### Core idea

Each track is represented using:

- lyric features from `cleaned_text` using TF-IDF
- optional numeric audio features such as `tempo`, `energy`, and `danceability`

If audio features are provided, the recommender concatenates text and audio representations into one hybrid feature space. Cosine similarity is then used to compare the query track with every other track.

### Internal flow

`build_hybrid_artifacts()`:

1. validates required columns
2. builds a TF-IDF matrix from `cleaned_text`
3. optionally converts selected audio columns into a sparse matrix
4. horizontally stacks text and audio features
5. returns reusable artifacts for fast repeated queries

`get_hybrid_recommendations()`:

1. validates inputs such as `top_n` and the ID column
2. builds artifacts if they were not passed in
3. locates the seed song using `song_id`
4. computes cosine similarity from the query row to all rows
5. excludes the seed song itself
6. ranks the top matches
7. returns a dataframe with:
   - `rank`
   - `track_id`
   - `artist`
   - `song`
   - `score`

### Why this design is useful

- reusable artifacts avoid rebuilding matrices for every query
- sparse matrices keep text-heavy similarity computation practical
- the recommender can run in pure lyric mode or hybrid lyric+audio mode
- the output schema is simple enough to integrate into a CLI, notebook, or web app later

## Data contract

The expected dataframe format is documented in `src/data_pipeline.py` and reinforced in `src/recommender.py`.

### Required columns

- `track_id`
- `artist`
- `song`
- `cleaned_text`

### Optional columns

- normalized numeric audio features such as:
  - `tempo`
  - `energy`
  - `danceability`
  - `valence`

### Expected meaning of `cleaned_text`

`cleaned_text` should contain preprocessed lyrics, typically after:

- lowercasing
- tokenization
- punctuation or special character removal
- stopword removal
- optional stemming or lemmatization

The recommender assumes this preprocessing has already been completed before inference.

## Data pipeline status

`src/data_pipeline.py` currently defines the interface for the dataset preparation step, but it is not implemented yet.

Right now:

- the function documents what the downstream recommender expects
- it raises `NotImplementedError`
- `main.py` therefore acts as a template for the intended integration rather than a fully runnable entry point

To complete the project pipeline, `load_and_clean_data()` should eventually:

1. load raw or processed data from the provided path
2. create or validate `track_id`
3. clean lyric text into `cleaned_text`
4. normalize any audio feature columns used for hybrid recommendation
5. return one combined dataframe matching the documented contract

## Notebook role

The notebook at `notebooks/music_recommendation_system.ipynb` is the experimental foundation of the project.

Based on the current notebook contents, it includes work around:

- loading `spotify_millsongdata.csv`
- generating `cleaned_text`
- TF-IDF-based lyric similarity
- sentence-transformer embeddings
- hybrid recommendation experiments using columns like:
  - `tempo`
  - `energy`
  - `valence`

The notebook is useful for:

- EDA
- feature experimentation
- validating recommendation quality
- prototyping before hardening code into `src/`

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

Current dependencies:

- `pandas`
- `numpy`
- `scikit-learn`
- `scipy`
- `nltk`
- `sentence-transformers`
- `faiss-cpu`

## How to use the code today

### Option 1: Use the notebook

This is the most complete path on the current `main` branch.

Use the notebook to:

- inspect the dataset
- preprocess lyrics
- experiment with TF-IDF, embedding, and hybrid approaches
- validate outputs before formalizing the data pipeline

### Option 2: Use `src/recommender.py` directly

If you already have a dataframe with the required columns, you can call the recommender directly:

```python
from src.recommender import get_hybrid_recommendations

results = get_hybrid_recommendations(
    song_id="track_123",
    combined_df=df,
    top_n=10,
    audio_feature_cols=["tempo", "energy", "valence"],
)
```

### Option 3: Finish the pipeline and use `main.py`

Once `src/data_pipeline.py` is implemented, `main.py` becomes the cleanest scripted entry point.

## Mock data for quick testing

`src/recommender.py` includes `make_mock_combined_df()`, which creates a tiny sample dataframe with:

- `track_id`
- `artist`
- `song`
- `cleaned_text`
- `tempo`
- `energy`

This is useful for:

- unit testing
- API verification
- development before the real data pipeline is ready

## Current limitations

- `main.py` is not runnable end-to-end yet because `load_and_clean_data()` is unimplemented
- there is no committed artifact-building script in `main`
- there is no UI or app layer in this branch
- the recommender assumes the input dataframe is already clean and aligned
- duplicate song IDs or poor `track_id` quality will affect recommendation lookup correctness

## Recommended next steps

To make the branch fully usable, the highest-value next tasks are:

1. implement `src/data_pipeline.py`
2. add a script to build processed datasets from raw CSV files
3. document expected dataset locations under `data/`
4. add tests for recommender input validation and ranking behavior
5. optionally add a simple CLI or Streamlit interface on top of `src/recommender.py`

## Summary

This repository already has a solid recommender core in `src/recommender.py`, with a clear contract for how the data layer should feed it. The notebook captures the experimental workflow, while `main.py` shows the intended production-style execution path. The primary missing piece on the current `main` branch is the implementation of the data pipeline that turns raw music data into the combined dataframe required by the hybrid recommender.
