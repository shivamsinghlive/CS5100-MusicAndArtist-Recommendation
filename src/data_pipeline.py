from __future__ import annotations

import pandas as pd
'''
This file is responsible for handling all the dirty work: 
reading the raw CSV files, cleaning the data, 
and aligning the audio features with the lyrics. 
Its output is a clean, 
merged DataFrame or Parquet file that your model 
can use directly.
'''

def load_and_clean_data(path: str) -> pd.DataFrame:
    """
    Member A owns this module.

    Expected output contract (what `src.recommender` expects):
    - Columns (minimum):
      - `track_id`: unique ID string for each track (recommended; can be generated if missing)
      - `artist`: artist display name
      - `song`: track title
      - `cleaned_text`: preprocessed lyrics string (tokenized/lemmatized/stopwords removed)
    - Optional (for hybrid):
      - normalized numeric audio features (e.g. `tempo`, `energy`, `danceability`, ...)

    `path` will typically be something like `data/raw/<file>.parquet` or `data/processed/<file>.parquet`.
    """
    raise NotImplementedError("Member A will implement data loading/cleaning here.")

