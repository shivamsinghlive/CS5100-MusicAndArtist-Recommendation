from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px


def create_3d_embedding_plot(embeddings: np.ndarray, metadata: pd.DataFrame):
    try:
        import umap  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install umap-learn to enable 3D visualization.") from exc

    reducer = umap.UMAP(n_components=3, random_state=42)
    reduced = reducer.fit_transform(embeddings)
    viz_df = metadata.copy().reset_index(drop=True)
    viz_df[["x", "y", "z"]] = reduced
    hover_cols = [col for col in ["artist", "song"] if col in viz_df.columns]
    return px.scatter_3d(viz_df, x="x", y="y", z="z", hover_data=hover_cols)
