from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass(frozen=True)
class HybridRecommenderArtifacts:
    model: SentenceTransformer      # 存储模型，以便后续对新输入的歌词做 encode
    embeddings: np.ndarray          # 存储所有歌曲的歌词向量
    hybrid_matrix: np.ndarray       # 存储“歌词向量 + 音频特征”的最终矩阵

#把文字变成数字，把不同的特征拼在一起。
'''
This function is used to build the hybrid artifacts for the hybrid recommender.
这是“离线阶段”运行的代码，任务是把原始表格变成数学矩阵。
这是“离线阶段”运行的代码，任务是把原始表格变成数学矩阵。

加载模型：SentenceTransformer 加载了多语言 BERT 模型。
生成 Embedding：这是最耗时的步骤。

核心逻辑：它不再是数单词出现了几次（TF-IDF），而是通过神经网络理解歌词的意境。

特征拼接 (Feature Fusion)：
np.hstack([embeddings, audio_features])。

复习要点：这就是所谓的 Hybrid（混合）。如果你不仅想让推荐的歌“歌词意思像”，还想让它们的“节奏、能量感”也像，就需要这一步。
'''
def build_hybrid_artifacts(
    combined_df: pd.DataFrame,
    *,
    model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2', # 推荐的多语言模型
    text_col: str = "cleaned_text",
    audio_feature_cols: Optional[Sequence[str]] = None,
) -> HybridRecommenderArtifacts:
    """
    将 TF-IDF 升级为 Multilingual BERT 嵌入。
    """
    if text_col not in combined_df.columns:
        raise ValueError(f"combined_df 必须包含 `{text_col}` 列")

    # 1. 加载多语言模型 (第一次运行会下载模型，约 500MB)它加载了一个能听懂 50 多种语言的“超级翻译官”
    print(f"正在加载模型: {model_name}...")
    model = SentenceTransformer(model_name)

    # 2. 生成歌词 Embedding
    # 注意：如果数据量很大（如 >1万条），这一步在 CPU 上会比较慢
    #理解： 这一步把每一行歌词都变成了一串很长的数字（向量）。这些数字就代表了这首歌的“语言特征”。
    print("正在生成歌词向量嵌入...")
    text_list = combined_df[text_col].fillna("").astype(str).tolist()
    embeddings = model.encode(text_list, show_progress_bar=True)

    # 3. 混合音频特征 (Hybrid Logic)
    if audio_feature_cols:
        print(f"正在合并音频特征: {audio_feature_cols}")
        missing = [c for c in audio_feature_cols if c not in combined_df.columns]
        if missing:
            raise ValueError(f"combined_df 缺失音频列: {missing}")

        # 将音频特征转化为 numpy array
        audio_features = combined_df[list(audio_feature_cols)].values
        
        # 将歌词向量与音频特征水平拼接 (Horizontal Stack)
        # 注意：这里我们使用 np.hstack，因为 BERT 向量是稠密的
        hybrid_matrix = np.hstack([embeddings, audio_features])
    else:
        hybrid_matrix = embeddings

    return HybridRecommenderArtifacts(
        model=model,
        embeddings=embeddings,
        hybrid_matrix=hybrid_matrix,
    )

'''
This function is used to get the hybrid recommendations for a given song.
'''
def get_hybrid_recommendations(
    query: str,                  # 现在支持输入 ID 或者 歌名
    combined_df: pd.DataFrame,
    artifacts: HybridRecommenderArtifacts,
    top_n: int = 10,
    *,
    id_col: str = "track_id",
    name_col: str = "song"       # 歌名列的名称
) -> pd.DataFrame:
    """
    核心推荐引擎：输入一个 ID 或 歌名，返回最相似的歌曲列表。
    """
    
    # --- 第一步：定位歌曲在库中的位置 ---
    # 首先尝试通过 ID 匹配
    match = combined_df.index[combined_df[id_col].astype(str) == str(query)]
    
    # 如果 ID 没匹配到，尝试通过歌名匹配 (忽略大小写)
    if len(match) == 0:
        match = combined_df.index[combined_df[name_col].str.lower() == str(query).lower()]
    
    # 如果还是没找到，说明库里确实没有
    if len(match) == 0:
        raise KeyError(f"在数据库中未找到相关的歌曲 ID 或歌名: '{query}'")

    # 获取这首歌在矩阵中的行索引（Row Position）
    # get_indexer 会根据 index 找到它在数组中的第几行
    query_row_pos = combined_df.index.get_indexer([match[0]])[0]
    
    # --- 第二步：提取特征并计算相似度 ---
    # 从“终极地图”里取出这首歌的特征向量
    # reshape(1, -1) 是为了把 1D 数组变成 2D 矩阵，满足相似度计算的格式
    query_vec = artifacts.hybrid_matrix[query_row_pos].reshape(1, -1)

    # 计算该向量与地图中“所有歌曲”的余弦相似度
    # 
    sims = cosine_similarity(query_vec, artifacts.hybrid_matrix).ravel()

    # --- 第三步：排序与筛选 ---
    # 排除掉搜索的歌曲本身（相似度设为负无穷），防止第一名永远是自己
    sims[query_row_pos] = -np.inf

    # 获取相似度最高的前 top_n 个索引
    # np.argsort 会返回排序后的下标，加负号表示降序排
    top_idx = np.argsort(-sims)[:top_n]

    # --- 第四步：包装结果返回 ---
    # 从原始数据表格 (combined_df) 中提取出这几首歌的信息
    out = combined_df.iloc[top_idx].copy()
    
    # 添加排名和相似度分数
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out["score"] = sims[top_idx]

    # 只返回用户关心的列
    return out[["rank", id_col, "artist", "song", "score"]].reset_index(drop=True)

'''
情况 1：输入的歌曲在你的“地图”里
场景： 用户输入“青花瓷”，而 Member A 之前洗好的数据（combined_df）里正好有这首歌。

定位： 程序会在 combined_df 的 song 列里搜寻“青花瓷”。

提取： 找到这首歌所在的行号（比如第 500 行）。

比对： 从 hybrid_matrix（你的终极地图）里直接取出第 500 行的那个数字向量。

计算： 用这个向量去和地图上所有其他行算余弦相似度。

返回： 挑出分数最高的 Top-N。
'''




# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Optional, Sequence

# import numpy as np
# import pandas as pd
# from scipy import sparse
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# '''
# 这是你的主战场。你目前正在做的 Multilingual Embeddings 
# 逻辑就应该写在这里 。核心逻辑是：“先提取特征存起来（Artifacts），等有人查询时再快速计算相似度。”
# 任务：
#  把你之前在 Notebook 里写的 TF-IDF 或现在的 BERT 模型封装成一个类（Class）或一系列函数 。目标：
#  输入一个歌名，这个文件里的逻辑要负责计算相似度并返回推荐列表 。

#  内容： 这是你作为 Member B 为团队提供的“服务接口”。它使用了 Artifacts（中间产物） 模式。
#  关键改进：内存优化：它不再预计算巨大的 $N \times N$ 矩阵，而是在用户请求时才计算“查询向量”与“整体矩阵”的相似度，
#  速度快且省内存。混合推荐 (Hybrid)：它已经预留了 audio_feature_cols 接口。
#  这意味着一旦 Member A 把音频特征（能量、节奏等）洗好，你只需要把列名传进去，它就能自动完成“歌词+音频”的联合推荐。
#  Mock 数据：它自带了一个 make_mock_combined_df。这意味着即使 Member A 还没把数据洗完，你现在也能运行测试代码。

# '''

# '''
# This class is used to store the artifacts for the hybrid recommender.
# 这是什么： 它像是一个“保险箱”，用来存放计算好的中间结果（Artifacts）。

# 为什么这样做： 在推荐系统中，计算歌词特征（TF-IDF）非常耗时。
# 我们把计算好的矩阵存进这个对象里，这样下次有人想要推荐时，直接拿出来用，而不需要重新计算整套数据。
# '''
# @dataclass(frozen=True)
# class HybridRecommenderArtifacts:
#     tfidf_vectorizer: TfidfVectorizer
#     tfidf_matrix: sparse.spmatrix
#     hybrid_matrix: sparse.spmatrix


# def build_hybrid_artifacts(
#     combined_df: pd.DataFrame,
#     *,
#     text_col: str = "cleaned_text",
#     audio_feature_cols: Optional[Sequence[str]] = None,
#     max_tfidf_features: int = 5000,

# ) -> HybridRecommenderArtifacts:
#     """
#     Build reusable matrices for fast recommendation queries.

#     Data contract (from `src.data_pipeline.load_and_clean_data`):
#     - `combined_df` must contain at least:
#       - `track_id` (recommended) OR a unique identifier column you will pass into `get_hybrid_recommendations`
#       - `song` (display name)
#       - `artist` (display name)
#       - `cleaned_text` (string): preprocessed lyrics for TF-IDF
#     - If you want hybrid recommendations, `combined_df` should also contain normalized numeric audio features
#       (e.g. `tempo`, `energy`, `danceability`, ...). Those columns are passed via `audio_feature_cols`.

#     Notes:
#     - This function DOES NOT compute an NxN similarity matrix (which is huge). We compute query-to-all cosine
#       similarity at request time using `cosine_similarity(query_vec, hybrid_matrix)`.
#     """
#     if text_col not in combined_df.columns:
#         raise ValueError(f"combined_df must include `{text_col}` column")

#     text_series = combined_df[text_col].fillna("").astype(str)
#     tfidf_vectorizer = TfidfVectorizer(max_features=max_tfidf_features)
#     tfidf_matrix = tfidf_vectorizer.fit_transform(text_series)

#     if audio_feature_cols:
#         missing = [c for c in audio_feature_cols if c not in combined_df.columns]
#         if missing:
#             raise ValueError(f"combined_df missing audio_feature_cols: {missing}")

#         audio = combined_df.loc[:, list(audio_feature_cols)].astype(float)
#         audio_sparse = sparse.csr_matrix(audio.values)
#         hybrid_matrix = sparse.hstack([tfidf_matrix, audio_sparse], format="csr")
#     else:
#         hybrid_matrix = tfidf_matrix

#     return HybridRecommenderArtifacts(
#         tfidf_vectorizer=tfidf_vectorizer,
#         tfidf_matrix=tfidf_matrix,
#         hybrid_matrix=hybrid_matrix,
#     )

# '''
# This function is used to get the hybrid recommendations for a given song.
# 这是整个系统的“出口”，负责根据用户给出的 song_id 找歌。

# 按需计算： 它不会预先计算所有歌两两之间的相似度（那样太占内存）。它只计算“你选的那首歌”和“库里其他所有歌”的相似度。

# 相似度计算： 使用 cosine_similarity 计算余弦相似度。分值越接近 1，表示两首歌越像。

# 排除自身： sims[query_row_pos] = -np.inf 确保推荐列表中不会出现用户正在听的那首歌。

# 排序与返回： 它会自动排序，选出最像的前 N 首歌，并返回一个干净的 DataFrame。
# '''

# def get_hybrid_recommendations(
#     song_id: str,
#     combined_df: pd.DataFrame,
#     top_n: int = 10,
#     *,
#     id_col: str = "track_id",
#     artifacts: Optional[HybridRecommenderArtifacts] = None,
#     text_col: str = "cleaned_text",
#     audio_feature_cols: Optional[Sequence[str]] = None,
#     max_tfidf_features: int = 5000,
# ) -> pd.DataFrame:
#     """
#     Return top-N recommendations for `song_id`.

#     **Contract**:
#     - `combined_df` should contain:
#       - `track_id` (or set `id_col` to your ID column)
#       - `song`, `artist`
#       - `cleaned_text` (preprocessed lyrics, string)
#     - Optional hybrid mode:
#       - pass `audio_feature_cols` listing numeric, normalized audio columns (tempo, energy, etc.)

#     Output:
#     - DataFrame with columns: `rank`, `track_id`, `artist`, `song`, `score`
#     """
#     if top_n <= 0:
#         raise ValueError("top_n must be positive")
#     if id_col not in combined_df.columns:
#         raise ValueError(f"combined_df must include `{id_col}` column (or set id_col)")

#     if artifacts is None:
#         artifacts = build_hybrid_artifacts(
#             combined_df,
#             text_col=text_col,
#             audio_feature_cols=audio_feature_cols,
#             max_tfidf_features=max_tfidf_features,
#         )

#     match = combined_df.index[combined_df[id_col].astype(str) == str(song_id)]
#     if len(match) == 0:
#         raise KeyError(f"song_id `{song_id}` not found in combined_df[{id_col}]")

#     query_row_pos = int(combined_df.index.get_indexer([match[0]])[0])
#     query_vec = artifacts.hybrid_matrix[query_row_pos]

#     sims = cosine_similarity(query_vec, artifacts.hybrid_matrix).ravel()
#     if sims.shape[0] != len(combined_df):
#         raise RuntimeError("similarity computation returned unexpected shape")

#     # exclude self
#     sims[query_row_pos] = -np.inf

#     top_idx = np.argpartition(-sims, kth=min(top_n, len(sims) - 1) - 1)[:top_n]
#     top_idx = top_idx[np.argsort(-sims[top_idx])]

#     out = combined_df.iloc[top_idx].copy()
#     out.insert(0, "rank", np.arange(1, len(out) + 1))
#     out["score"] = sims[top_idx]

#     # normalize output columns
#     keep_cols = ["rank", id_col, "artist", "song", "score"]
#     for col in ("artist", "song"):
#         if col not in out.columns:
#             out[col] = ""
#     return out.loc[:, keep_cols].rename(columns={id_col: "track_id"}).reset_index(drop=True)


# def make_mock_combined_df() -> pd.DataFrame:
#     """
#     A tiny fake dataset for Member B to test end-to-end before Member A finishes the pipeline.
#     """
#     return pd.DataFrame(
#         {
#             "track_id": ["track_1", "track_2", "track_3"],
#             "artist": ["A", "B", "C"],
#             "song": ["Love Song", "Sad Song", "Dance Song"],
#             "cleaned_text": [
#                 "love love heart forever",
#                 "sad tears lonely night",
#                 "dance party move tonight",
#             ],
#             # Optional audio features (already normalized here)
#             "tempo": [0.4, 0.2, 0.9],
#             "energy": [0.5, 0.3, 0.95],
#         }
#     )

