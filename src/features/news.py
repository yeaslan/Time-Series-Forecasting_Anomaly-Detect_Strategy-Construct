from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def embed_news(news_df: pd.DataFrame, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> pd.DataFrame:
    if news_df.empty:
        return news_df

    model = SentenceTransformer(model_name)
    text_series = news_df["headline"].fillna("")
    if "body" in news_df:
        text_series = text_series + " " + news_df["body"].fillna("")
    embeddings = model.encode(text_series.tolist(), batch_size=64, show_progress_bar=False)
    embed_cols = [f"news_emb_{i}" for i in range(embeddings.shape[1])]
    embed_df = pd.DataFrame(embeddings, columns=embed_cols)
    return pd.concat([news_df.reset_index(drop=True), embed_df], axis=1)


def aggregate_news_embeddings(
    news_embeddings: pd.DataFrame,
    prices: pd.DataFrame,
    lookback_days: int = 5,
    decay: float = 0.7,
    ts_col: str = "timestamp",
) -> pd.DataFrame:
    if news_embeddings.empty:
        return prices

    news_embeddings = news_embeddings.copy()
    news_embeddings[ts_col] = pd.to_datetime(news_embeddings[ts_col], utc=True)

    embed_cols = [col for col in news_embeddings.columns if col.startswith("news_emb_")]
    aggregated_frames = []
    for ticker, subdf in news_embeddings.groupby("ticker"):
        subdf = subdf.sort_values(ts_col)
        for col in embed_cols:
            subdf[f"{col}_agg"] = subdf[col].ewm(span=lookback_days, adjust=False).mean()
        aggregated_frames.append(subdf[["ticker", ts_col] + [f"{col}_agg" for col in embed_cols]])

    if not aggregated_frames:
        return prices

    aggregated = pd.concat(aggregated_frames, ignore_index=True)
    merged = prices.merge(aggregated, on=["ticker", ts_col], how="left")
    return merged
