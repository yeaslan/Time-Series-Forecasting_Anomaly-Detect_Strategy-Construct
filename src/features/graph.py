from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd
from node2vec import Node2Vec


def build_correlation_graph(edges: pd.DataFrame, threshold: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    for _, row in edges.iterrows():
        if row["corr_60d"] >= threshold:
            graph.add_edge(row["source"], row["target"], weight=row["corr_60d"])
    return graph


def compute_graph_features(graph: nx.Graph) -> pd.DataFrame:
    if graph.number_of_nodes() == 0:
        return pd.DataFrame()

    degree = dict(graph.degree())
    clustering = nx.clustering(graph, weight="weight")
    eigenvector = nx.eigenvector_centrality_numpy(graph, weight="weight")

    df = pd.DataFrame(
        {
            "ticker": list(graph.nodes()),
            "graph_degree": [degree[node] for node in graph.nodes()],
            "graph_clustering": [clustering[node] for node in graph.nodes()],
            "graph_eigenvector": [eigenvector[node] for node in graph.nodes()],
        }
    )
    return df


def compute_node2vec_embeddings(graph: nx.Graph, dimensions: int = 32, walk_length: int = 10, num_walks: int = 100) -> pd.DataFrame:
    if graph.number_of_nodes() == 0:
        return pd.DataFrame()

    node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=1, quiet=True)
    model = node2vec.fit(window=5, min_count=1, batch_words=4)

    embeddings = []
    for node in graph.nodes():
        embeddings.append({"ticker": node, **{f"graph_emb_{i}": vec for i, vec in enumerate(model.wv[node])}})
    return pd.DataFrame(embeddings)


def enrich_with_graph_features(
    df: pd.DataFrame,
    edges: pd.DataFrame,
    corr_threshold: float = 0.5,
    embedding_dim: int = 32,
) -> pd.DataFrame:
    graph = build_correlation_graph(edges, threshold=corr_threshold)
    graph_feats = compute_graph_features(graph)
    graph_embeds = compute_node2vec_embeddings(graph, dimensions=embedding_dim)

    merged = df.merge(graph_feats, on="ticker", how="left").merge(graph_embeds, on="ticker", how="left")
    return merged
