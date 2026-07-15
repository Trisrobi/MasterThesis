"""
Utilities for converting rolling financial graphs into
PyTorch Geometric datasets.
"""
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
def turn_adjacency_matrix_to_edge_index(adjacency_matrix,ticker_list):
    """
    Convert a binary directed adjacency matrix into PyTorch Geometric edge_index.

    adjacency_matrix:
        pd.DataFrame with rows = source tickers, columns = target tickers
        or np.ndarray ordered according to ticker_list.

    ticker_list:
        fixed ticker ordering used for node indexing.
    """

    if isinstance(adjacency_matrix, pd.DataFrame):
        adj= adjacency_matrix.loc[ticker_list, ticker_list].values
    else:
        adj=np.asarray(adjacency_matrix)
    source_nodes,target_nodes=np.where(adj == 1)
    edge_index= torch.tensor(
        np.vstack([source_nodes, target_nodes]),
        dtype=torch.long,
    )
    return edge_index



def build_graph_snapshot(date,processed,adjacency_matrix,ticker_list,node_feature_cols):
    day_df=(processed[processed["date"]==date]).copy()

    if len(day_df) != len(ticker_list):
        raise ValueError(f"Date {date} has {len(day_df)} rows, expected {len(ticker_list)}.")

    day_df = day_df.set_index('tic').loc[ticker_list]

    if day_df[node_feature_cols].isna().any().any():
        raise ValueError(f"Missing node features for Date {date}")

    x=torch.tensor(day_df[node_feature_cols].values,
                dtype=torch.float)

    edge_index=turn_adjacency_matrix_to_edge_index(adjacency_matrix,ticker_list)

    graph=Data(
        x=x,
        edge_index=edge_index,
    )
    graph.date=str(date)
    graph.tickers=ticker_list
    graph.num_tickers = len(ticker_list)
    graph.feature_names = node_feature_cols

    return graph

def build_pyg_graph_dataset(processed, adjacency_matrices, ticker_list, node_feature_cols=("returns",),):
    """
    Convert rolling Granger adjacency matrices into a list of PyG Data objects.
    Only dates available in both processed and adjacency_matrices are used.
    """
    # TODO Phase 3:
    # Add daily returns to processed before building PyG graph objects.
    # Required because node_feature_cols=["returns"] depends on this column.

    graphs=[]

    processed_dates=set(processed["date"].unique())
    graph_dates= set(adjacency_matrices.keys())

    valid_dates=sorted(processed_dates.intersection(graph_dates))

    for date in valid_dates:
        graph= build_graph_snapshot(date, processed, adjacency_matrices[date],ticker_list,list(node_feature_cols),)
        graphs.append(graph)
    
    return graphs


def embeddings_to_dataframe(graphs, embeddings_list, prefix="gnn_emb"):
    rows=[]

    for graph, embeddings in zip(graphs, embeddings_list):
        embeddings_np = embeddings.detach().cpu().numpy()

        for i, tic in enumerate(graph.tickers):
            row ={
                "date": graph.date,
                "tic": tic,
            }

            for j in range(embeddings_np.shape[1]):
                row[f"{prefix}_{j}"] = embeddings_np[i,j]

            rows.append(row)
    return pd.DataFrame(rows)