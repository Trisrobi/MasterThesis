from __future__ import annotations

import os

import pandas as pd
import torch

def build_or_load_edge_index_bank(
        adjacency_by_date: dict,
        pickle_path:str,
        ticker_order: list[str],
)->tuple[list[torch.Tensor], dict[pd.Timestamp, int], list[pd.Timestamp]]:
    """
    Convert date-indexed adjacency DataFrames into a cached bank of
    PyTorch Geometric edge_index tensors.

    Parameters
    ----------
    adjacency_by_date
        Mapping from date to adjacency DataFrame.

    pickle_path
        Path used to cache the edge-index bank and its metadata.

    ticker_order
        Canonical node order. This must match the order used in the
        FinRL state and GNN node-feature matrix.

    Returns
    -------
    edge_index_bank
        List where edge_index_bank[i] has shape [2, num_edges_i].

    date_to_graph_id
        Mapping from graph date to its index in edge_index_bank.

    graph_dates
        Ordered graph dates corresponding to the bank.
    """
    if os.path.exists(pickle_path):
        print("Loading existing Granger edge-index bank...")

        cached= pd.read_pickle(pickle_path)

        return (
            cached["edge_index_bank"],
            cached["date_to_graph_id"],
            cached["graph_dates"],
        )
    
    print("Computing Granger edge-index bank...")

    normalised_adjacency_by_date = {
        pd.Timestamp(date): adjacency
        for date, adjacency in adjacency_by_date.items()
    }
    
    graph_dates = sorted(normalised_adjacency_by_date.keys())

    edge_index_bank=[]

    for date in graph_dates:
        adjacency= normalised_adjacency_by_date[date]

        missing_rows= set(ticker_order)- set(adjacency.index)
        missing_columns= set(ticker_order)- set(adjacency.columns)

        if missing_rows or missing_columns:
            raise ValueError(
                f"Ticker mismatch for graph date {date}. "
                f"Missing rows: {sorted(missing_rows)}. "
                f"Missing columns: {sorted(missing_columns)}. "
            )
        
        ordered_adjacency= adjacency.loc[
            ticker_order,
            ticker_order,
        ]

        adjacency_tensor= torch.as_tensor(
            ordered_adjacency.to_numpy(),
            dtype=torch.float32,
        )

        edge_index=(
            adjacency_tensor
            .nonzero(as_tuple=False)
            .t()
            .contiguous()
            .long()
        )

        edge_index_bank.append(edge_index)
    
    date_to_graph_id = {
        date: graph_id
        for graph_id, date in enumerate(graph_dates)
    }

    cache = {
        "edge_index_bank": edge_index_bank,
        "date_to_graph_id": date_to_graph_id,
        "graph_dates": graph_dates,
        "ticker_order": list(ticker_order),
    }

    pd.to_pickle(cache, pickle_path)

    print("Saved Granger edge-index bank.")

    return(
        edge_index_bank,
        date_to_graph_id,
        graph_dates,
    )