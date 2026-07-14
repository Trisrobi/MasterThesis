import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Correlation-network settings.
# threshold: minimum absolute correlation needed to create an edge.
# use_abs=True means strong negative and strong positive correlations are both treated as links.



def build_graph_at_date(rolling_corr, date, threshold=0.5, use_abs=True):
    """Build a weighted correlation graph for a single date.

    Parameters
    ----------
    rolling_corr : pd.DataFrame
        Multi-index rolling correlation matrix created by returns.rolling(...).corr().
    date : str or pd.Timestamp
        Date for which to extract the correlation matrix.
    threshold : float
        Correlation cutoff. Edges below this value are removed.
    use_abs : bool
        If True, threshold absolute correlations. If False, keep only positive correlations
        above the threshold.
    """
    corr_matrix = rolling_corr.loc[date].copy().fillna(0)
    adj_matrix=corr_matrix.abs() if use_abs else corr_matrix.copy()


    adj_matrix[adj_matrix < threshold] = 0
    np.fill_diagonal(adj_matrix.values, 0)

    G = nx.from_pandas_adjacency(adj_matrix)

    return G, adj_matrix

def graph_summary_stats(G):
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    components = list(nx.connected_components(G))
    largest_component_size = max(len(c) for c in components) if components else 0
    for u, v, d in G.edges(data=True):
        d["distance"] = 1 / d["weight"]

    betweenness = nx.betweenness_centrality(
        G,
        weight="distance"
    )

    stats = {
        "nodes": n_nodes,
        "edges": n_edges,
        "density": nx.density(G),
        "avg_clustering": nx.average_clustering(G, weight="weight"),
        "largest_component": largest_component_size,
        "largest_component_share": largest_component_size / n_nodes if n_nodes > 0 else 0,
        "avg_strength": np.mean([v for _, v in G.degree(weight="weight")]),
        "avg_betweenness": np.mean(list(betweenness.values())),
        "max_betweenness": max(betweenness.values()) if betweenness else 0
    }

    return stats

def build_network_dataframe(rolling_corr,threshold=0.5,use_abs=True):
    network_data=[]
    unique_dates = rolling_corr.index.get_level_values(0).unique()
    
    for date in unique_dates:
        
        G, adj_matrix = build_graph_at_date(
            rolling_corr,
            date,
            threshold=threshold,
            use_abs=use_abs
        )
        # compute features
        degree = nx.degree_centrality(G)
        clustering = nx.clustering(G, weight="weight")
        strength=dict(G.degree(weight="weight"))
        for u, v, d in G.edges(data=True):
            d["distance"] = 1 / d["weight"]
        betweenness= nx.betweenness_centrality(G,weight='distance')

        # store results
        for tic in adj_matrix.columns:
            network_data.append({
                "date": date,
                "tic": tic,
                "degree": degree.get(tic, 0),
                "clustering":clustering.get(tic,0),
                "strength": strength.get(tic, 0),
                "betweenness": betweenness.get(tic,0)
            })
    return pd.DataFrame(network_data)

def plot_graph_at_date(rolling_corr, date, threshold=0.5, use_abs=True, show_table=True, save_path=None, pos=None, Title=True):
    G, _ = build_graph_at_date(rolling_corr, date, threshold=threshold, use_abs=use_abs)

    fig = plt.figure(figsize=(13, 10))

    ax_graph = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
    ax_table = plt.subplot2grid((4, 1), (3, 0))

    if pos is None:
        pos = nx.spring_layout(G, seed=42)

    strength = dict(G.degree(weight="weight"))

    node_sizes = [
        100 + 10 * strength[node]
        for node in G.nodes()
    ]

    edge_widths = [
        0.5 + 2 * G[u][v]["weight"]
        for u, v in G.edges()
    ]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.8, ax=ax_graph)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4, ax=ax_graph)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax_graph)

    if Title:
        ax_graph.set_title(f"Network on {date} | threshold={threshold}")
    ax_graph.axis("off")

    ax_table.axis("off")

    if show_table:
        stats = graph_summary_stats(G)

        table_data = [
            ["Nodes", stats["nodes"]],
            ["Edges", stats["edges"]],
            ["Density", f'{stats["density"]:.3f}'],
            ["Avg clustering", f'{stats["avg_clustering"]:.3f}'],
            ["Largest component", stats["largest_component"]],
            ["Largest comp. share", f'{stats["largest_component_share"]:.2%}'],
            ["Avg strength", f'{stats["avg_strength"]:.3f}'],
            ["Avg Betweenness", f'{stats["avg_betweenness"]:.3f}'],
            ["Max Betweenness", f'{stats["max_betweenness"]:.3f}']
        ]

        table = ax_table.table(
            cellText=table_data,
            colLabels=["Metric", "Value"],
            loc="center",
            cellLoc="left"
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.1, 1.2)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return G


# ============================================================
# State-space extension: node-level network features
# ============================================================
# The original FinRL state contains:
# cash balance, prices, holdings, and technical indicators.
#
# This extension adds node-level network variables for each stock.
#
# IMPORTANT:
# Network features are shifted by one trading day within each ticker.
# This avoids look-ahead bias: the agent at time t should not use
# network information computed using same-day closing prices.
# ============================================================

def compile_node_information_correlation(network_data, processed_data):
    network_data = network_data.copy()
    processed_data = processed_data.copy()
    network_data["date"] = pd.to_datetime(network_data["date"])
    processed_data["date"] = pd.to_datetime(processed_data["date"])

    NODE_NETWORK_FEATURES = ["degree", "clustering", "strength", "betweenness"]

    network_data[NODE_NETWORK_FEATURES] = (
        network_data
        .groupby("tic")[NODE_NETWORK_FEATURES]
        .shift(1)
    )
    network_data=network_data.dropna()

    processed_data = processed_data.merge(network_data,on=["date", "tic"],how="left",validate="many_to_one")
    processed_data[NODE_NETWORK_FEATURES] = processed_data[NODE_NETWORK_FEATURES].fillna(0)
    return processed_data

# ============================================================
# State-space extension: market-level network features
# ============================================================
# Node-level features describe individual stocks.
# Market-level features describe the structure of the whole market.
#
# Market-level metrics are computed from the already-shifted node-level
# network features. Therefore, all network information available at date t
# is based only on information up to date t-1, preventing look-ahead bias.
#
#
# These variables provide information about the overall state
# of the market network.
#
# Examples include:
# - Average Degree Centrality
# - Average Clustering Coefficient
# - Average Strength
# - Average Betweenness Centrality
#
# This allows the agent to interpret node-level positions
# relative to the structure of the market as a whole.
#
# This is useful because the importance of a stock's node-level
# position depends on the overall market structure.
# ============================================================

def compile_market_information_correlation(network_data,processed_data):
    network_data = network_data.copy()
    processed_data = processed_data.copy()
    network_data["date"] = pd.to_datetime(network_data["date"])
    processed_data["date"] = pd.to_datetime(processed_data["date"])
    average_centralities_date= network_data.groupby('date')['degree'].mean()
    average_clustering_date= network_data.groupby('date')['clustering'].mean()
    average_betweenness_date= network_data.groupby('date')['betweenness'].mean()
    average_strength_date= network_data.groupby('date')['strength'].mean()

    average_measures_date=pd.DataFrame({
        'Average_Centrality': average_centralities_date,
        'Average_Clustering': average_clustering_date, 
        'Average_Betweenness': average_betweenness_date,
        'Average_Strength': average_strength_date
    }).reset_index()

    MARKET_NETWORK_FEATURES = ["Average_Centrality", "Average_Clustering", "Average_Betweenness", "Average_Strength"]
    average_measures_date[MARKET_NETWORK_FEATURES] = (average_measures_date[MARKET_NETWORK_FEATURES].shift(1))

    processed_data = processed_data.merge(
        average_measures_date,
        on="date",
        how="left",
        validate="many_to_one"
    )


    processed_data[MARKET_NETWORK_FEATURES] = processed_data[MARKET_NETWORK_FEATURES].fillna(0)
    return processed_data


def add_correlation_network_features(
    processed_data,
    rolling_corr,
    threshold=0.5,
    use_abs=True
):
    network_data = build_network_dataframe(
        rolling_corr,
        threshold=threshold,
        use_abs=use_abs
    )

    processed_data = compile_node_information_correlation(
        network_data,
        processed_data
    )

    processed_data = compile_market_information_correlation(
        network_data,
        processed_data
    )

    return processed_data