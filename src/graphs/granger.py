import networkx as nx
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
import os

# Granger-network settings.

'''
Option 1 FULL VAR(Not too useful unless parameters are changed)
from statsmodels.tsa.api import VAR

window= cfg.REBALANCE_WINDOW
for end in range(window, len(returns)-1):
    returns_window= returns.iloc[end-window:end]
    model= VAR(returns_window)
    results= model.fit(maxlags=5)
    print(results.summary())
'''






def build_or_load_adjacency_matrices_granger(returns, pickle_path, window, maxlag, alpha=0.05,boolean=False):
    if os.path.exists(pickle_path):
        print("Loading existing Granger adjacency matrices...")
        granger_adj_matrices = pd.read_pickle(pickle_path)
    else:
        print("Computing Granger adjacency matrices...")
        granger_adj_matrices = {}


        for end in range(window, len(returns)):
            date = returns.index[end]
            returns_window= returns.iloc[end-window:end]
            tickers=returns_window.columns

            adj= pd.DataFrame(
                0,
                index=tickers,
                columns=tickers,
                dtype=float
            )
            for causing in tickers:
                for caused in tickers:
                    if causing==caused:
                        continue

                    pair_data=returns_window[[caused, causing]].dropna()

                    try:
                        result=grangercausalitytests(
                            pair_data,
                            maxlag=maxlag,
                            verbose=boolean
                        )
                        p_value=result[maxlag][0]["ssr_ftest"][1]

                        if p_value <alpha:
                            adj.loc[causing, caused]=1.0

                    except Exception as e:
                        print(f"{causing} -> {caused}: {e}")
                        adj.loc[causing, caused]=0.0
            granger_adj_matrices[date]=adj

        pd.to_pickle(granger_adj_matrices,pickle_path)
        print("Saved Granger adjacency matrices.")
    return granger_adj_matrices
    

#Calculation of Key measures


def build_network_dataframe_granger(adjacency_matrices):
    
    network_data =[]
    for date, adjacency_matrix in adjacency_matrices.items():
        G = nx.from_pandas_adjacency(adjacency_matrix, create_using=nx.DiGraph)
        
        # optional: remove zero-weight edges if needed
        G.remove_edges_from([
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("weight", 0) == 0
        ])

        #print("Nodes:", G.number_of_nodes())
        #print("Edges:", G.number_of_edges())
        
        # compute features
        in_degree = dict(G.in_degree(weight="weight"))
        out_degree = dict(G.out_degree(weight="weight"))
        pagerank = nx.pagerank(G, weight="weight")

        # store results
        network_features = pd.DataFrame({
            "date": date,
            "tic": list(G.nodes()),
            "granger_in_degree": [in_degree.get(tic, 0) for tic in G.nodes()],
            "granger_out_degree": [out_degree.get(tic, 0) for tic in G.nodes()],
            "granger_pagerank": [pagerank.get(tic, 0) for tic in G.nodes()],
        })
        network_data.append(network_features)
    return pd.concat(network_data, ignore_index=True)


def compile_node_information_granger(network_data, processed_data):
    network_data = network_data.copy()
    processed_data = processed_data.copy()
    network_data["date"] = pd.to_datetime(network_data["date"])
    processed_data["date"] = pd.to_datetime(processed_data["date"])

    NODE_NETWORK_FEATURES = ["granger_in_degree", "granger_out_degree", "granger_pagerank"]

    network_data[NODE_NETWORK_FEATURES] = (
        network_data
        .groupby("tic")[NODE_NETWORK_FEATURES]
        .shift(1)
    )
    network_data=network_data.dropna()

    processed_data = processed_data.merge(network_data,on=["date", "tic"],how="left",validate="many_to_one")
    processed_data[NODE_NETWORK_FEATURES] = processed_data[NODE_NETWORK_FEATURES].fillna(0)
    return processed_data

def compile_market_information_granger(network_data,processed_data):
    network_data = network_data.copy()
    processed_data = processed_data.copy()
    network_data["date"] = pd.to_datetime(network_data["date"])
    processed_data["date"] = pd.to_datetime(processed_data["date"])
    average_granger_in_degree= network_data.groupby('date')['granger_in_degree'].mean()
    average_granger_out_degree= network_data.groupby('date')['granger_out_degree'].mean()
    average_granger_pagerank= network_data.groupby('date')['granger_pagerank'].mean()

    average_measures_date=pd.DataFrame({
        'Average_granger_in_degree': average_granger_in_degree,
        'Average_granger_out_degree': average_granger_out_degree, 
        'Average_granger_pagerank': average_granger_pagerank
    }).reset_index()

    MARKET_NETWORK_FEATURES_GRANGER = ["Average_granger_in_degree", "Average_granger_out_degree", "Average_granger_pagerank"]
    average_measures_date[MARKET_NETWORK_FEATURES_GRANGER] = (average_measures_date[MARKET_NETWORK_FEATURES_GRANGER].shift(1))

    
    processed_data = processed_data.merge(
        average_measures_date,
        on="date",
        how="left",
        validate="many_to_one"
    )


    processed_data[MARKET_NETWORK_FEATURES_GRANGER] = processed_data[MARKET_NETWORK_FEATURES_GRANGER].fillna(0)
    return processed_data


def add_granger_network_features(processed_data, adjacency_matrices):
    network_data = build_network_dataframe_granger(adjacency_matrices)

    processed_data = compile_node_information_granger(
        network_data,
        processed_data
    )

    processed_data = compile_market_information_granger(
        network_data,
        processed_data
    )

    return processed_data