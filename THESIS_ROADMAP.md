# Thesis Roadmap

This document serves as a living roadmap for the development of the thesis.

The project is divided into several stages. The aim is to first reproduce and extend the original FinRL implementation using Granger-causality networks, before investigating Graph Neural Networks (GNNs) as a more advanced graph representation.

---

# Current Status

## Completed

### Refactoring
- [x] Configuration module
- [x] Correlation network module
- [x] Granger network module

### Correlation Network
- [x] Correlation graph construction
- [x] Node-level features
- [x] Market-level features
- [x] Integration into the RL environment

### Granger Network
- [x] Rolling pairwise Granger graphs
- [x] Pickle caching of adjacency matrices
- [x] Directed graph construction
- [x] Node-level features
    - In-degree
    - Out-degree
    - PageRank
- [x] Market-level features
- [x] Integration into the RL environment
- [x] Initial validation

---

# TO-DO

## Refactoring

- [ ] Complete modularization of the remaining notebook.
- [ ] Move RL training into a dedicated module.
- [ ] Move environment construction into a dedicated module.
- [ ] Move evaluation and plotting into a dedicated module.

---

## Granger Graph Construction

- [ ] Decide on the final edge weighting scheme.
    - Binary
    - 1 − p-value
    - −log10(p-value)
    - F-statistic

- [ ] Compare weighted vs binary graphs.

---

## Directed Network Features

Currently implemented

- [x] In-degree
- [x] Out-degree
- [x] PageRank

Still to investigate

- [ ] Betweenness
- [ ] HITS Hub score
- [ ] HITS Authority score
- [ ] Graph density
- [ ] Reciprocity
- [ ] Strongly connected components
- [ ] Largest strongly connected component
- [ ] Additional market-level descriptors

---

## Data Pipeline

- [ ] Shift market-level network features by one period to eliminate potential look-ahead bias.
- [ ] Verify whether the same issue exists in the correlation-network implementation.
- [ ] Add additional sanity checks and validation.

---

## Experimental Evaluation

- [ ] Correlation vs Granger comparison.
- [ ] Different edge weighting schemes.
- [ ] Different rolling window lengths.
- [ ] Different lag lengths.
- [ ] Feature ablation studies.

---

## Graph Neural Network Integration

This constitutes the second methodological contribution of the thesis.

- [ ] Design node feature matrix.
- [ ] Construct graph datasets from rolling Granger networks.
- [ ] Build graph representation (edge index + node features).
- [ ] Integrate a first Graph Convolutional Network (GCN).
- [ ] Feed learned graph embeddings into the RL environment.
- [ ] Compare handcrafted network features against learned graph representations.

---

# FUTURE WORK

These ideas are beyond the intended scope of the current thesis.

## Advanced Graph Neural Networks

- Graph Attention Networks (GAT)
- GraphSAGE
- Temporal Graph Neural Networks
- Dynamic Graph Neural Networks
- Heterogeneous Graph Neural Networks

---

## Reinforcement Learning

- Alternative reward functions
    - Sharpe ratio
    - Sortino ratio
    - CVaR
    - Drawdown-aware rewards

- Alternative RL algorithms
- Multi-agent reinforcement learning
- Offline reinforcement learning

---

## Alternative Networks

- Commodity networks
- Foreign exchange networks
- Cross-market information networks
- Sector interaction networks
- Macroeconomic networks

---

## Alternative Graph Construction

- Full multivariate VAR networks
- Transfer entropy
- Bayesian networks
- Dynamic Bayesian networks

---

## Scaling

- S&P 500
- European equities
- Global equity universe
- Intraday networks

---

# Limitations Encountered

Current limitations of the implementation.

## Graph Construction

- Pairwise Granger causality ignores higher-order interactions.
- Static significance threshold.
- Binary graph representation (current implementation).

---

## Computational

- Rolling Granger estimation is computationally expensive.
- Large universes require substantial computation.

---

## Methodological

- Daily frequency only.
- Fixed rolling estimation window.
- Current implementation relies on handcrafted graph features.