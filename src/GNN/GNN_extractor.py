from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Batch, Data
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.GNN.models import SimpleGCNEncoder

class GNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Initial SB3-compatible GNN feature extractor.

    This prototype assumes:
    - a fixed graph for all observations;
    - a flat Box observation;
    - the observation contains only node-level features;
    - the flat observation can be reshaped into
      [num_nodes, node_feature_dim].

    The extractor returns flattened node embeddings with shape:

        [batch_size, num_nodes * embedding_dim]
    """

    def __init__(
        self,
        observation_space,
        edge_index: torch.Tensor,
        num_nodes: int,
        node_feature_dim: int,
        embedding_dim: int = 8,
        hidden_channels: int = 16,
    ) -> None:
        self.num_nodes= num_nodes
        self.node_feature_dim = node_feature_dim
        self.embedding_dim= embedding_dim

        expected_observation_dim = num_nodes* node_feature_dim
        actual_observation_dim = int(observation_space.shape[0])
        
        if actual_observation_dim!=expected_observation_dim:
            raise ValueError(
                "Observation Dimension does not match the requested graph shape. "
                f"Received {actual_observation_dim} features, but "
                "num_nodes * node_feature_dim = "
                f"{num_nodes} * {node_feature_dim} = "
                f"{expected_observation_dim} features ."
            )
        
        features_dim= num_nodes*embedding_dim

        super().__init__(
            observation_space=observation_space,
            features_dim=features_dim
        )

        self.gnn = SimpleGCNEncoder(
            in_channels=node_feature_dim,
            hidden_channels=hidden_channels,
            embedding_dim=embedding_dim,
        )
        edge_index= torch.as_tensor(edge_index, dtype=torch.long)

        if edge_index.ndim!=2 or edge_index.shape[0] !=2:
            raise ValueError(
                f"edge_index must have shape: [2,num_edges]. "
                f"Received {tuple(edge_index.shape)}."
            )
        
        # register_buffer ensures that edge_index moves with the model
        # between CPU and GPU, while remaining non-trainable.
        self.register_buffer("edge_index", edge_index)

    def forward(self,observations: torch.Tensor)->torch.Tensor:
        """
        Convert a batch of flat observations into flattened GNN embeddings.

        Parameters
        ----------
        observations:
            Tensor of shape
            [batch_size, num_nodes * node_feature_dim].

        Returns
        -------
        torch.Tensor
            Tensor of shape
            [batch_size, num_nodes * embedding_dim].
        """
        batch_size= observations.shape[0]

        node_features = observations.reshape(
            batch_size,
            self.num_nodes,
            self.node_feature_dim,
        )

        graph_batch=self._build_graph_batch(node_features)
        node_embeddings = self.gnn(graph_batch)

        return node_embeddings.reshape(
            batch_size,
            self.num_nodes*self.embedding_dim,
        )
    
    def _build_graph_batch(self, node_features: torch.Tensor)-> Batch:
        """
        Build a PyTorch Geometric batch using the same fixed edge_index
        for every observation in the SB3 batch.
        """

        graphs=[
            Data(
                x=node_features[batch_index],
                edge_index= self.edge_index,
            )
            for batch_index in range(node_features.shape[0])
        ]
        return Batch.from_data_list(graphs)