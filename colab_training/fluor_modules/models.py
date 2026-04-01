"""Neural network model architectures for fluorescent molecule property prediction.

Contains three model classes and a factory function:
- FingerprintAttentionCNN: CNN with attention for fingerprint feature extraction
- GraphFingerprintsModel: GNN + attention CNN for abs/em (state_dict compatible with Model_abs.pth, Model_em.pth)
- GraphFingerprintsModelFC: GNN + simple FC for plqy/k (state_dict compatible with Model_plqy.pth, Model_k.pth)
- build_model: Factory that reads MODEL_REGISTRY and instantiates the correct class

IMPORTANT: Layer attribute names must exactly match the pretrained .pth files so that
state_dict keys align. Verified against:
  - Model_abs.pth prefixes: fp_extractor, gnn, predict, readout, solvent_extractor
  - Model_plqy.pth prefixes: fp_fc, gnn, predict, readout

Reference: design.md section "models.py", Fluor-RLAT/training.py
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from dgllife.model.gnn import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout

from fluor_modules.config import MODEL_REGISTRY

logger = logging.getLogger(__name__)


class FingerprintAttentionCNN(nn.Module):
    """CNN with attention mechanism for fingerprint feature extraction.

    Applies 1D convolution with learned attention weights to extract
    informative features from concatenated fingerprints.

    Architecture:
        Input [B, D] -> unsqueeze -> [B, 1, D]
        -> conv_feat [B, C, D] (feature extraction)
        -> conv_attn [B, C, D] (attention weights)
        -> softmax attention -> weighted sum [B, C]
        -> concat with max pool [B, C] -> [B, 2C]

    Args:
        input_dim: Dimension of input fingerprint vector.
        conv_channels: Number of convolution channels. Output dim = 2 * conv_channels.

    Attribute names (conv_feat, conv_attn) match pretrained state_dict keys
    under the fp_extractor prefix.
    """

    def __init__(self, input_dim: int, conv_channels: int = 256) -> None:
        super(FingerprintAttentionCNN, self).__init__()
        self.conv_feat = nn.Conv1d(1, conv_channels, kernel_size=3, padding=1)
        self.conv_attn = nn.Conv1d(1, conv_channels, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=-1)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, D].

        Returns:
            Output tensor of shape [B, 2 * conv_channels].
        """
        x = x.unsqueeze(1)  # [B, 1, D]
        feat_map = self.conv_feat(x)  # [B, C, D]
        attn_map = self.conv_attn(x)  # [B, C, D]
        attn_weights = self.softmax(attn_map)
        attn_out = torch.sum(feat_map * attn_weights, dim=-1)  # [B, C]
        pooled = self.pool(feat_map).squeeze(-1)  # [B, C]
        return torch.cat([attn_out, pooled], dim=1)  # [B, 2C]


class GraphFingerprintsModel(nn.Module):
    """GNN + attention CNN model for abs/em property prediction.

    Combines:
    - AttentiveFP GNN for molecular graph encoding -> graph_feat_size dims
    - FingerprintAttentionCNN for molecule fingerprint + descriptors -> 2 * graph_feat_size dims
    - FC network for solvent fingerprint -> graph_feat_size dims

    Prediction head input: graph_feat_size * 4 = graph + solvent + 2*cnn_channels
    (e.g. 256 + 256 + 512 = 1024 for default graph_feat_size=256)

    Attribute names (gnn, readout, fp_extractor, solvent_extractor, predict) match
    pretrained Model_abs.pth and Model_em.pth state_dict key prefixes exactly.

    Args:
        node_feat_size: Dimension of node features from atom featurizer.
        edge_feat_size: Dimension of edge features from bond featurizer.
        solvent_dim: Dimension of solvent fingerprint (typically 1024).
        smiles_extra_dim: Dimension of molecule FP + extra features (typically 1168).
        graph_feat_size: GNN hidden/output dimension (default 256).
        num_layers: Number of AttentiveFP GNN layers.
        num_timesteps: Number of AttentiveFP readout timesteps.
        n_tasks: Number of prediction targets (default 1).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        node_feat_size: int,
        edge_feat_size: int,
        solvent_dim: int,
        smiles_extra_dim: int,
        graph_feat_size: int = 256,
        num_layers: int = 2,
        num_timesteps: int = 2,
        n_tasks: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super(GraphFingerprintsModel, self).__init__()

        self.gnn = AttentiveFPGNN(
            node_feat_size=node_feat_size,
            edge_feat_size=edge_feat_size,
            num_layers=num_layers,
            graph_feat_size=graph_feat_size,
            dropout=dropout,
        )
        self.readout = AttentiveFPReadout(
            feat_size=graph_feat_size,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )

        # Fingerprint extractor with attention CNN
        # Attribute name "fp_extractor" matches pretrained state_dict prefix
        self.fp_extractor = FingerprintAttentionCNN(
            smiles_extra_dim,
            conv_channels=graph_feat_size,
        )

        # Solvent fingerprint extractor (simple FC)
        # Attribute name "solvent_extractor" matches pretrained state_dict prefix
        # Sequential indices: 0=Linear, 1=ReLU, 2=Dropout, 3=Linear
        self.solvent_extractor = nn.Sequential(
            nn.Linear(solvent_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, graph_feat_size),
        )

        # Prediction head
        # Input: graph(g) + solvent(g) + fp_extractor(2g) = 4g
        total_input_dim = graph_feat_size + graph_feat_size + 2 * graph_feat_size
        # Sequential indices: 0=Dropout, 1=Linear, 2=ReLU, 3=Linear
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_tasks),
        )

        # Store for forward pass splitting
        self.solvent_dim = solvent_dim

    def forward(
        self,
        g: "dgl.DGLGraph",  # noqa: F821
        node_feats: torch.Tensor,
        edge_feats: Optional[torch.Tensor],
        fingerprints: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            g: Batched DGL graph.
            node_feats: Node feature tensor.
            edge_feats: Edge feature tensor (may be None).
            fingerprints: Concatenated fingerprint tensor [B, solvent_dim + smiles_extra_dim].

        Returns:
            Prediction tensor of shape [B, n_tasks].
        """
        # Handle missing edge features
        if edge_feats is None or edge_feats.size(0) == 0:
            num_edges = g.number_of_edges()
            edge_feats = torch.zeros((num_edges, 10), device=g.device)

        # Graph encoding
        node_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_feats)  # [B, graph_feat_size]

        # Split fingerprints: solvent | smiles+extra
        solvent_feat = fingerprints[:, : self.solvent_dim]  # [B, solvent_dim]
        smiles_extra_feat = fingerprints[:, self.solvent_dim :]  # [B, smiles_extra_dim]

        # Extract features from each branch
        solvent_out = self.solvent_extractor(solvent_feat)  # [B, graph_feat_size]
        smiles_extra_out = self.fp_extractor(smiles_extra_feat)  # [B, 2*graph_feat_size]

        # Concatenate and predict
        combined = torch.cat([graph_feats, solvent_out, smiles_extra_out], dim=1)
        return self.predict(combined)


class GraphFingerprintsModelFC(nn.Module):
    """GNN + simple FC model for plqy/k property prediction.

    Combines:
    - AttentiveFP GNN for molecular graph encoding -> graph_feat_size dims
    - Simple FC network for all fingerprints combined -> graph_feat_size dims

    Prediction head input: graph_feat_size * 2 = graph + fp
    (e.g. 256 + 256 = 512 for default graph_feat_size=256)

    Attribute names (gnn, readout, fp_fc, predict) match pretrained
    Model_plqy.pth and Model_k.pth state_dict key prefixes exactly.

    Args:
        node_feat_size: Dimension of node features from atom featurizer.
        edge_feat_size: Dimension of edge features from bond featurizer.
        fp_size: Dimension of full fingerprint vector (typically 2192).
        graph_feat_size: GNN hidden/output dimension (default 256).
        num_layers: Number of AttentiveFP GNN layers.
        num_timesteps: Number of AttentiveFP readout timesteps.
        n_tasks: Number of prediction targets (default 1).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        node_feat_size: int,
        edge_feat_size: int,
        fp_size: int,
        graph_feat_size: int = 256,
        num_layers: int = 2,
        num_timesteps: int = 2,
        n_tasks: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super(GraphFingerprintsModelFC, self).__init__()

        self.gnn = AttentiveFPGNN(
            node_feat_size=node_feat_size,
            edge_feat_size=edge_feat_size,
            num_layers=num_layers,
            graph_feat_size=graph_feat_size,
            dropout=dropout,
        )
        self.readout = AttentiveFPReadout(
            feat_size=graph_feat_size,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )

        # Simple FC for fingerprints
        # Attribute name "fp_fc" matches pretrained state_dict prefix
        # Sequential indices: 0=Linear, 1=ReLU, 2=Dropout, 3=Linear
        self.fp_fc = nn.Sequential(
            nn.Linear(fp_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, graph_feat_size),
        )

        # Prediction head
        # Input: graph(g) + fp(g) = 2g
        # Sequential indices: 0=Dropout, 1=Linear, 2=ReLU, 3=Linear
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, n_tasks),
        )

    def forward(
        self,
        g: "dgl.DGLGraph",  # noqa: F821
        node_feats: torch.Tensor,
        edge_feats: Optional[torch.Tensor],
        fingerprints: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            g: Batched DGL graph.
            node_feats: Node feature tensor.
            edge_feats: Edge feature tensor (may be None).
            fingerprints: Full fingerprint tensor [B, fp_size].

        Returns:
            Prediction tensor of shape [B, n_tasks].
        """
        # Handle missing edge features
        if edge_feats is None or edge_feats.size(0) == 0:
            num_edges = g.number_of_edges()
            edge_feats = torch.zeros((num_edges, 10), device=g.device)

        node_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_feats)  # [B, graph_feat_size]
        fp_feats = self.fp_fc(fingerprints)  # [B, graph_feat_size]
        combined = torch.cat([graph_feats, fp_feats], dim=1)
        return self.predict(combined)


# ---------------------------------------------------------------------------
# Model class lookup for build_model factory
# ---------------------------------------------------------------------------
_MODEL_CLASSES: dict[str, type[nn.Module]] = {
    "GraphFingerprintsModel": GraphFingerprintsModel,
    "GraphFingerprintsModelFC": GraphFingerprintsModelFC,
}


def build_model(
    target: str,
    node_feat_size: int,
    edge_feat_size: int,
    solvent_dim: int,
    smiles_extra_dim: int,
    fp_size: int,
    graph_feat_size: int = 256,
) -> nn.Module:
    """Factory: instantiate the correct model class from MODEL_REGISTRY[target].

    Reads the model_class, num_layers, num_timesteps, and dropout from
    MODEL_REGISTRY for the given target property and creates the
    corresponding model instance.

    Args:
        target: Property name (abs, em, plqy, k).
        node_feat_size: Dimension of node features.
        edge_feat_size: Dimension of edge features.
        solvent_dim: Dimension of solvent fingerprint (used by GraphFingerprintsModel).
        smiles_extra_dim: Dimension of smiles+extra features (used by GraphFingerprintsModel).
        fp_size: Dimension of full fingerprint vector (used by GraphFingerprintsModelFC).
        graph_feat_size: GNN hidden dimension (default 256).

    Returns:
        Instantiated model (GraphFingerprintsModel or GraphFingerprintsModelFC).

    Raises:
        KeyError: If target is not in MODEL_REGISTRY.
        ValueError: If model_class in registry is unknown.
    """
    if target not in MODEL_REGISTRY:
        raise KeyError(f"Unknown target '{target}'. Valid: {list(MODEL_REGISTRY.keys())}")

    cfg = MODEL_REGISTRY[target]
    class_name = cfg["model_class"]

    if class_name not in _MODEL_CLASSES:
        raise ValueError(
            f"Unknown model_class '{class_name}' for target '{target}'. "
            f"Valid: {list(_MODEL_CLASSES.keys())}"
        )

    model_cls = _MODEL_CLASSES[class_name]
    num_layers = cfg["num_layers"]
    num_timesteps = cfg["num_timesteps"]
    dropout = cfg["dropout"]

    if model_cls is GraphFingerprintsModel:
        return GraphFingerprintsModel(
            node_feat_size=node_feat_size,
            edge_feat_size=edge_feat_size,
            solvent_dim=solvent_dim,
            smiles_extra_dim=smiles_extra_dim,
            graph_feat_size=graph_feat_size,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )
    else:
        # GraphFingerprintsModelFC
        return GraphFingerprintsModelFC(
            node_feat_size=node_feat_size,
            edge_feat_size=edge_feat_size,
            fp_size=fp_size,
            graph_feat_size=graph_feat_size,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )
