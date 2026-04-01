"""Tests for fluor_modules.models -- model architectures and build_model factory.

Covers:
- Property 12: Model architecture layer names (Hypothesis PBT)
- Property 13: Model dimension invariants (Hypothesis PBT)
- Unit tests: forward pass shape, parameter counts, build_model factory
"""

import dgl
import torch
import pytest
from hypothesis import given, settings
from hypothesis.strategies import sampled_from

from fluor_modules.models import (
    FingerprintAttentionCNN,
    GraphFingerprintsModel,
    GraphFingerprintsModelFC,
    build_model,
)


# ---------------------------------------------------------------------------
# Strategies for property-based tests
# ---------------------------------------------------------------------------

graph_feat_sizes = sampled_from([32, 64, 128])
num_layers_st = sampled_from([2, 3])
num_timesteps_st = sampled_from([1, 2, 3])
dropout_st = sampled_from([0.2, 0.3, 0.4])

# Fixed small dims to keep tests fast
NODE_FEAT: int = 39
EDGE_FEAT: int = 12
SOLVENT_DIM: int = 1024
SMILES_EXTRA_DIM: int = 1168
FP_SIZE: int = 2192


# ---------------------------------------------------------------------------
# Property 12: Model architecture layer names
# **Validates: Requirements 8.1, 8.2**
# ---------------------------------------------------------------------------

class TestProperty12LayerNames:
    """For any instantiated model, state_dict keys SHALL contain the expected prefixes."""

    @settings(max_examples=100)
    @given(
        g=graph_feat_sizes,
        nl=num_layers_st,
        nt=num_timesteps_st,
        dp=dropout_st,
    )
    def test_graph_fingerprints_model_prefixes(
        self, g: int, nl: int, nt: int, dp: float
    ) -> None:
        """GraphFingerprintsModel state_dict keys contain {gnn, readout, fp_extractor, solvent_extractor, predict}."""
        model = GraphFingerprintsModel(
            node_feat_size=NODE_FEAT,
            edge_feat_size=EDGE_FEAT,
            solvent_dim=SOLVENT_DIM,
            smiles_extra_dim=SMILES_EXTRA_DIM,
            graph_feat_size=g,
            num_layers=nl,
            num_timesteps=nt,
            dropout=dp,
        )
        keys = set(model.state_dict().keys())
        required_prefixes = {"gnn", "readout", "fp_extractor", "solvent_extractor", "predict"}
        found_prefixes = {k.split(".")[0] for k in keys}
        assert required_prefixes.issubset(found_prefixes), (
            f"Missing prefixes: {required_prefixes - found_prefixes}"
        )

    @settings(max_examples=100)
    @given(
        g=graph_feat_sizes,
        nl=num_layers_st,
        nt=num_timesteps_st,
        dp=dropout_st,
    )
    def test_graph_fingerprints_model_fc_prefixes(
        self, g: int, nl: int, nt: int, dp: float
    ) -> None:
        """GraphFingerprintsModelFC state_dict keys contain {gnn, readout, fp_fc, predict}."""
        model = GraphFingerprintsModelFC(
            node_feat_size=NODE_FEAT,
            edge_feat_size=EDGE_FEAT,
            fp_size=FP_SIZE,
            graph_feat_size=g,
            num_layers=nl,
            num_timesteps=nt,
            dropout=dp,
        )
        keys = set(model.state_dict().keys())
        required_prefixes = {"gnn", "readout", "fp_fc", "predict"}
        found_prefixes = {k.split(".")[0] for k in keys}
        assert required_prefixes.issubset(found_prefixes), (
            f"Missing prefixes: {required_prefixes - found_prefixes}"
        )


# ---------------------------------------------------------------------------
# Property 13: Model dimension invariants
# **Validates: Requirements 8.3, 8.4, 8.5**
# ---------------------------------------------------------------------------

class TestProperty13DimensionInvariants:
    """For any graph_feat_size g, dimension invariants SHALL hold."""

    @settings(max_examples=100)
    @given(g=sampled_from([32, 64, 128, 256]))
    def test_graph_fingerprints_model_predict_input_dim(self, g: int) -> None:
        """GraphFingerprintsModel predict layer input dim == 4 * g."""
        model = GraphFingerprintsModel(
            node_feat_size=NODE_FEAT,
            edge_feat_size=EDGE_FEAT,
            solvent_dim=SOLVENT_DIM,
            smiles_extra_dim=SMILES_EXTRA_DIM,
            graph_feat_size=g,
        )
        # predict is a Sequential; first real Linear is at index 1 (index 0 is Dropout)
        linear_layer = model.predict[1]
        assert linear_layer.in_features == 4 * g, (
            f"Expected predict input dim {4 * g}, got {linear_layer.in_features}"
        )

    @settings(max_examples=100)
    @given(g=sampled_from([32, 64, 128, 256]))
    def test_graph_fingerprints_model_fc_predict_input_dim(self, g: int) -> None:
        """GraphFingerprintsModelFC predict layer input dim == 2 * g."""
        model = GraphFingerprintsModelFC(
            node_feat_size=NODE_FEAT,
            edge_feat_size=EDGE_FEAT,
            fp_size=FP_SIZE,
            graph_feat_size=g,
        )
        linear_layer = model.predict[1]
        assert linear_layer.in_features == 2 * g, (
            f"Expected predict input dim {2 * g}, got {linear_layer.in_features}"
        )

    @settings(max_examples=100)
    @given(g=sampled_from([32, 64, 128, 256]))
    def test_fingerprint_attention_cnn_output_dim(self, g: int) -> None:
        """FingerprintAttentionCNN output dim == 2 * conv_channels."""
        conv_channels = g  # design uses graph_feat_size as conv_channels
        cnn = FingerprintAttentionCNN(input_dim=SMILES_EXTRA_DIM, conv_channels=conv_channels)
        # Run a forward pass to verify output shape
        x = torch.randn(2, SMILES_EXTRA_DIM)
        out = cnn(x)
        assert out.shape[1] == 2 * conv_channels, (
            f"Expected output dim {2 * conv_channels}, got {out.shape[1]}"
        )


# ---------------------------------------------------------------------------
# Unit tests for models.py
# ---------------------------------------------------------------------------

def _make_small_graph(node_feat_size: int, edge_feat_size: int) -> tuple:
    """Create a small DGL graph with 2 nodes and 2 edges for testing.

    Returns (graph, node_feats, edge_feats) as a batch of 1.
    """
    g = dgl.graph(([0, 1], [1, 0]))
    g.ndata["h"] = torch.randn(2, node_feat_size)
    g.edata["e"] = torch.randn(2, edge_feat_size)
    return g, g.ndata["h"], g.edata["e"]


class TestForwardPass:
    """Verify forward pass output shape is [B, 1] for both model classes."""

    def test_graph_fingerprints_model_forward(self, small_model_factory: dict) -> None:
        """GraphFingerprintsModel forward produces shape [1, 1]."""
        p = small_model_factory
        model = GraphFingerprintsModel(
            node_feat_size=p["node_feat_size"],
            edge_feat_size=p["edge_feat_size"],
            solvent_dim=p["solvent_dim"],
            smiles_extra_dim=p["smiles_extra_dim"],
            graph_feat_size=p["graph_feat_size"],
        )
        model.eval()
        g, node_feats, edge_feats = _make_small_graph(
            p["node_feat_size"], p["edge_feat_size"]
        )
        # fingerprints: [1, solvent_dim + smiles_extra_dim]
        fp = torch.randn(1, p["solvent_dim"] + p["smiles_extra_dim"])
        with torch.no_grad():
            out = model(g, node_feats, edge_feats, fp)
        assert out.shape == (1, 1), f"Expected (1, 1), got {out.shape}"

    def test_graph_fingerprints_model_fc_forward(self, small_model_factory: dict) -> None:
        """GraphFingerprintsModelFC forward produces shape [1, 1]."""
        p = small_model_factory
        model = GraphFingerprintsModelFC(
            node_feat_size=p["node_feat_size"],
            edge_feat_size=p["edge_feat_size"],
            fp_size=p["fp_size"],
            graph_feat_size=p["graph_feat_size"],
        )
        model.eval()
        g, node_feats, edge_feats = _make_small_graph(
            p["node_feat_size"], p["edge_feat_size"]
        )
        fp = torch.randn(1, p["fp_size"])
        with torch.no_grad():
            out = model(g, node_feats, edge_feats, fp)
        assert out.shape == (1, 1), f"Expected (1, 1), got {out.shape}"

    def test_batched_forward_shape(self, small_model_factory: dict) -> None:
        """Forward pass with batch of 2 graphs produces shape [2, 1]."""
        p = small_model_factory
        model = GraphFingerprintsModelFC(
            node_feat_size=p["node_feat_size"],
            edge_feat_size=p["edge_feat_size"],
            fp_size=p["fp_size"],
            graph_feat_size=p["graph_feat_size"],
        )
        model.eval()
        g1 = dgl.graph(([0, 1], [1, 0]))
        g1.ndata["h"] = torch.randn(2, p["node_feat_size"])
        g1.edata["e"] = torch.randn(2, p["edge_feat_size"])
        g2 = dgl.graph(([0, 1, 2], [1, 2, 0]))
        g2.ndata["h"] = torch.randn(3, p["node_feat_size"])
        g2.edata["e"] = torch.randn(3, p["edge_feat_size"])
        bg = dgl.batch([g1, g2])
        fp = torch.randn(2, p["fp_size"])
        with torch.no_grad():
            out = model(bg, bg.ndata["h"], bg.edata["e"], fp)
        assert out.shape == (2, 1), f"Expected (2, 1), got {out.shape}"


class TestParameterCounts:
    """Verify parameter counts are positive (models have learnable params)."""

    def test_graph_fingerprints_model_params(self, small_model_factory: dict) -> None:
        """GraphFingerprintsModel has > 0 parameters."""
        p = small_model_factory
        model = GraphFingerprintsModel(
            node_feat_size=p["node_feat_size"],
            edge_feat_size=p["edge_feat_size"],
            solvent_dim=p["solvent_dim"],
            smiles_extra_dim=p["smiles_extra_dim"],
            graph_feat_size=p["graph_feat_size"],
        )
        total = sum(param.numel() for param in model.parameters())
        assert total > 0, "Model should have learnable parameters"

    def test_graph_fingerprints_model_fc_params(self, small_model_factory: dict) -> None:
        """GraphFingerprintsModelFC has > 0 parameters."""
        p = small_model_factory
        model = GraphFingerprintsModelFC(
            node_feat_size=p["node_feat_size"],
            edge_feat_size=p["edge_feat_size"],
            fp_size=p["fp_size"],
            graph_feat_size=p["graph_feat_size"],
        )
        total = sum(param.numel() for param in model.parameters())
        assert total > 0, "Model should have learnable parameters"


class TestBuildModel:
    """Verify build_model factory returns the correct class for each target."""

    def test_abs_returns_graph_fingerprints_model(self, small_model_factory: dict) -> None:
        """build_model('abs', ...) returns GraphFingerprintsModel."""
        p = small_model_factory
        model = build_model(
            target="abs",
            node_feat_size=p["node_feat_size"],
            edge_feat_size=p["edge_feat_size"],
            solvent_dim=p["solvent_dim"],
            smiles_extra_dim=p["smiles_extra_dim"],
            fp_size=p["fp_size"],
            graph_feat_size=p["graph_feat_size"],
        )
        assert isinstance(model, GraphFingerprintsModel)

    def test_em_returns_graph_fingerprints_model(self, small_model_factory: dict) -> None:
        """build_model('em', ...) returns GraphFingerprintsModel."""
        p = small_model_factory
        model = build_model(
            target="em",
            node_feat_size=p["node_feat_size"],
            edge_feat_size=p["edge_feat_size"],
            solvent_dim=p["solvent_dim"],
            smiles_extra_dim=p["smiles_extra_dim"],
            fp_size=p["fp_size"],
            graph_feat_size=p["graph_feat_size"],
        )
        assert isinstance(model, GraphFingerprintsModel)

    def test_plqy_returns_graph_fingerprints_model_fc(self, small_model_factory: dict) -> None:
        """build_model('plqy', ...) returns GraphFingerprintsModelFC."""
        p = small_model_factory
        model = build_model(
            target="plqy",
            node_feat_size=p["node_feat_size"],
            edge_feat_size=p["edge_feat_size"],
            solvent_dim=p["solvent_dim"],
            smiles_extra_dim=p["smiles_extra_dim"],
            fp_size=p["fp_size"],
            graph_feat_size=p["graph_feat_size"],
        )
        assert isinstance(model, GraphFingerprintsModelFC)

    def test_k_returns_graph_fingerprints_model_fc(self, small_model_factory: dict) -> None:
        """build_model('k', ...) returns GraphFingerprintsModelFC."""
        p = small_model_factory
        model = build_model(
            target="k",
            node_feat_size=p["node_feat_size"],
            edge_feat_size=p["edge_feat_size"],
            solvent_dim=p["solvent_dim"],
            smiles_extra_dim=p["smiles_extra_dim"],
            fp_size=p["fp_size"],
            graph_feat_size=p["graph_feat_size"],
        )
        assert isinstance(model, GraphFingerprintsModelFC)

    def test_unknown_target_raises(self, small_model_factory: dict) -> None:
        """build_model with unknown target raises KeyError."""
        p = small_model_factory
        with pytest.raises(KeyError, match="Unknown target"):
            build_model(
                target="nonexistent",
                node_feat_size=p["node_feat_size"],
                edge_feat_size=p["edge_feat_size"],
                solvent_dim=p["solvent_dim"],
                smiles_extra_dim=p["smiles_extra_dim"],
                fp_size=p["fp_size"],
                graph_feat_size=p["graph_feat_size"],
            )
