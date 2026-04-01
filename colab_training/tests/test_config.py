"""Property-based and unit tests for fluor_modules.config.

Tests verify that MODEL_REGISTRY is complete and consistent with
the REQUIRED_MODEL_REGISTRY_KEYS specification.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from fluor_modules.config import (
    MODEL_REGISTRY,
    PROPERTIES,
    REQUIRED_MODEL_REGISTRY_KEYS,
)


# ---------------------------------------------------------------------------
# Property 14: Model Registry completeness
# For any property key in MODEL_REGISTRY, the config dict SHALL contain
# all required keys: num_layers, num_timesteps, dropout, alpha, model_class.
# **Validates: Requirements 12.4**
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(prop=st.sampled_from(PROPERTIES))
def test_model_registry_completeness(prop: str) -> None:
    """Every property in MODEL_REGISTRY has all REQUIRED_MODEL_REGISTRY_KEYS.

    **Validates: Requirements 12.4**
    """
    config = MODEL_REGISTRY[prop]
    missing = [k for k in REQUIRED_MODEL_REGISTRY_KEYS if k not in config]
    assert not missing, (
        f"MODEL_REGISTRY['{prop}'] is missing keys: {missing}"
    )

# ---------------------------------------------------------------------------
# Additional imports for unit tests
# ---------------------------------------------------------------------------
from fluor_modules.config import (
    BATCH_SIZE,
    FEATURE_VECTOR_DIM,
    GRAPH_FEAT_SIZE,
    LEARNING_RATE,
    MORGAN_NBITS,
    MORGAN_RADIUS,
    NUM_SCAFFOLD_FLAGS,
    NUMERIC_FEATURE_COUNT,
)


# ---------------------------------------------------------------------------
# Unit tests for config.py
# **Validates: Requirements 12.4**
# ---------------------------------------------------------------------------


class TestModelRegistryProperties:
    """Verify MODEL_REGISTRY has exactly the 4 expected property keys."""

    def test_registry_has_exactly_four_properties(self) -> None:
        expected = {"abs", "em", "plqy", "k"}
        assert set(MODEL_REGISTRY.keys()) == expected


class TestConfigConstants:
    """Verify all config constants match expected values."""

    def test_graph_feat_size(self) -> None:
        assert GRAPH_FEAT_SIZE == 256

    def test_batch_size(self) -> None:
        assert BATCH_SIZE == 32

    def test_learning_rate(self) -> None:
        assert LEARNING_RATE == 5e-4

    def test_morgan_radius(self) -> None:
        assert MORGAN_RADIUS == 2

    def test_morgan_nbits(self) -> None:
        assert MORGAN_NBITS == 1024

    def test_num_scaffold_flags(self) -> None:
        assert NUM_SCAFFOLD_FLAGS == 136

    def test_feature_vector_dim(self) -> None:
        assert FEATURE_VECTOR_DIM == 2192

    def test_numeric_feature_count(self) -> None:
        assert NUMERIC_FEATURE_COUNT == 8


class TestFeatureVectorDimDecomposition:
    """Verify FEATURE_VECTOR_DIM equals the sum of its components."""

    def test_dim_equals_component_sum(self) -> None:
        expected = (
            MORGAN_NBITS          # solvent fingerprint
            + MORGAN_NBITS        # molecule fingerprint
            + NUMERIC_FEATURE_COUNT
            + NUM_SCAFFOLD_FLAGS
        )
        assert FEATURE_VECTOR_DIM == expected


class TestModelClassAssignment:
    """Verify each property maps to the correct model class name."""

    def test_model_class_values_are_valid(self) -> None:
        valid_classes = {"GraphFingerprintsModel", "GraphFingerprintsModelFC"}
        for prop, cfg in MODEL_REGISTRY.items():
            assert cfg["model_class"] in valid_classes, (
                f"MODEL_REGISTRY['{prop}']['model_class'] = "
                f"'{cfg['model_class']}' is not a valid model class"
            )

    def test_abs_uses_graph_fingerprints_model(self) -> None:
        assert MODEL_REGISTRY["abs"]["model_class"] == "GraphFingerprintsModel"

    def test_em_uses_graph_fingerprints_model(self) -> None:
        assert MODEL_REGISTRY["em"]["model_class"] == "GraphFingerprintsModel"

    def test_plqy_uses_graph_fingerprints_model_fc(self) -> None:
        assert MODEL_REGISTRY["plqy"]["model_class"] == "GraphFingerprintsModelFC"

    def test_k_uses_graph_fingerprints_model_fc(self) -> None:
        assert MODEL_REGISTRY["k"]["model_class"] == "GraphFingerprintsModelFC"
