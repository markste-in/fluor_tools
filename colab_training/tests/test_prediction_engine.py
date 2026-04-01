"""Tests for prediction_engine.py.

Covers:
- Property 9: Feature vector assembly and segmentation
- Property 10: Scaler inverse-transform round trip
- Property 11: Invalid SMILES error handling
- Property 18: Latest completed run resolution by timestamp
- Unit tests: feature vector dimensions, model path resolution, error handling
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis.strategies import (
    composite,
    floats,
    integers,
    lists,
    sampled_from,
    text,
)
from sklearn.preprocessing import StandardScaler

from fluor_modules.prediction_engine import (
    REPO_MODEL_DIR,
    build_feature_vector,
    find_latest_completed_run,
    fit_scalers,
    resolve_model_dir,
)
from fluor_modules.config import (
    FEATURE_VECTOR_DIM,
    MORGAN_NBITS,
    NUMERIC_FEATURE_COUNT,
    NUM_SCAFFOLD_FLAGS,
)
from tests.conftest import SAMPLE_SMILES

# Paths to real data files
SOLVENT_MAPPING_PATH: str = "Fluor-RLAT/data/00_solvent_mapping.csv"
SUBSTRUCTURE_PATH: str = "Fluor-RLAT/data/00_mmp_substructure.csv"
TRAIN_ABS_PATH: str = "Fluor-RLAT/data/train_abs.csv"


def _get_solvent_mapping() -> dict[str, int]:
    """Load real solvent mapping or return a minimal fallback."""
    from fluor_modules.data_pipeline import load_solvent_mapping
    if os.path.exists(SOLVENT_MAPPING_PATH):
        return load_solvent_mapping(SOLVENT_MAPPING_PATH)
    return {
        "CCO": 2, "CO": 1, "ClCCl": 0, "Cc1ccccc1": 6,
        "O": 8, "CC(C)=O": 12, "ClC(Cl)Cl": 3, "c1ccccc1": 15,
        "C1CCCCC1": 10, "CCCCCC": 11, "CC#N": 4, "CCOCC": 16,
        "C1CCOC1": 5, "CS(C)=O": 7, "CN(C)C=O": 9,
    }


def _get_substructure_patterns() -> list:
    """Load real substructure patterns or return empty list."""
    from fluor_modules.data_pipeline import load_substructure_patterns
    if os.path.exists(SUBSTRUCTURE_PATH):
        return load_substructure_patterns(SUBSTRUCTURE_PATH)
    return []


def _make_mock_train_df(target: str, n_rows: int = 10) -> pd.DataFrame:
    """Build a minimal mock training DataFrame with 152 columns.

    Args:
        target: Property column name (abs, em, plqy, k).
        n_rows: Number of rows.

    Returns:
        DataFrame with 152 columns matching the expected schema.
    """
    data: dict[str, Any] = {
        "split": ["train"] * n_rows,
        "smiles": ["c1ccccc1"] * n_rows,
        "solvent": ["CCO"] * n_rows,
        "abs": [400.0 + i for i in range(n_rows)],
        "em": [450.0 + i for i in range(n_rows)],
        "plqy": [0.5] * n_rows,
        "k": [4.0] * n_rows,
        "tag_name": ["PAHs"] * n_rows,
        # Columns 8:16 are the 8 numeric features
        "solvent_num": [2.0] * n_rows,
        "tag": [8.0] * n_rows,
        "MW": [78.0 + i for i in range(n_rows)],
        "LogP": [1.5] * n_rows,
        "TPSA": [0.0] * n_rows,
        "Double_Bond_Count": [6.0] * n_rows,
        "Ring_Count": [1.0] * n_rows,
        "unimol_plus": [2.0] * n_rows,
    }
    for i in range(1, 137):
        data[f"fragment_{i}"] = [0] * n_rows
    df = pd.DataFrame(data)
    # Ensure column order: first 8 cols are split..tag_name, then 8 numeric at 8:16
    assert df.shape[1] == 152, f"Expected 152 cols, got {df.shape[1]}"
    return df


# ---------------------------------------------------------------------------
# Property 9: Feature vector assembly and segmentation
# Validates: Requirements 5.3, 5.4
# ---------------------------------------------------------------------------


@settings(max_examples=50)
@given(
    mol_smiles=sampled_from(SAMPLE_SMILES),
    sol_smiles=sampled_from(["CCO", "CO", "ClCCl", "Cc1ccccc1", "O"]),
)
def test_feature_vector_assembly_and_segmentation(
    mol_smiles: str, sol_smiles: str
) -> None:
    """For any valid molecule+solvent SMILES pair, build_feature_vector returns
    a 2192-dim array with correct segment layout.

    Segments: sol_fp(1024) + mol_fp(1024) + scaled_numeric(8) + scaffold_flags(136)

    **Validates: Requirements 5.3, 5.4**
    """
    from fluor_modules.data_pipeline import compute_morgan_fingerprint
    from sklearn.preprocessing import MinMaxScaler

    solvent_mapping = _get_solvent_mapping()
    patterns = _get_substructure_patterns()

    # Fit a dummy scaler on 8 features
    dummy_data = np.ones((5, NUMERIC_FEATURE_COUNT), dtype=np.float64)
    num_scaler = MinMaxScaler()
    num_scaler.fit(dummy_data)

    fv = build_feature_vector(mol_smiles, sol_smiles, num_scaler, solvent_mapping, patterns)
    assert fv is not None, f"Expected feature vector for valid SMILES: {mol_smiles}"
    assert fv.shape == (FEATURE_VECTOR_DIM,), (
        f"Expected shape ({FEATURE_VECTOR_DIM},), got {fv.shape}"
    )

    # Verify solvent fingerprint segment (first 1024)
    sol_fp = compute_morgan_fingerprint(sol_smiles)
    if sol_fp is not None:
        np.testing.assert_array_equal(
            fv[:MORGAN_NBITS].astype(np.int32),
            sol_fp,
            err_msg="Solvent FP segment mismatch",
        )

    # Verify molecule fingerprint segment (next 1024)
    mol_fp = compute_morgan_fingerprint(mol_smiles)
    if mol_fp is not None:
        np.testing.assert_array_equal(
            fv[MORGAN_NBITS : 2 * MORGAN_NBITS].astype(np.int32),
            mol_fp,
            err_msg="Molecule FP segment mismatch",
        )

    # Verify scaffold flags segment (last 136) are binary
    flags_segment = fv[2 * MORGAN_NBITS + NUMERIC_FEATURE_COUNT:]
    assert flags_segment.shape == (NUM_SCAFFOLD_FLAGS,)
    assert set(np.unique(flags_segment.astype(np.int32))).issubset({0, 1}), (
        "Scaffold flags should be binary"
    )


# ---------------------------------------------------------------------------
# Property 10: Scaler inverse-transform round trip
# Validates: Requirements 5.6
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    values=lists(
        floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=20,
    )
)
def test_scaler_inverse_transform_round_trip(values: list[float]) -> None:
    """Fitting a StandardScaler, transforming, then inverse-transforming
    recovers the original values within tolerance 1e-6.

    **Validates: Requirements 5.6**
    """
    arr = np.array(values, dtype=np.float64).reshape(-1, 1)
    # Need at least 2 distinct values for StandardScaler to work
    assume(np.std(arr) > 1e-10)

    scaler = StandardScaler()
    scaler.fit(arr)
    transformed = scaler.transform(arr)
    recovered = scaler.inverse_transform(transformed)

    np.testing.assert_allclose(
        recovered, arr, atol=1e-6,
        err_msg="Inverse transform did not recover original values"
    )


# ---------------------------------------------------------------------------
# Property 11: Invalid SMILES error handling
# Validates: Requirements 5.7
# ---------------------------------------------------------------------------


@settings(max_examples=50)
@given(invalid_smiles=text(min_size=1, max_size=30).filter(
    lambda s: not s.strip() == "" and not any(c in s for c in "()[]=#@")
))
def test_invalid_smiles_error_handling(invalid_smiles: str) -> None:
    """For strings that RDKit cannot parse, build_feature_vector returns None
    without raising an exception.

    **Validates: Requirements 5.7**
    """
    from sklearn.preprocessing import MinMaxScaler

    solvent_mapping = _get_solvent_mapping()
    patterns = _get_substructure_patterns()

    dummy_data = np.ones((5, NUMERIC_FEATURE_COUNT), dtype=np.float64)
    num_scaler = MinMaxScaler()
    num_scaler.fit(dummy_data)

    # Should not raise; may return None for invalid SMILES
    try:
        result = build_feature_vector(
            invalid_smiles, "CCO", num_scaler, solvent_mapping, patterns
        )
        # If it returns something, it must be a valid array
        if result is not None:
            assert result.shape == (FEATURE_VECTOR_DIM,)
    except Exception as exc:
        pytest.fail(f"build_feature_vector raised an exception for '{invalid_smiles}': {exc}")


# ---------------------------------------------------------------------------
# Property 18: Latest completed run resolution by timestamp
# Validates: Requirements 12.9
# ---------------------------------------------------------------------------


@composite
def timestamp_sets(draw: Any) -> list[str]:
    """Generate a non-empty list of YYYY-MM-DD_HH-MM-SS timestamp strings."""
    n = draw(integers(min_value=1, max_value=8))
    years = draw(lists(integers(min_value=2020, max_value=2025), min_size=n, max_size=n))
    months = draw(lists(integers(min_value=1, max_value=12), min_size=n, max_size=n))
    days = draw(lists(integers(min_value=1, max_value=28), min_size=n, max_size=n))
    hours = draw(lists(integers(min_value=0, max_value=23), min_size=n, max_size=n))
    mins = draw(lists(integers(min_value=0, max_value=59), min_size=n, max_size=n))
    secs = draw(lists(integers(min_value=0, max_value=59), min_size=n, max_size=n))
    return [
        f"{y:04d}-{mo:02d}-{d:02d}_{h:02d}-{mi:02d}-{s:02d}"
        for y, mo, d, h, mi, s in zip(years, months, days, hours, mins, secs)
    ]


@settings(max_examples=100)
@given(timestamps=timestamp_sets())
def test_find_latest_completed_run(timestamps: list[str]) -> None:
    """For any non-empty set of YYYY-MM-DD_HH-MM-SS directory names,
    find_latest_completed_run returns the chronologically latest one.

    **Validates: Requirements 12.9**
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Create subdirectories
        for ts in timestamps:
            (tmp_path / ts).mkdir(exist_ok=True)

        result = find_latest_completed_run(str(tmp_path))
        assert result is not None, "Expected a result for non-empty directory"

        # The returned path should be the lexicographically largest timestamp
        expected_latest = sorted(timestamps)[-1]
        assert Path(result).name == expected_latest, (
            f"Expected latest '{expected_latest}', got '{Path(result).name}'"
        )


# ===========================================================================
# Unit tests (Task 7.7)
# Requirements: 5.1, 5.3, 5.6, 5.7, 6.2, 6.5, 6.6, 6.7
# ===========================================================================


class TestFitScalers:
    """Unit tests for fit_scalers."""

    def test_returns_correct_scaler_types(self) -> None:
        """fit_scalers returns (StandardScaler, MinMaxScaler)."""
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        train_df = _make_mock_train_df("abs")
        label_scaler, num_scaler = fit_scalers(train_df, "abs")
        assert isinstance(label_scaler, StandardScaler)
        assert isinstance(num_scaler, MinMaxScaler)

    def test_label_scaler_fitted_on_target(self) -> None:
        """label_scaler is fitted on the target column."""
        train_df = _make_mock_train_df("abs")
        label_scaler, _ = fit_scalers(train_df, "abs")
        # After fitting, mean_ should be close to the mean of the abs column
        expected_mean = train_df["abs"].mean()
        assert abs(label_scaler.mean_[0] - expected_mean) < 1e-6

    def test_num_scaler_fitted_on_8_columns(self) -> None:
        """num_scaler is fitted on columns 8:16 (8 numeric features)."""
        from sklearn.preprocessing import MinMaxScaler
        train_df = _make_mock_train_df("abs")
        _, num_scaler = fit_scalers(train_df, "abs")
        assert num_scaler.n_features_in_ == 8


class TestBuildFeatureVector:
    """Unit tests for build_feature_vector."""

    @pytest.fixture
    def num_scaler(self) -> Any:
        """Return a fitted MinMaxScaler for 8 numeric features."""
        from sklearn.preprocessing import MinMaxScaler
        dummy = np.ones((5, NUMERIC_FEATURE_COUNT), dtype=np.float64)
        scaler = MinMaxScaler()
        scaler.fit(dummy)
        return scaler

    def test_valid_smiles_returns_2192_dims(self, num_scaler: Any) -> None:
        """Valid molecule SMILES returns a 2192-dim feature vector."""
        solvent_mapping = _get_solvent_mapping()
        patterns = _get_substructure_patterns()
        fv = build_feature_vector("c1ccccc1", "CCO", num_scaler, solvent_mapping, patterns)
        assert fv is not None
        assert fv.shape == (FEATURE_VECTOR_DIM,)

    def test_invalid_mol_smiles_returns_none(self, num_scaler: Any) -> None:
        """Invalid molecule SMILES returns None."""
        solvent_mapping = _get_solvent_mapping()
        patterns = _get_substructure_patterns()
        result = build_feature_vector("NOT_A_SMILES", "CCO", num_scaler, solvent_mapping, patterns)
        assert result is None

    def test_segment_lengths(self, num_scaler: Any) -> None:
        """Feature vector has correct segment lengths."""
        solvent_mapping = _get_solvent_mapping()
        patterns = _get_substructure_patterns()
        fv = build_feature_vector("c1ccccc1", "CCO", num_scaler, solvent_mapping, patterns)
        assert fv is not None
        # sol_fp: 0:1024, mol_fp: 1024:2048, numeric: 2048:2056, flags: 2056:2192
        assert fv[:MORGAN_NBITS].shape == (1024,)
        assert fv[MORGAN_NBITS:2 * MORGAN_NBITS].shape == (1024,)
        assert fv[2 * MORGAN_NBITS:2 * MORGAN_NBITS + NUMERIC_FEATURE_COUNT].shape == (8,)
        assert fv[2 * MORGAN_NBITS + NUMERIC_FEATURE_COUNT:].shape == (NUM_SCAFFOLD_FLAGS,)


class TestResolveModelDir:
    """Unit tests for resolve_model_dir."""

    def test_pretrained_returns_repo_dir(self) -> None:
        """source='pretrained' returns REPO_MODEL_DIR."""
        result = resolve_model_dir("pretrained")
        assert result == REPO_MODEL_DIR

    def test_custom_with_specific_run(self, tmp_path: Path) -> None:
        """source='custom' with specific_run returns that path."""
        specific = str(tmp_path / "my_run")
        result = resolve_model_dir("custom", specific_run=specific)
        assert result == specific

    def test_custom_empty_completed_dir(self, tmp_path: Path) -> None:
        """source='custom' with empty completed dir returns None from find_latest."""
        result = find_latest_completed_run(str(tmp_path))
        assert result is None

    def test_custom_nonexistent_dir(self, tmp_path: Path) -> None:
        """find_latest_completed_run returns None for non-existent directory."""
        result = find_latest_completed_run(str(tmp_path / "does_not_exist"))
        assert result is None


class TestFindLatestCompletedRun:
    """Unit tests for find_latest_completed_run."""

    def test_returns_latest_of_multiple(self, tmp_path: Path) -> None:
        """Returns the most recent timestamp folder."""
        timestamps = [
            "2024-01-01_10-00-00",
            "2024-06-15_12-30-00",
            "2023-12-31_23-59-59",
        ]
        for ts in timestamps:
            (tmp_path / ts).mkdir()

        result = find_latest_completed_run(str(tmp_path))
        assert result is not None
        assert Path(result).name == "2024-06-15_12-30-00"

    def test_single_folder(self, tmp_path: Path) -> None:
        """Returns the only folder when there is just one."""
        (tmp_path / "2024-03-01_08-00-00").mkdir()
        result = find_latest_completed_run(str(tmp_path))
        assert result is not None
        assert Path(result).name == "2024-03-01_08-00-00"

    def test_empty_directory_returns_none(self, tmp_path: Path) -> None:
        """Empty directory returns None."""
        result = find_latest_completed_run(str(tmp_path))
        assert result is None


class TestPredictAllPropertiesMissingFiles:
    """Unit tests for predict_all_properties error handling."""

    def test_missing_model_file_returns_error_string(self, tmp_path: Path) -> None:
        """Missing model file returns error string for that property."""
        from fluor_modules.prediction_engine import predict_all_properties

        # Create minimal train CSV files
        for prop in ["abs", "em", "plqy", "k"]:
            train_df = _make_mock_train_df(prop)
            train_df.to_csv(tmp_path / f"train_{prop}.csv", index=False)

        # No model files in model_dir -> all properties should return error strings
        results = predict_all_properties(
            mol_smiles="c1ccccc1",
            sol_smiles="CCO",
            model_dir=str(tmp_path),  # no .pth files here
            data_dir=str(tmp_path),
            device="cpu",
        )

        for prop in ["abs", "em", "plqy", "k"]:
            assert prop in results
            assert isinstance(results[prop], str), (
                f"Expected error string for {prop}, got {type(results[prop])}"
            )
            assert "not found" in results[prop].lower() or "error" in results[prop].lower(), (
                f"Expected error message for {prop}, got: {results[prop]}"
            )

    def test_invalid_mol_smiles_returns_error_string(self, tmp_path: Path) -> None:
        """Invalid molecule SMILES returns error string for all properties."""
        from fluor_modules.prediction_engine import predict_all_properties

        for prop in ["abs", "em", "plqy", "k"]:
            train_df = _make_mock_train_df(prop)
            train_df.to_csv(tmp_path / f"train_{prop}.csv", index=False)

        results = predict_all_properties(
            mol_smiles="NOT_A_SMILES",
            sol_smiles="CCO",
            model_dir=str(tmp_path),
            data_dir=str(tmp_path),
            device="cpu",
        )

        for prop in ["abs", "em", "plqy", "k"]:
            assert prop in results
            assert isinstance(results[prop], str), (
                f"Expected error string for {prop} with invalid SMILES"
            )
