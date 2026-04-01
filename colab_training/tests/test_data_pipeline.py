"""Tests for data_pipeline.py core functions.

Covers:
- Property 2: Morgan fingerprint shape and type
- Property 3: Molecular descriptors completeness
- Property 4: Scaffold classification mutual exclusivity
- Property 15: Fingerprint determinism
- Unit tests for solvent mapping, descriptors, scaffold, and data loading
"""

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis.strategies import (
    booleans,
    composite,
    floats,
    integers,
    lists,
    sampled_from,
    text,
)

from fluor_modules.data_pipeline import (
    SOLVENT_NAME_TO_SMILES,
    compute_molecular_descriptors,
    compute_morgan_fingerprint,
    compute_scaffold_flags,
    detect_scaffold,
    load_solvent_mapping,
    load_substructure_patterns,
    map_solvent_name_to_smiles,
    merge_training_data,
    process_input_csv,
    split_by_property,
)
from tests.conftest import SAMPLE_SMILES


# ---------------------------------------------------------------------------
# Valid scaffold family names from the actual code
# (16 defined families + "Other" fallback)
# ---------------------------------------------------------------------------
VALID_SCAFFOLD_FAMILIES: set[str] = {
    "SquaricAcid", "Naphthalimide", "Coumarin", "Carbazole",
    "Cyanine", "BODIPY", "Triphenylamine", "Porphyrin",
    "PAHs", "Acridines", "5p6", "6p6", "5n6", "6n6",
    "Azo", "Benz", "Other",
}

# Paths to real data files (relative to workspace root)
SOLVENT_MAPPING_PATH: str = "Fluor-RLAT/data/00_solvent_mapping.csv"
SUBSTRUCTURE_PATH: str = "Fluor-RLAT/data/00_mmp_substructure.csv"


# ---------------------------------------------------------------------------
# Property 2: Morgan fingerprint shape and type
# Validates: Requirements 2.4
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(smiles=sampled_from(SAMPLE_SMILES))
def test_morgan_fingerprint_shape_and_type(smiles: str) -> None:
    """For any valid SMILES, compute_morgan_fingerprint returns shape (1024,),
    int dtype, and all elements are 0 or 1.

    **Validates: Requirements 2.4**
    """
    fp = compute_morgan_fingerprint(smiles)
    assert fp is not None, f"Expected fingerprint for valid SMILES: {smiles}"
    assert fp.shape == (1024,), f"Expected shape (1024,), got {fp.shape}"
    assert np.issubdtype(fp.dtype, np.integer), (
        f"Expected integer dtype, got {fp.dtype}"
    )
    assert set(np.unique(fp)).issubset({0, 1}), (
        f"Expected only 0/1 values, got unique: {np.unique(fp)}"
    )


# ---------------------------------------------------------------------------
# Property 3: Molecular descriptors completeness
# Validates: Requirements 2.5
# ---------------------------------------------------------------------------

EXPECTED_DESCRIPTOR_KEYS: set[str] = {
    "MW", "LogP", "TPSA", "Double_Bond_Count", "Ring_Count",
}


@settings(max_examples=100)
@given(smiles=sampled_from(SAMPLE_SMILES))
def test_molecular_descriptors_completeness(smiles: str) -> None:
    """For any valid SMILES, compute_molecular_descriptors returns a dict
    with all 5 keys and finite numeric values.

    **Validates: Requirements 2.5**
    """
    desc = compute_molecular_descriptors(smiles)
    assert desc is not None, f"Expected descriptors for valid SMILES: {smiles}"
    assert set(desc.keys()) == EXPECTED_DESCRIPTOR_KEYS, (
        f"Expected keys {EXPECTED_DESCRIPTOR_KEYS}, got {set(desc.keys())}"
    )
    for key, value in desc.items():
        assert isinstance(value, (int, float)), (
            f"Expected numeric value for {key}, got {type(value)}"
        )
        assert np.isfinite(value), f"Expected finite value for {key}, got {value}"


# ---------------------------------------------------------------------------
# Property 4: Scaffold classification mutual exclusivity
# Validates: Requirements 2.6, 12.6
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(smiles=sampled_from(SAMPLE_SMILES))
def test_scaffold_classification_mutual_exclusivity(smiles: str) -> None:
    """For any valid SMILES, detect_scaffold returns exactly one (tag, family_name)
    where family_name is in the valid set. compute_scaffold_flags returns shape
    (136,) with all elements in {0, 1}.

    **Validates: Requirements 2.6, 12.6**
    """
    tag, family_name = detect_scaffold(smiles)
    assert isinstance(tag, int), f"Expected int tag, got {type(tag)}"
    assert isinstance(family_name, str), f"Expected str family, got {type(family_name)}"
    assert family_name in VALID_SCAFFOLD_FAMILIES, (
        f"Family '{family_name}' not in valid set: {VALID_SCAFFOLD_FAMILIES}"
    )

    # Scaffold flags need substructure patterns; load from real file if available,
    # otherwise use an empty pattern list (flags will be all zeros).
    if os.path.exists(SUBSTRUCTURE_PATH):
        patterns = load_substructure_patterns(SUBSTRUCTURE_PATH)
    else:
        patterns = []

    flags = compute_scaffold_flags(smiles, patterns)
    assert flags.shape == (136,), f"Expected shape (136,), got {flags.shape}"
    assert set(np.unique(flags)).issubset({0, 1}), (
        f"Expected only 0/1 values in flags, got unique: {np.unique(flags)}"
    )


# ---------------------------------------------------------------------------
# Property 15: Fingerprint determinism
# Validates: Requirements 12.5
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(smiles=sampled_from(SAMPLE_SMILES))
def test_fingerprint_determinism(smiles: str) -> None:
    """For any valid SMILES, calling compute_morgan_fingerprint twice
    produces identical arrays.

    **Validates: Requirements 12.5**
    """
    fp1 = compute_morgan_fingerprint(smiles)
    fp2 = compute_morgan_fingerprint(smiles)
    assert fp1 is not None, f"Expected fingerprint for valid SMILES: {smiles}"
    assert fp2 is not None, f"Expected fingerprint for valid SMILES: {smiles}"
    np.testing.assert_array_equal(fp1, fp2, err_msg="Fingerprints differ between calls")


# ===========================================================================
# Unit tests (Task 4.7)
# Requirements: 2.4, 2.5, 2.6, 2.7, 2.8
# ===========================================================================


class TestSolventMapping:
    """Unit tests for solvent name -> SMILES mapping."""

    @pytest.mark.parametrize(
        "name, expected_smiles",
        [
            ("Toluol", "Cc1ccccc1"),
            ("EtOH", "CCO"),
            ("MeOH", "CO"),
            ("DCM", "ClCCl"),
            ("Dichlormethan", "ClCCl"),
            ("CHCl3", "ClC(Cl)Cl"),
            ("Benzol", "c1ccccc1"),
            ("Wasser", "O"),
            ("Aceton", "CC(C)=O"),
            ("Cyclohexan", "C1CCCCC1"),
            ("Hexan", "CCCCCC"),
            ("Acetonitril", "CC#N"),
            ("Diethylether", "CCOCC"),
            ("THF", "C1CCOC1"),
            ("DMSO", "CS(C)=O"),
            ("DMF", "CN(C)C=O"),
        ],
    )
    def test_german_solvent_names(self, name: str, expected_smiles: str) -> None:
        """Known German solvent names map to expected SMILES."""
        result = map_solvent_name_to_smiles(name)
        assert result == expected_smiles, (
            f"Expected '{expected_smiles}' for '{name}', got '{result}'"
        )

    def test_unknown_solvent_returns_none(self) -> None:
        """Unknown solvent name that is not valid SMILES returns None."""
        result = map_solvent_name_to_smiles("UnknownSolvent")
        assert result is None

    def test_solvent_name_to_smiles_dict_completeness(self) -> None:
        """SOLVENT_NAME_TO_SMILES contains all expected German names."""
        german_names = [
            "Toluol", "EtOH", "MeOH", "DCM", "Dichlormethan", "CHCl3",
            "Benzol", "Wasser", "Aceton", "Cyclohexan", "Hexan",
            "Acetonitril", "Diethylether", "THF", "DMSO", "DMF",
        ]
        for name in german_names:
            assert name in SOLVENT_NAME_TO_SMILES, f"Missing German name: {name}"


class TestMorganFingerprint:
    """Unit tests for Morgan fingerprint computation."""

    def test_benzene_has_nonzero_fingerprint(self) -> None:
        """Benzene 'c1ccccc1' should have a non-zero fingerprint."""
        fp = compute_morgan_fingerprint("c1ccccc1")
        assert fp is not None
        assert fp.sum() > 0, "Benzene fingerprint should have non-zero bits"

    def test_invalid_smiles_returns_none(self) -> None:
        """Invalid SMILES returns None."""
        result = compute_morgan_fingerprint("NOT_A_SMILES")
        assert result is None

    def test_empty_string_returns_all_zeros(self) -> None:
        """Empty string is parsed by RDKit as an empty molecule, returning all-zero fingerprint."""
        # Assumption: RDKit treats "" as a valid (empty) molecule, not as invalid SMILES.
        result = compute_morgan_fingerprint("")
        assert result is not None
        assert result.sum() == 0, "Empty molecule fingerprint should be all zeros"


class TestMolecularDescriptors:
    """Unit tests for molecular descriptor computation."""

    def test_invalid_smiles_returns_none(self) -> None:
        """Invalid SMILES returns None for descriptors."""
        result = compute_molecular_descriptors("NOT_A_SMILES")
        assert result is None

    def test_benzene_descriptors(self) -> None:
        """Benzene should have known approximate descriptor values."""
        desc = compute_molecular_descriptors("c1ccccc1")
        assert desc is not None
        # Benzene MW ~ 78.11
        assert 77.0 < desc["MW"] < 80.0, f"Unexpected MW: {desc['MW']}"
        # Benzene has 1 ring
        assert desc["Ring_Count"] == 1.0


class TestScaffoldDetection:
    """Unit tests for scaffold detection."""

    def test_naphthalene_is_pahs(self) -> None:
        """Naphthalene 'c1ccc2ccccc2c1' should be classified as PAHs."""
        tag, family = detect_scaffold("c1ccc2ccccc2c1")
        assert family == "PAHs", f"Expected PAHs, got {family}"

    def test_invalid_smiles_returns_other(self) -> None:
        """Invalid SMILES returns ('Other', -1) fallback."""
        tag, family = detect_scaffold("NOT_A_SMILES")
        assert family == "Other"
        assert tag == -1


class TestDataFileLoading:
    """Unit tests for loading real data files.

    These tests use the actual CSV files from the repository.
    Tests are skipped if the files do not exist.
    """

    @pytest.fixture
    def solvent_mapping_path(self) -> str:
        """Return path to solvent mapping CSV, skip if missing."""
        if not os.path.exists(SOLVENT_MAPPING_PATH):
            pytest.skip(f"Solvent mapping file not found: {SOLVENT_MAPPING_PATH}")
        return SOLVENT_MAPPING_PATH

    @pytest.fixture
    def substructure_path(self) -> str:
        """Return path to substructure CSV, skip if missing."""
        if not os.path.exists(SUBSTRUCTURE_PATH):
            pytest.skip(f"Substructure file not found: {SUBSTRUCTURE_PATH}")
        return SUBSTRUCTURE_PATH

    def test_load_solvent_mapping(self, solvent_mapping_path: str) -> None:
        """load_solvent_mapping loads from the real CSV and returns a dict."""
        mapping = load_solvent_mapping(solvent_mapping_path)
        assert isinstance(mapping, dict)
        assert len(mapping) > 0, "Solvent mapping should not be empty"
        # Check a known entry: Toluene SMILES -> numeric ID
        assert "Cc1ccccc1" in mapping, "Toluene SMILES should be in mapping"
        assert mapping["Cc1ccccc1"] == 6

    def test_load_substructure_patterns_count(self, substructure_path: str) -> None:
        """load_substructure_patterns loads 136 patterns from the real CSV."""
        patterns = load_substructure_patterns(substructure_path)
        assert isinstance(patterns, list)
        # The CSV has 136 rows; some patterns might fail to compile,
        # but we expect most to succeed.
        assert len(patterns) > 100, (
            f"Expected >100 compiled patterns, got {len(patterns)}"
        )
        # Ideally all 136 compile
        assert len(patterns) <= 136, (
            f"Expected <=136 patterns, got {len(patterns)}"
        )

    def test_load_substructure_patterns_structure(self, substructure_path: str) -> None:
        """Each pattern is a (int_index, compiled_mol) tuple."""
        patterns = load_substructure_patterns(substructure_path)
        for idx, mol in patterns:
            assert isinstance(idx, int), f"Expected int index, got {type(idx)}"
            assert mol is not None, f"Pattern at index {idx} should not be None"


# ===========================================================================
# Tests for Tasks 5.3 - 5.8
# ===========================================================================

# ---------------------------------------------------------------------------
# Helper: known solvents that exist in the real mapping file
# (German names that map_solvent_name_to_smiles recognizes)
# ---------------------------------------------------------------------------
_KNOWN_SOLVENTS: list[str] = [
    "EtOH", "MeOH", "DCM", "Toluol", "Wasser", "Aceton",
    "CHCl3", "Benzol", "Cyclohexan", "Hexan", "Acetonitril",
    "Diethylether", "THF", "DMSO", "DMF",
]

_UNKNOWN_SOLVENTS: list[str] = [
    "UnknownSolvent", "FakeSolvent", "XYZ123", "Nope",
]


def _make_csv_content(
    rows: list[dict[str, str]],
) -> str:
    """Build CSV string from list of row dicts.

    Args:
        rows: List of dicts with keys matching CSV columns.

    Returns:
        CSV string with header and data rows.
    """
    cols = ["name", "solvent", "abs", "em", "epsilon", "mw", "plqy", "smiles"]
    lines = [",".join(cols)]
    for row in rows:
        lines.append(",".join(str(row.get(c, "")) for c in cols))
    return "\n".join(lines)


def _get_solvent_mapping() -> dict[str, int]:
    """Load real solvent mapping if available, else return a minimal one.

    Returns:
        Dict mapping solvent SMILES to numeric IDs.
    """
    if os.path.exists(SOLVENT_MAPPING_PATH):
        return load_solvent_mapping(SOLVENT_MAPPING_PATH)
    # Minimal fallback covering the solvents used in tests
    return {
        "CCO": 2, "CO": 1, "ClCCl": 0, "Cc1ccccc1": 6,
        "O": 8, "CC(C)=O": 12, "ClC(Cl)Cl": 3, "c1ccccc1": 15,
        "C1CCCCC1": 10, "CCCCCC": 11, "CC#N": 4, "CCOCC": 16,
        "C1CCOC1": 5, "CS(C)=O": 7, "CN(C)C=O": 9,
    }


def _get_substructure_patterns() -> list:
    """Load real substructure patterns if available, else return empty list.

    Returns:
        List of (index, compiled_mol) tuples.
    """
    if os.path.exists(SUBSTRUCTURE_PATH):
        return load_substructure_patterns(SUBSTRUCTURE_PATH)
    return []


# ---------------------------------------------------------------------------
# Property 1: Invalid rows are skipped with reasons
# Validates: Requirements 2.2, 2.3
# ---------------------------------------------------------------------------


@composite
def csv_rows_with_invalids(draw: Any) -> list[dict[str, str]]:
    """Generate a list of CSV row dicts, some with empty SMILES or unknown solvents.

    Strategy: for each row, randomly decide if it should be valid, have
    empty SMILES, or have an unknown solvent.
    """
    n = draw(integers(min_value=1, max_value=10))
    rows: list[dict[str, str]] = []
    for i in range(n):
        kind = draw(sampled_from(["valid", "empty_smiles", "unknown_solvent"]))
        if kind == "valid":
            rows.append({
                "name": f"mol_{i}",
                "solvent": draw(sampled_from(_KNOWN_SOLVENTS)),
                "abs": "400.0",
                "em": "450.0",
                "epsilon": "25000.0",
                "mw": "200.0",
                "plqy": "0.5",
                "smiles": draw(sampled_from(SAMPLE_SMILES)),
            })
        elif kind == "empty_smiles":
            rows.append({
                "name": f"empty_{i}",
                "solvent": draw(sampled_from(_KNOWN_SOLVENTS)),
                "abs": "400.0",
                "em": "450.0",
                "epsilon": "25000.0",
                "mw": "200.0",
                "plqy": "0.5",
                "smiles": "",
            })
        else:
            rows.append({
                "name": f"badsol_{i}",
                "solvent": draw(sampled_from(_UNKNOWN_SOLVENTS)),
                "abs": "400.0",
                "em": "450.0",
                "epsilon": "25000.0",
                "mw": "200.0",
                "plqy": "0.5",
                "smiles": draw(sampled_from(SAMPLE_SMILES)),
            })
    return rows


@settings(max_examples=50)
@given(rows=csv_rows_with_invalids())
def test_invalid_rows_skipped_with_reasons(
    rows: list[dict[str, str]],
) -> None:
    """For any CSV with empty SMILES or unknown solvents, all invalid rows
    appear in the skipped list with reasons.

    **Validates: Requirements 2.2, 2.3**
    """
    import tempfile

    csv_content = _make_csv_content(rows)
    solvent_mapping = _get_solvent_mapping()
    patterns = _get_substructure_patterns()

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "test_input.csv"
        csv_file.write_text(csv_content)

        main_df, smiles_fp_df, sol_fp_df, skipped = process_input_csv(
            str(csv_file), solvent_mapping, patterns
        )

    # Count expected invalids
    expected_empty = sum(1 for r in rows if r["smiles"].strip() == "")
    expected_unknown_sol = sum(
        1 for r in rows
        if r["smiles"].strip() != ""
        and map_solvent_name_to_smiles(r["solvent"]) is None
    )

    # All empty-SMILES rows must be skipped
    skipped_names = [name for name, _ in skipped]
    for r in rows:
        if r["smiles"].strip() == "":
            assert r["name"] in skipped_names, (
                f"Row '{r['name']}' with empty SMILES should be skipped"
            )
        elif map_solvent_name_to_smiles(r["solvent"]) is None:
            assert r["name"] in skipped_names, (
                f"Row '{r['name']}' with unknown solvent should be skipped"
            )

    # Total skipped >= empty + unknown (could be more if SMILES is invalid)
    assert len(skipped) >= expected_empty + expected_unknown_sol


# ---------------------------------------------------------------------------
# Property 5: Epsilon to k conversion
# Validates: Requirements 2.9
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(epsilon=floats(min_value=0.001, max_value=1e10))
def test_epsilon_to_k_conversion(epsilon: float) -> None:
    """For any positive float epsilon, k == log10(epsilon) within tolerance 1e-10.

    **Validates: Requirements 2.9**
    """
    import tempfile

    csv_content = _make_csv_content([{
        "name": "test_mol",
        "solvent": "EtOH",
        "abs": "400.0",
        "em": "450.0",
        "epsilon": str(epsilon),
        "mw": "200.0",
        "plqy": "0.5",
        "smiles": "c1ccccc1",  # benzene - always valid
    }])

    solvent_mapping = _get_solvent_mapping()
    patterns = _get_substructure_patterns()

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "eps_test.csv"
        csv_file.write_text(csv_content)
        main_df, _, _, skipped = process_input_csv(
            str(csv_file), solvent_mapping, patterns
        )

    assert len(skipped) == 0, f"Row should not be skipped: {skipped}"
    assert len(main_df) == 1, "Expected exactly 1 row"

    k_val = main_df["k"].iloc[0]
    expected_k = np.log10(epsilon)
    assert abs(k_val - expected_k) < 1e-10, (
        f"k={k_val} != log10({epsilon})={expected_k}"
    )


# ---------------------------------------------------------------------------
# Property 6: Output DataFrame dimensions
# Validates: Requirements 2.10
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    smiles_choices=lists(
        sampled_from(SAMPLE_SMILES),
        min_size=1,
        max_size=5,
    ),
)
def test_output_dataframe_dimensions(
    smiles_choices: list[str],
) -> None:
    """For valid processed input, main_df has 152 columns, FP DFs have
    1024 columns each, and all have the same row count.

    **Validates: Requirements 2.10**
    """
    import tempfile

    rows = []
    for i, smi in enumerate(smiles_choices):
        rows.append({
            "name": f"mol_{i}",
            "solvent": "EtOH",
            "abs": "400.0",
            "em": "450.0",
            "epsilon": "25000.0",
            "mw": "200.0",
            "plqy": "0.5",
            "smiles": smi,
        })

    csv_content = _make_csv_content(rows)
    solvent_mapping = _get_solvent_mapping()
    patterns = _get_substructure_patterns()

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "dim_test.csv"
        csv_file.write_text(csv_content)
        main_df, smiles_fp_df, sol_fp_df, skipped = process_input_csv(
            str(csv_file), solvent_mapping, patterns
        )

    # Only check if we got at least one valid row
    n_valid = len(main_df)
    if n_valid == 0:
        return  # all rows were invalid SMILES, nothing to check

    assert main_df.shape[1] == 152, (
        f"main_df should have 152 columns, got {main_df.shape[1]}"
    )
    assert smiles_fp_df.shape[1] == 1024, (
        f"smiles_fp_df should have 1024 columns, got {smiles_fp_df.shape[1]}"
    )
    assert sol_fp_df.shape[1] == 1024, (
        f"sol_fp_df should have 1024 columns, got {sol_fp_df.shape[1]}"
    )
    # All three must have the same row count
    assert main_df.shape[0] == smiles_fp_df.shape[0] == sol_fp_df.shape[0], (
        f"Row count mismatch: main={main_df.shape[0]}, "
        f"smiles_fp={smiles_fp_df.shape[0]}, sol_fp={sol_fp_df.shape[0]}"
    )


# ---------------------------------------------------------------------------
# Property 7: Property-specific NaN filtering
# Validates: Requirements 2.11
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(prop=sampled_from(["abs", "em", "plqy", "k"]))
def test_property_specific_nan_filtering(prop: str) -> None:
    """For any property, filtered output has zero NaN in that property column.
    Row count equals non-NaN count in source.

    **Validates: Requirements 2.11**
    """
    # Build a small DataFrame inline (avoids function-scoped fixture issue)
    n_rows = 5
    main_data: dict[str, Any] = {
        "split": ["train"] * n_rows,
        "smiles": SAMPLE_SMILES[:n_rows],
        "solvent": ["CCO"] * n_rows,
        "abs": [400.0, 450.0, np.nan, 500.0, 350.0],
        "em": [450.0, np.nan, 550.0, 600.0, 400.0],
        "plqy": [0.5, 0.3, 0.8, np.nan, 0.1],
        "k": [4.0, 3.5, np.nan, 4.5, 3.0],
        "tag_name": ["PAHs"] * n_rows,
        "solvent_num": [2] * n_rows,
        "tag": [8] * n_rows,
        "MW": [200.0] * n_rows,
        "LogP": [3.0] * n_rows,
        "TPSA": [20.0] * n_rows,
        "Double_Bond_Count": [5.0] * n_rows,
        "Ring_Count": [2.0] * n_rows,
        "unimol_plus": [3.0] * n_rows,
    }
    for i in range(1, 137):
        main_data[f"fragment_{i}"] = [0] * n_rows
    main_df = pd.DataFrame(main_data)

    fp_cols = [str(i) for i in range(1024)]
    smiles_fp_df = pd.DataFrame(
        np.zeros((n_rows, 1024), dtype=np.int32), columns=fp_cols
    )
    sol_fp_df = pd.DataFrame(
        np.zeros((n_rows, 1024), dtype=np.int32), columns=fp_cols
    )

    result = split_by_property(main_df, smiles_fp_df, sol_fp_df)

    if prop not in result:
        return  # property column not present

    filtered_main, filtered_smiles, filtered_sol = result[prop]

    # Zero NaN in the property column
    nan_count = filtered_main[prop].isna().sum()
    assert nan_count == 0, (
        f"Filtered {prop} DataFrame has {nan_count} NaN values"
    )

    # Row count matches non-NaN count in source
    expected_rows = main_df[prop].notna().sum()
    assert len(filtered_main) == expected_rows, (
        f"Expected {expected_rows} rows for {prop}, got {len(filtered_main)}"
    )

    # All three DFs have same row count
    assert len(filtered_main) == len(filtered_smiles) == len(filtered_sol), (
        f"Row count mismatch in split for {prop}"
    )


# ---------------------------------------------------------------------------
# Property 8: Merge preserves data integrity
# Validates: Requirements 3.1, 3.2
# ---------------------------------------------------------------------------


@composite
def random_training_dfs(draw: Any) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate a random triple of (main, smiles_fp, sol_fp) DataFrames.

    Keeps it simple: random numeric data with correct column counts.
    """
    n_rows = draw(integers(min_value=1, max_value=5))

    # Build main_df with 152 columns
    main_data: dict[str, Any] = {
        "split": ["train"] * n_rows,
        "smiles": ["c1ccccc1"] * n_rows,
        "solvent": ["CCO"] * n_rows,
        "abs": [400.0] * n_rows,
        "em": [450.0] * n_rows,
        "plqy": [0.5] * n_rows,
        "k": [4.0] * n_rows,
        "tag_name": ["PAHs"] * n_rows,
        "solvent_num": [2] * n_rows,
        "tag": [8] * n_rows,
        "MW": [200.0] * n_rows,
        "LogP": [3.0] * n_rows,
        "TPSA": [20.0] * n_rows,
        "Double_Bond_Count": [5] * n_rows,
        "Ring_Count": [2] * n_rows,
        "unimol_plus": [3.0] * n_rows,
    }
    for i in range(1, 137):
        main_data[f"fragment_{i}"] = [0] * n_rows

    main_df = pd.DataFrame(main_data)

    # FP DataFrames with named columns
    fp_cols = [f"fp_{i}" for i in range(1024)]
    smiles_fp = pd.DataFrame(
        np.zeros((n_rows, 1024), dtype=np.int32), columns=fp_cols
    )
    sol_fp = pd.DataFrame(
        np.zeros((n_rows, 1024), dtype=np.int32), columns=fp_cols
    )

    return main_df, smiles_fp, sol_fp


@settings(max_examples=100)
@given(
    new_dfs=random_training_dfs(),
    existing_dfs=random_training_dfs(),
)
def test_merge_preserves_data_integrity(
    new_dfs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    existing_dfs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """Merged row count = sum of input row counts. Column names match
    existing DataFrame.

    **Validates: Requirements 3.1, 3.2**
    """
    new_main, new_smiles, new_sol = new_dfs
    ex_main, ex_smiles, ex_sol = existing_dfs

    merged_main, merged_smiles, merged_sol = merge_training_data(
        new_main, new_smiles, new_sol,
        ex_main, ex_smiles, ex_sol,
    )

    expected_rows = len(new_main) + len(ex_main)
    assert len(merged_main) == expected_rows, (
        f"Expected {expected_rows} rows, got {len(merged_main)}"
    )
    assert len(merged_smiles) == expected_rows
    assert len(merged_sol) == expected_rows

    # Column names of merged FP DFs match existing
    assert list(merged_smiles.columns) == list(ex_smiles.columns), (
        "Merged smiles FP columns should match existing"
    )
    assert list(merged_sol.columns) == list(ex_sol.columns), (
        "Merged solvent FP columns should match existing"
    )


# ===========================================================================
# Unit tests for CSV processing and merging (Task 5.8)
# Requirements: 2.1, 2.2, 2.3, 2.9, 2.10, 2.11, 3.1, 3.2
# ===========================================================================


class TestProcessInputCSV:
    """Unit tests for process_input_csv."""

    def test_mixed_valid_invalid_rows(self, tmp_path: Path) -> None:
        """CSV with mixed valid/invalid rows produces correct skip counts."""
        csv_content = _make_csv_content([
            {
                "name": "valid_1",
                "solvent": "EtOH",
                "abs": "400.0",
                "em": "450.0",
                "epsilon": "25000.0",
                "mw": "200.0",
                "plqy": "0.5",
                "smiles": "c1ccccc1",
            },
            {
                "name": "empty_smiles",
                "solvent": "EtOH",
                "abs": "400.0",
                "em": "450.0",
                "epsilon": "10000.0",
                "mw": "150.0",
                "plqy": "0.3",
                "smiles": "",
            },
            {
                "name": "bad_solvent",
                "solvent": "FakeSolvent",
                "abs": "400.0",
                "em": "450.0",
                "epsilon": "10000.0",
                "mw": "150.0",
                "plqy": "0.3",
                "smiles": "CCO",
            },
            {
                "name": "invalid_smiles",
                "solvent": "DCM",
                "abs": "400.0",
                "em": "450.0",
                "epsilon": "10000.0",
                "mw": "150.0",
                "plqy": "0.3",
                "smiles": "NOT_A_SMILES",
            },
            {
                "name": "valid_2",
                "solvent": "Wasser",
                "abs": "500.0",
                "em": "550.0",
                "epsilon": "50000.0",
                "mw": "300.0",
                "plqy": "0.8",
                "smiles": "CCO",
            },
        ])
        csv_file = tmp_path / "mixed.csv"
        csv_file.write_text(csv_content)

        solvent_mapping = _get_solvent_mapping()
        patterns = _get_substructure_patterns()

        main_df, smiles_fp, sol_fp, skipped = process_input_csv(
            str(csv_file), solvent_mapping, patterns
        )

        # 2 valid rows, 3 skipped (empty SMILES, bad solvent, invalid SMILES)
        assert len(main_df) == 2, f"Expected 2 valid rows, got {len(main_df)}"
        assert len(skipped) == 3, f"Expected 3 skipped, got {len(skipped)}"

        skipped_names = [name for name, _ in skipped]
        assert "empty_smiles" in skipped_names
        assert "bad_solvent" in skipped_names
        assert "invalid_smiles" in skipped_names

    def test_empty_csv(self, tmp_path: Path) -> None:
        """Empty CSV (header only, no data rows) returns empty DataFrames."""
        csv_content = "name,solvent,abs,em,epsilon,mw,plqy,smiles\n"
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text(csv_content)

        solvent_mapping = _get_solvent_mapping()
        patterns = _get_substructure_patterns()

        main_df, smiles_fp, sol_fp, skipped = process_input_csv(
            str(csv_file), solvent_mapping, patterns
        )

        assert len(main_df) == 0
        assert len(smiles_fp) == 0
        assert len(sol_fp) == 0
        assert len(skipped) == 0
        # Column counts should still be correct
        assert main_df.shape[1] == 152
        assert smiles_fp.shape[1] == 1024
        assert sol_fp.shape[1] == 1024

    def test_epsilon_zero_gives_nan_k(self, tmp_path: Path) -> None:
        """Epsilon=0 should produce k=NaN (log10(0) is undefined)."""
        csv_content = _make_csv_content([{
            "name": "eps_zero",
            "solvent": "EtOH",
            "abs": "400.0",
            "em": "450.0",
            "epsilon": "0",
            "mw": "200.0",
            "plqy": "0.5",
            "smiles": "c1ccccc1",
        }])
        csv_file = tmp_path / "eps_zero.csv"
        csv_file.write_text(csv_content)

        solvent_mapping = _get_solvent_mapping()
        patterns = _get_substructure_patterns()

        main_df, _, _, skipped = process_input_csv(
            str(csv_file), solvent_mapping, patterns
        )

        assert len(main_df) == 1
        assert np.isnan(main_df["k"].iloc[0]), "k should be NaN for epsilon=0"

    def test_epsilon_negative_gives_nan_k(self, tmp_path: Path) -> None:
        """Epsilon<0 should produce k=NaN (log10 of negative is undefined)."""
        csv_content = _make_csv_content([{
            "name": "eps_neg",
            "solvent": "EtOH",
            "abs": "400.0",
            "em": "450.0",
            "epsilon": "-100",
            "mw": "200.0",
            "plqy": "0.5",
            "smiles": "c1ccccc1",
        }])
        csv_file = tmp_path / "eps_neg.csv"
        csv_file.write_text(csv_content)

        solvent_mapping = _get_solvent_mapping()
        patterns = _get_substructure_patterns()

        main_df, _, _, skipped = process_input_csv(
            str(csv_file), solvent_mapping, patterns
        )

        assert len(main_df) == 1
        assert np.isnan(main_df["k"].iloc[0]), "k should be NaN for epsilon<0"

    def test_missing_required_columns_raises(self, tmp_path: Path) -> None:
        """CSV missing required columns raises ValueError."""
        # CSV with only 'name' and 'smiles' columns -- missing others
        csv_content = "name,smiles\nmol1,c1ccccc1\n"
        csv_file = tmp_path / "bad_cols.csv"
        csv_file.write_text(csv_content)

        solvent_mapping = _get_solvent_mapping()
        patterns = _get_substructure_patterns()

        with pytest.raises(ValueError, match="Missing required columns"):
            process_input_csv(str(csv_file), solvent_mapping, patterns)


class TestMergeTrainingData:
    """Unit tests for merge_training_data."""

    def test_merge_two_small_dataframes(self) -> None:
        """Merge of two small DataFrames produces correct row count."""
        cols_main = (
            ["split", "smiles", "solvent", "abs", "em", "plqy", "k",
             "tag_name", "solvent_num", "tag",
             "MW", "LogP", "TPSA", "Double_Bond_Count", "Ring_Count",
             "unimol_plus"]
            + [f"fragment_{i}" for i in range(1, 137)]
        )
        fp_cols = [str(i) for i in range(1024)]

        def _make_df(n: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            """Create a triple of DataFrames with n rows."""
            data: dict[str, Any] = {"split": ["train"] * n}
            data["smiles"] = ["c1ccccc1"] * n
            data["solvent"] = ["CCO"] * n
            for c in ["abs", "em", "plqy", "k"]:
                data[c] = [1.0] * n
            data["tag_name"] = ["PAHs"] * n
            data["solvent_num"] = [2] * n
            data["tag"] = [8] * n
            for c in ["MW", "LogP", "TPSA", "Double_Bond_Count",
                       "Ring_Count", "unimol_plus"]:
                data[c] = [1.0] * n
            for i in range(1, 137):
                data[f"fragment_{i}"] = [0] * n
            main = pd.DataFrame(data, columns=cols_main)
            smiles_fp = pd.DataFrame(
                np.zeros((n, 1024), dtype=np.int32), columns=fp_cols
            )
            sol_fp = pd.DataFrame(
                np.zeros((n, 1024), dtype=np.int32), columns=fp_cols
            )
            return main, smiles_fp, sol_fp

        new_main, new_smiles, new_sol = _make_df(3)
        ex_main, ex_smiles, ex_sol = _make_df(4)

        merged_main, merged_smiles, merged_sol = merge_training_data(
            new_main, new_smiles, new_sol,
            ex_main, ex_smiles, ex_sol,
        )

        assert len(merged_main) == 7, f"Expected 7 rows, got {len(merged_main)}"
        assert len(merged_smiles) == 7
        assert len(merged_sol) == 7
        assert list(merged_smiles.columns) == fp_cols
        assert list(merged_sol.columns) == fp_cols
