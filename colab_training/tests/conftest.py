"""Shared test fixtures for fluor_modules tests.

Provides:
- sample_smiles: ~20 known-valid SMILES from FluoDB dataset
- mock DataFrames for data pipeline testing
- temp directory fixture
- small model factory fixture (graph_feat_size=32)
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixture 1: Known-valid SMILES from Fluor-RLAT/data/00_FluoDB.csv
# Covers scaffold families: PAHs, BODIPY, Coumarin, Cyanine, Acridines,
# Carbazole, Triphenylamine, 5p6, 6n6, 5n6, 6p6
# ---------------------------------------------------------------------------

SAMPLE_SMILES: list[str] = [
    # PAHs
    "CN(C)c1ccc(C#Cc2ccc3ccc4c(C#Cc5ccc(N(C)C)cc5)ccc5ccc2c3c54)cc1",
    "CC(=O)c1ccc2cc(C#Cc3cn([C@H]4C[C@H](O)[C@@H](CO)O4)c4ncnc(N)c34)ccc2c1",
    # BODIPY
    "C#CC1=C(C)C2=C(C)c3c(C)c(C#C)c(C)n3[B-](F)(F)[N+]2=C1C",
    "F[B-]1(F)n2c(ccc2N2CCCCC2)C(c2c(Cl)cccc2Cl)=C2C=CC=[N+]21",
    # Coumarin
    "COc1ccc2c(c1)oc(=O)c1ccccc12",
    # Cyanine
    "CCCCN1C(=O)C2=C(c3ccc(-c4ccccc4)cc3)N=C(O)C2=C1c1ccc(-c2ccccc2)cc1",
    # Acridines
    "Cc1ccc(C(=O)Oc2cccc3c2C(=O)c2cccc(OC(=O)c4ccc(C)cc4)c2C3=O)cc1",
    "CCN(CC)c1ccc2c(c1)Oc1cc(N(CC)CC)ccc1C2=C1C=CC(OC)=CC1=[O+]C",
    # Carbazole
    "CCn1c2ccccc2c2cc(C3=[O+][B-](F)(F)OC(C)=C3)ccc21",
    # 5p6
    "COc1ccccc1-c1nc(-c2ccccc2)c2ccccn12",
    "CC(C)(C)c1ccc2c(c1)c1cccc3cnc2n31",
    "COc1cc2[nH]c3c(c2cc1OCC(=O)O)CCCC3=O",
    # 6n6
    "CC(C)(C)c1ccc2ccc(C(c3ccccc3)c3ccccc3)c-2cc1",
    "Cc1cc(O)c(C(C)C)cc1/N=C/c1ccc(C(F)(F)F)cc1",
    # 5n6
    "Fc1ccc(-c2nc3n(c2-c2ccncc2)CCS3)cc1",
    # 6p6
    "COc1ccc(-c2cc(-c3ccc(C)cc3)nc3c(C#N)c(N4CCCC4)[nH]c(=N)c23)cc1",
    # Triphenylamine
    "N#CC(C#N)=C1C=C(C=Cc2ccc(N(c3ccccc3)c3ccccc3)cc2)OC(C=Cc2ccc(N(c3ccccc3)c3ccccc3)cc2)=C1",
    # Additional simple/small molecules for fast tests
    "c1ccccc1",  # benzene
    "CCO",  # ethanol
    "CC(=O)O",  # acetic acid
]


@pytest.fixture
def sample_smiles() -> list[str]:
    """Return ~20 known-valid SMILES strings from the FluoDB dataset.

    Covers diverse scaffold families: PAHs, BODIPY, Coumarin, Cyanine,
    Acridines, Carbazole, Triphenylamine, 5p6, 6n6, 5n6, 6p6.
    Also includes a few simple molecules (benzene, ethanol, acetic acid).
    """
    return list(SAMPLE_SMILES)


# ---------------------------------------------------------------------------
# Fixture 2: Mock DataFrames for data pipeline testing
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_input_df() -> pd.DataFrame:
    """Return a small mock DataFrame matching the user input CSV schema.

    Columns: name, solvent, abs, em, epsilon, mw, plqy, smiles.
    Includes some rows with missing values and invalid data to test
    skip logic in process_input_csv.
    """
    return pd.DataFrame({
        "name": [
            "Coumarin-1", "BODIPY-1", "Rhodamine-B", "Bad-SMILES",
            "No-Solvent", "Pyrene-1",
        ],
        "solvent": [
            "EtOH", "DCM", "Wasser", "Toluol",
            "UnknownSolvent", "Aceton",
        ],
        "abs": [400.0, 527.0, 554.0, 350.0, 420.0, 334.0],
        "em": [450.0, 552.0, 580.0, 400.0, 470.0, 394.0],
        "epsilon": [25000.0, 41020.0, 106000.0, 10000.0, 30000.0, 54000.0],
        "mw": [226.2, 310.2, 479.0, 150.0, 300.0, 202.3],
        "plqy": [0.5, 0.54, 0.65, 0.1, 0.3, 0.32],
        "smiles": [
            "COc1ccc2c(c1)oc(=O)c1ccccc12",  # coumarin
            "C#CC1=C(C)C2=C(C)c3c(C)c(C#C)c(C)n3[B-](F)(F)[N+]2=C1C",  # BODIPY
            "CCN(CC)c1ccc2c(-c3ccccc3C(=O)O)c3ccc(=[N+](CC)CC)cc-3oc2c1",  # rhodamine
            "NOT_A_SMILES",  # invalid
            "c1ccccc1",  # benzene (valid SMILES but unknown solvent)
            "c1cc2ccc3cccc4ccc(c1)c2c34",  # pyrene
        ],
    })


@pytest.fixture
def mock_main_df() -> pd.DataFrame:
    """Return a small mock main DataFrame with 152 columns.

    Mimics the structure produced by process_input_csv:
    split, smiles, solvent, abs, em, plqy, k, tag_name, solvent_num, tag,
    MW, LogP, TPSA, Double_Bond_Count, Ring_Count, unimol_plus,
    fragment_1..fragment_136.
    """
    n_rows = 5
    data: dict[str, Any] = {
        "split": ["train"] * n_rows,
        "smiles": SAMPLE_SMILES[:n_rows],
        "solvent": ["CCO"] * n_rows,
        "abs": [400.0, 450.0, np.nan, 500.0, 350.0],
        "em": [450.0, np.nan, 550.0, 600.0, 400.0],
        "plqy": [0.5, 0.3, 0.8, np.nan, 0.1],
        "k": [4.0, 3.5, np.nan, 4.5, 3.0],
        "tag_name": ["PAHs", "PAHs", "BODIPY", "BODIPY", "Coumarin"],
        "solvent_num": [2, 2, 2, 2, 2],
        "tag": [8, 8, 5, 5, 2],
        "MW": [488.6, 442.5, 310.2, 420.1, 226.2],
        "LogP": [7.5, 2.4, 3.5, 5.4, 3.0],
        "TPSA": [6.5, 123.5, 7.9, 11.2, 39.4],
        "Double_Bond_Count": [31, 22, 8, 14, 17],
        "Ring_Count": [6, 5, 3, 5, 3],
        "unimol_plus": [2.8, 3.4, 3.0, 3.5, 4.4],
    }
    # Add 136 fragment columns
    for i in range(1, 137):
        data[f"fragment_{i}"] = np.random.randint(0, 2, size=n_rows).tolist()
    return pd.DataFrame(data)


@pytest.fixture
def mock_smiles_fp_df() -> pd.DataFrame:
    """Return a small mock molecule fingerprint DataFrame (1024 columns)."""
    n_rows = 5
    data = {
        f"fp_{i}": np.random.randint(0, 2, size=n_rows).tolist()
        for i in range(1024)
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_solvent_fp_df() -> pd.DataFrame:
    """Return a small mock solvent fingerprint DataFrame (1024 columns)."""
    n_rows = 5
    data = {
        f"sol_fp_{i}": np.random.randint(0, 2, size=n_rows).tolist()
        for i in range(1024)
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Fixture 3: Temp directory
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for test file I/O.

    Uses pytest's built-in tmp_path fixture for automatic cleanup.
    """
    return tmp_path


# ---------------------------------------------------------------------------
# Fixture 4: Small model factory (graph_feat_size=32 for fast tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def small_model_factory() -> dict[str, Any]:
    """Return parameters for building small test models.

    Uses graph_feat_size=32 (instead of production 256) to keep tests fast.
    Assumption: small models exercise the same code paths as production models,
    just with fewer parameters.
    """
    return {
        "graph_feat_size": 32,
        "node_feat_size": 39,   # typical DGLLife canonical atom featurizer dim
        "edge_feat_size": 12,   # typical DGLLife canonical bond featurizer dim
        "fp_size": 2192,        # full feature vector dimension
        "solvent_dim": 1024,    # solvent fingerprint dimension
        "smiles_extra_dim": 1168,  # 1024 + 8 + 136
    }
