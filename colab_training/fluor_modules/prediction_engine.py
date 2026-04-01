"""Prediction engine for fluorescent molecule property prediction.

Handles scaler fitting, feature vector assembly, model loading, and inference.
Depends on: models.py, data_pipeline.py, config.py.

Reference: design.md section "prediction_engine.py"
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dgllife.utils import AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
from dgllife.utils import smiles_to_bigraph
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from fluor_modules.config import (
    COMPLETED_RUNS_DIR,
    FEATURE_VECTOR_DIM,
    MODEL_REGISTRY,
    MORGAN_NBITS,
    MORGAN_RADIUS,
    NUMERIC_FEATURE_COUNT,
    NUM_SCAFFOLD_FLAGS,
    PROPERTIES,
)
from fluor_modules.data_pipeline import (
    compute_morgan_fingerprint,
    compute_molecular_descriptors,
    compute_scaffold_flags,
    map_solvent_smiles_to_id,
)
from fluor_modules.models import build_model

logger = logging.getLogger(__name__)

# Path to the pretrained models bundled in the repository.
# Assumption: the repo root contains a Fluor-RLAT/ folder with Model_*.pth files.
REPO_MODEL_DIR: str = "Fluor-RLAT"

# AttentiveFP featurizers (module-level singletons to avoid repeated init).
_ATOM_FEATURIZER = AttentiveFPAtomFeaturizer(atom_data_field="hv")
_BOND_FEATURIZER = AttentiveFPBondFeaturizer(bond_data_field="he")


# ---------------------------------------------------------------------------
# Scaler fitting
# ---------------------------------------------------------------------------


def fit_scalers(
    train_df: pd.DataFrame, target: str
) -> tuple[StandardScaler, MinMaxScaler]:
    """Fit StandardScaler on labels and MinMaxScaler on numeric features.

    The training DataFrame has 152 columns. Columns at positions 8:16
    (0-indexed) are the 8 numeric features:
      solvent_num, tag, MW, LogP, TPSA, Double_Bond_Count, Ring_Count, unimol_plus

    Args:
        train_df: Training DataFrame with 152 columns.
        target: Property name (abs, em, plqy, k).

    Returns:
        (label_scaler, num_scaler) tuple.
    """
    label_scaler = StandardScaler()
    label_scaler.fit(train_df[[target]].values)

    num_scaler = MinMaxScaler()
    num_features = train_df.iloc[:, 8:16].values
    num_scaler.fit(num_features)

    return label_scaler, num_scaler


# ---------------------------------------------------------------------------
# Feature vector assembly
# ---------------------------------------------------------------------------


def build_feature_vector(
    mol_smiles: str,
    sol_smiles: str,
    num_scaler: MinMaxScaler,
    solvent_mapping: dict[str, int],
    substructure_patterns: list,
) -> np.ndarray | None:
    """Assemble the 2192-dim feature vector for a molecule/solvent pair.

    Layout: sol_fp(1024) + mol_fp(1024) + scaled_numeric(8) + scaffold_flags(136)

    The 8 numeric features (in order) are:
      solvent_num, tag, MW, LogP, TPSA, Double_Bond_Count, Ring_Count, unimol_plus

    Assumption: unimol_plus is not computed here; it is set to 0 as a placeholder
    since it requires an external model not available at inference time.

    Args:
        mol_smiles: Molecule SMILES string.
        sol_smiles: Solvent SMILES string.
        num_scaler: Fitted MinMaxScaler for the 8 numeric features.
        solvent_mapping: Dict mapping solvent SMILES to numeric IDs.
        substructure_patterns: Compiled SMARTS patterns for scaffold flags.

    Returns:
        Numpy array of shape (2192,), or None if mol_smiles is invalid.
    """
    # Solvent fingerprint
    sol_fp = compute_morgan_fingerprint(sol_smiles, radius=MORGAN_RADIUS, n_bits=MORGAN_NBITS)
    if sol_fp is None:
        logger.warning("Invalid solvent SMILES: %s", sol_smiles)
        sol_fp = np.zeros(MORGAN_NBITS, dtype=np.int32)

    # Molecule fingerprint
    mol_fp = compute_morgan_fingerprint(mol_smiles, radius=MORGAN_RADIUS, n_bits=MORGAN_NBITS)
    if mol_fp is None:
        logger.warning("Invalid molecule SMILES: %s", mol_smiles)
        return None

    # Molecular descriptors
    desc = compute_molecular_descriptors(mol_smiles)
    if desc is None:
        logger.warning("Could not compute descriptors for: %s", mol_smiles)
        return None

    # Numeric features: solvent_num, tag, MW, LogP, TPSA, Double_Bond_Count, Ring_Count, unimol_plus
    solvent_num = float(map_solvent_smiles_to_id(sol_smiles, solvent_mapping) or 0)
    # tag is the scaffold tag; compute it from detect_scaffold
    from fluor_modules.data_pipeline import detect_scaffold
    tag, _ = detect_scaffold(mol_smiles)
    tag_val = float(tag if tag >= 0 else 0)

    raw_numeric = np.array([
        solvent_num,
        tag_val,
        desc["MW"],
        desc["LogP"],
        desc["TPSA"],
        desc["Double_Bond_Count"],
        desc["Ring_Count"],
        0.0,  # unimol_plus placeholder (not available at inference time)
    ], dtype=np.float64).reshape(1, -1)

    scaled_numeric = num_scaler.transform(raw_numeric).flatten().astype(np.float32)

    # Scaffold flags
    flags = compute_scaffold_flags(mol_smiles, substructure_patterns).astype(np.float32)

    # Concatenate: sol_fp + mol_fp + scaled_numeric + flags
    feature_vector = np.concatenate([
        sol_fp.astype(np.float32),
        mol_fp.astype(np.float32),
        scaled_numeric,
        flags,
    ])

    assert feature_vector.shape == (FEATURE_VECTOR_DIM,), (
        f"Expected {FEATURE_VECTOR_DIM} dims, got {feature_vector.shape}"
    )
    return feature_vector


# ---------------------------------------------------------------------------
# Single-molecule inference
# ---------------------------------------------------------------------------


def predict_single(
    mol_smiles: str,
    sol_smiles: str,
    model: nn.Module,
    model_class_name: str,
    feature_vector: np.ndarray,
    label_scaler: StandardScaler,
    device: str,
) -> float:
    """Run inference for one molecule on one target property.

    For GraphFingerprintsModel (abs/em): split feature_vector at dim 1024
    into solvent (first 1024) and smiles+extra (remaining 1168).
    For GraphFingerprintsModelFC (plqy/k): pass full 2192-dim vector.

    Args:
        mol_smiles: Molecule SMILES string.
        sol_smiles: Solvent SMILES string (unused here, kept for API consistency).
        model: Loaded nn.Module in eval mode.
        model_class_name: "GraphFingerprintsModel" or "GraphFingerprintsModelFC".
        feature_vector: 2192-dim numpy array from build_feature_vector.
        label_scaler: Fitted StandardScaler for inverse-transforming predictions.
        device: "cpu" or "cuda".

    Returns:
        Predicted value on the original (unscaled) scale.
    """
    model.eval()

    # Build DGL graph from molecule SMILES
    graph = smiles_to_bigraph(
        mol_smiles,
        node_featurizer=_ATOM_FEATURIZER,
        edge_featurizer=_BOND_FEATURIZER,
    )
    graph = graph.to(device)
    node_feats = graph.ndata["hv"].to(device)
    edge_feats = graph.edata.get("he")
    if edge_feats is not None:
        edge_feats = edge_feats.to(device)

    # Prepare fingerprint tensor
    fp_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(graph, node_feats, edge_feats, fp_tensor)

    # Inverse-transform to original scale
    pred_np = pred.cpu().numpy().reshape(-1, 1)
    result = label_scaler.inverse_transform(pred_np)
    return float(result[0, 0])


# ---------------------------------------------------------------------------
# Multi-property prediction
# ---------------------------------------------------------------------------


def predict_all_properties(
    mol_smiles: str,
    sol_smiles: str,
    model_dir: str,
    data_dir: str,
    device: str,
) -> dict[str, float | str]:
    """Predict abs, em, plqy, k for a molecule/solvent pair.

    For each property:
    1. Load train data from data_dir/train_{property}.csv
    2. Fit scalers
    3. Build feature vector
    4. Load model from model_dir/Model_{property}.pth
    5. Run inference

    Args:
        mol_smiles: Molecule SMILES string.
        sol_smiles: Solvent SMILES string.
        model_dir: Directory containing Model_{property}.pth files.
        data_dir: Directory containing train_{property}.csv files.
        device: "cpu" or "cuda".

    Returns:
        Dict mapping property name to predicted float or error string.
    """
    from fluor_modules.data_pipeline import load_solvent_mapping, load_substructure_patterns

    results: dict[str, float | str] = {}

    # Load shared resources once (assume standard paths relative to data_dir)
    solvent_mapping_path = os.path.join(data_dir, "00_solvent_mapping.csv")
    substructure_path = os.path.join(data_dir, "00_mmp_substructure.csv")

    solvent_mapping: dict[str, int] = {}
    if os.path.exists(solvent_mapping_path):
        solvent_mapping = load_solvent_mapping(solvent_mapping_path)

    substructure_patterns: list = []
    if os.path.exists(substructure_path):
        substructure_patterns = load_substructure_patterns(substructure_path)

    node_feat_size = _ATOM_FEATURIZER.feat_size("hv")
    edge_feat_size = _BOND_FEATURIZER.feat_size("he")

    for prop in PROPERTIES:
        try:
            # Load training data and fit scalers
            train_csv = os.path.join(data_dir, f"train_{prop}.csv")
            if not os.path.exists(train_csv):
                results[prop] = f"Training data not found: {train_csv}"
                continue

            train_df = pd.read_csv(train_csv)
            label_scaler, num_scaler = fit_scalers(train_df, prop)

            # Build feature vector
            fv = build_feature_vector(
                mol_smiles, sol_smiles, num_scaler, solvent_mapping, substructure_patterns
            )
            if fv is None:
                results[prop] = f"Invalid SMILES: {mol_smiles}"
                continue

            # Load model
            model_path = os.path.join(model_dir, f"Model_{prop}.pth")
            if not os.path.exists(model_path):
                results[prop] = f"Model file not found: {model_path}"
                continue

            cfg = MODEL_REGISTRY[prop]
            model_class_name = cfg["model_class"]
            model = build_model(
                target=prop,
                node_feat_size=node_feat_size,
                edge_feat_size=edge_feat_size,
                solvent_dim=MORGAN_NBITS,
                smiles_extra_dim=MORGAN_NBITS + NUMERIC_FEATURE_COUNT + NUM_SCAFFOLD_FLAGS,
                fp_size=FEATURE_VECTOR_DIM,
                graph_feat_size=256,
            )
            state = torch.load(model_path, map_location=device)
            # Support both raw state_dict and checkpoint dicts
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state)
            model.to(device)
            model.eval()

            results[prop] = predict_single(
                mol_smiles, sol_smiles, model, model_class_name, fv, label_scaler, device
            )

        except Exception as exc:
            logger.exception("Error predicting %s for %s", prop, mol_smiles)
            results[prop] = f"Error: {exc}"

    return results


# ---------------------------------------------------------------------------
# Model directory resolution
# ---------------------------------------------------------------------------


def resolve_model_dir(source: str, specific_run: str | None = None) -> str:
    """Resolve the model directory based on source type.

    Args:
        source: "pretrained" to use the bundled Fluor-RLAT/ models,
                "custom" to use a completed training run.
        specific_run: If source="custom" and this is provided, use it directly.
                      Otherwise, find the latest completed run.

    Returns:
        Path string to the model directory.
    """
    if source == "pretrained":
        return REPO_MODEL_DIR

    # source == "custom"
    if specific_run is not None:
        return specific_run

    latest = find_latest_completed_run(COMPLETED_RUNS_DIR)
    if latest is None:
        logger.warning("No completed runs found in %s", COMPLETED_RUNS_DIR)
        return COMPLETED_RUNS_DIR
    return latest


def find_latest_completed_run(completed_dir: str) -> str | None:
    """Find the most recent timestamped subfolder in completed_dir.

    Subfolders are expected to be named in YYYY-MM-DD_HH-MM-SS format.
    Lexicographic sort is sufficient because the format is ISO-like.

    Args:
        completed_dir: Path to the completed runs directory.

    Returns:
        Full path to the most recent subfolder, or None if the directory
        is empty or does not exist.
    """
    dir_path = Path(completed_dir)
    if not dir_path.exists() or not dir_path.is_dir():
        return None

    subfolders = [
        p for p in dir_path.iterdir()
        if p.is_dir()
    ]
    if not subfolders:
        return None

    # Sort lexicographically; YYYY-MM-DD_HH-MM-SS format sorts correctly
    subfolders.sort(key=lambda p: p.name)
    return str(subfolders[-1])
