"""Training engine for fluorescent molecule property prediction.

Handles TrainingConfig, LDS weights, dataset/dataloader, checkpointing,
session management, the full training loop, and archiving completed runs.

Reference: design.md section "training_engine.py"
"""

import copy
import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from dgllife.utils import (
    AttentiveFPAtomFeaturizer,
    AttentiveFPBondFeaturizer,
    smiles_to_bigraph,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset

from fluor_modules.config import GRAPH_FEAT_SIZE, MODEL_REGISTRY, MORGAN_NBITS
from fluor_modules.models import build_model

logger = logging.getLogger(__name__)

# AttentiveFP featurizers (module-level singletons).
_ATOM_FEATURIZER = AttentiveFPAtomFeaturizer(atom_data_field="hv")
_BOND_FEATURIZER = AttentiveFPBondFeaturizer(bond_data_field="he")


# ---------------------------------------------------------------------------
# Task 9.1: TrainingConfig dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """All parameters for a training run. Serializable to/from JSON.

    Attributes:
        targets: List of property names to train (e.g. ["abs", "em"]).
        epochs: Maximum number of training epochs.
        patience: Early stopping patience (epochs without improvement).
        learning_rate: Initial learning rate for Adam optimizer.
        batch_size: DataLoader batch size.
        model_configs: Per-target model hyperparameter dicts.
        data_source: Path to the data directory.
        lr_scheduler_factor: ReduceLROnPlateau reduction factor.
        lr_scheduler_patience: ReduceLROnPlateau patience.
        lr_scheduler_min: ReduceLROnPlateau minimum learning rate.
        run_id: Unique identifier for this run (UUID string).
    """

    targets: list[str]
    epochs: int
    patience: int
    learning_rate: float
    batch_size: int
    model_configs: dict[str, dict]
    data_source: str
    lr_scheduler_factor: float
    lr_scheduler_patience: int
    lr_scheduler_min: float
    run_id: str

    def to_json(self) -> str:
        """Serialize this config to a JSON string.

        Returns:
            JSON string representation of all fields.
        """
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "TrainingConfig":
        """Deserialize a TrainingConfig from a JSON string.

        Args:
            json_str: JSON string produced by to_json().

        Returns:
            TrainingConfig instance with all fields restored.
        """
        data = json.loads(json_str)
        return cls(**data)

    def matches(self, other: "TrainingConfig") -> bool:
        """Check whether this config equals another (all fields equal).

        Args:
            other: Another TrainingConfig to compare against.

        Returns:
            True if all fields are equal, False otherwise.
        """
        return asdict(self) == asdict(other)


# ---------------------------------------------------------------------------
# Task 9.2: LDS weights, MolecularDataset, collate_fn
# ---------------------------------------------------------------------------


def compute_lds_weights(
    labels: np.ndarray,
    alpha: float = 0.5,
    kernel_size: int = 5,
) -> np.ndarray:
    """Compute Label Distribution Smoothing (LDS) sample weights.

    When alpha=0, returns uniform weights of 1.0 for all samples.
    For alpha > 0, uses KernelDensity (sklearn) to estimate the density
    of the label distribution, then weights = 1 / density^alpha,
    normalized so the mean weight equals 1.0.

    Args:
        labels: 1-D numpy array of float target values.
        alpha: Reweighting exponent. 0 = uniform, higher = stronger reweighting.
        kernel_size: KDE bandwidth (controls smoothness of density estimate).

    Returns:
        Numpy array of float32 weights with the same length as labels.
    """
    n = len(labels)
    if alpha == 0 or n == 0:
        return np.ones(n, dtype=np.float32)

    values = np.array(labels, dtype=np.float64).reshape(-1, 1)
    kde = KernelDensity(kernel="gaussian", bandwidth=float(kernel_size)).fit(values)
    log_densities = kde.score_samples(values)
    densities = np.exp(log_densities)
    # Avoid division by zero for degenerate distributions
    densities = np.clip(densities, 1e-12, None)
    weights = 1.0 / (densities ** alpha)
    weights = weights / np.mean(weights)
    return weights.astype(np.float32)


class MolecularDataset(Dataset):
    """PyTorch Dataset wrapping DGL graphs, fingerprints, labels, masks, weights.

    Each item is a tuple: (graph, fingerprint, label, mask, weight).

    Args:
        graphs: List of DGL graphs.
        fingerprints: Float32 tensor of shape [N, fp_dim].
        labels: Float32 tensor of shape [N, 1].
        masks: Float32 tensor of shape [N, 1] or None.
        weights: Float32 tensor of shape [N] or None.
    """

    def __init__(
        self,
        graphs: list,
        fingerprints: torch.Tensor,
        labels: torch.Tensor,
        masks: torch.Tensor | None,
        weights: torch.Tensor | None,
    ) -> None:
        self.graphs = graphs
        self.fingerprints = fingerprints
        self.labels = labels
        self.masks = masks
        self.weights = weights

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.graphs)

    def __getitem__(self, idx: int) -> tuple:
        """Return one sample as (graph, fp, label, mask, weight).

        Args:
            idx: Sample index.

        Returns:
            Tuple of (DGLGraph, fp_tensor, label_tensor, mask_tensor, weight_tensor).
            mask and weight may be None if not provided.
        """
        graph = self.graphs[idx]
        fp = self.fingerprints[idx]
        label = self.labels[idx]
        mask = self.masks[idx] if self.masks is not None else None
        weight = self.weights[idx] if self.weights is not None else None
        return graph, fp, label, mask, weight


def collate_fn(batch: list) -> tuple:
    """Custom collate function for MolecularDataset batches.

    Batches DGL graphs via dgl.batch and stacks all tensors.
    Handles None masks and weights gracefully.

    Args:
        batch: List of (graph, fp, label, mask, weight) tuples.

    Returns:
        Tuple of (batched_graph, fp_batch, label_batch, mask_batch, weight_batch).
        mask_batch and weight_batch may be None.
    """
    graphs, fps, labels, masks, weights = zip(*batch)
    batched_graph = dgl.batch(graphs)
    fp_batch = torch.stack(fps)
    label_batch = torch.stack(labels)
    mask_batch = torch.stack(masks) if masks[0] is not None else None
    weight_batch = torch.stack(weights) if weights[0] is not None else None
    return batched_graph, fp_batch, label_batch, mask_batch, weight_batch


# ---------------------------------------------------------------------------
# Task 9.3: Checkpoint save/load and session management
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer: Any,
    best_val_loss: float,
    best_model_state: dict,
    history: dict,
    epochs_without_improvement: int,
    scaler: StandardScaler,
    config: dict,
) -> None:
    """Save a comprehensive training checkpoint to a .pth file.

    Checkpoint contents match the design spec:
    epoch, model_state_dict (best), optimizer_state_dict, best_val_loss,
    history, epochs_without_improvement, scaler_mean, scaler_scale,
    config, timestamp.

    Args:
        path: File path to save the checkpoint.
        epoch: Current epoch number.
        model: The nn.Module (current state, not necessarily best).
        optimizer: The optimizer instance.
        best_val_loss: Lowest validation loss seen so far.
        best_model_state: State dict of the best model weights.
        history: Dict with keys "train_loss" and "val_loss" (lists of floats).
        epochs_without_improvement: Counter for early stopping.
        scaler: Fitted StandardScaler for label inverse-transform.
        config: Model config dict for this target.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": best_model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "history": history,
        "epochs_without_improvement": epochs_without_improvement,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "config": config,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    torch.save(checkpoint, path)
    logger.debug("Checkpoint saved to %s (epoch %d)", path, epoch)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Any,
    device: str,
) -> dict:
    """Load a checkpoint and restore model + optimizer state.

    If loading fails (corrupt file), logs a warning and returns an empty dict
    so the caller can start training from scratch.

    Args:
        path: Path to the .pth checkpoint file.
        model: The nn.Module to restore weights into.
        optimizer: The optimizer to restore state into.
        device: Device string ("cpu" or "cuda") for map_location.

    Returns:
        Dict with keys: epoch, best_val_loss, epochs_without_improvement,
        history, scaler_mean, scaler_scale, config.
        Empty dict on failure.
    """
    try:
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Checkpoint loaded from %s (epoch %d)", path, checkpoint["epoch"])
        return {
            "epoch": checkpoint["epoch"],
            "best_val_loss": checkpoint["best_val_loss"],
            "epochs_without_improvement": checkpoint["epochs_without_improvement"],
            "history": checkpoint["history"],
            "scaler_mean": checkpoint.get("scaler_mean"),
            "scaler_scale": checkpoint.get("scaler_scale"),
            "config": checkpoint.get("config", {}),
        }
    except Exception as exc:
        logger.warning("Failed to load checkpoint from %s: %s. Starting fresh.", path, exc)
        return {}


def check_existing_session(
    active_dir: str,
    requested_config: TrainingConfig,
) -> str:
    """Check the active/ directory for an existing training session.

    Reads training_config.json from active_dir and compares it to the
    requested config using TrainingConfig.matches().

    Args:
        active_dir: Path to the active training session directory.
        requested_config: The config for the new/resumed run.

    Returns:
        "match"    - existing config matches requested config (safe to resume)
        "mismatch" - existing config differs (user must decide)
        "none"     - no training_config.json found
    """
    config_path = os.path.join(active_dir, "training_config.json")
    if not os.path.exists(config_path):
        return "none"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            existing = TrainingConfig.from_json(f.read())
        if existing.matches(requested_config):
            return "match"
        return "mismatch"
    except Exception as exc:
        logger.warning("Could not read training_config.json: %s", exc)
        return "none"


# ---------------------------------------------------------------------------
# Task 9.4: train_model loop and archive_completed_run
# ---------------------------------------------------------------------------


def _load_fp_csv(path: str) -> torch.Tensor:
    """Load a fingerprint CSV file as a float32 tensor.

    Args:
        path: Path to the CSV file (no header assumed for FP files).

    Returns:
        Float32 tensor of shape [N, cols].
    """
    df = pd.read_csv(path, header=0)
    return torch.tensor(df.values, dtype=torch.float32)


def _build_dataset(
    main_df: pd.DataFrame,
    fp_tensor: torch.Tensor,
    target: str,
    label_scaler: StandardScaler,
    lds_weights: np.ndarray | None,
) -> MolecularDataset:
    """Build a MolecularDataset from a main DataFrame and fingerprint tensor.

    Converts SMILES to DGL graphs, scales labels, and assembles the dataset.

    Args:
        main_df: Main training/validation DataFrame (152 cols).
        fp_tensor: Fingerprint tensor [N, fp_dim].
        target: Property column name.
        label_scaler: Fitted StandardScaler for labels.
        lds_weights: Optional numpy array of LDS weights (train only).

    Returns:
        MolecularDataset instance.
    """
    graphs = []
    valid_indices = []
    for i, smi in enumerate(main_df["smiles"].tolist()):
        g = smiles_to_bigraph(
            smi,
            node_featurizer=_ATOM_FEATURIZER,
            edge_featurizer=_BOND_FEATURIZER,
        )
        if g is not None:
            graphs.append(g)
            valid_indices.append(i)
        else:
            logger.warning("Could not build graph for SMILES at index %d: %s", i, smi)

    if not valid_indices:
        raise ValueError("No valid graphs could be built from the provided SMILES.")

    labels_raw = main_df.iloc[valid_indices][target].values.reshape(-1, 1)
    labels_scaled = label_scaler.transform(labels_raw)
    label_tensor = torch.tensor(labels_scaled, dtype=torch.float32)
    fp_subset = fp_tensor[valid_indices]

    weight_tensor: torch.Tensor | None = None
    if lds_weights is not None:
        weight_tensor = torch.tensor(
            lds_weights[valid_indices], dtype=torch.float32
        ).unsqueeze(1)

    return MolecularDataset(
        graphs=graphs,
        fingerprints=fp_subset,
        labels=label_tensor,
        masks=None,
        weights=weight_tensor,
    )


def train_model(
    target: str,
    data_dir: str,
    active_dir: str,
    config: dict,
    epochs: int,
    patience: int,
    learning_rate: float,
    device: str,
) -> dict:
    """Full training loop for one target property.

    Loads data, builds model, trains with Adam + ReduceLROnPlateau,
    applies LDS weighting, uses early stopping, saves checkpoints,
    and returns final validation metrics.

    Data files expected in data_dir:
      train_{target}.csv, valid_{target}.csv
      train_smiles_{target}.csv, train_sol_{target}.csv
      valid_smiles_{target}.csv, valid_sol_{target}.csv

    Fingerprint assembly (matches prediction_engine.py):
      For GraphFingerprintsModel (abs/em):
        fp = [sol_fp(1024) | smiles_fp(1024) | extra(144)]
      For GraphFingerprintsModelFC (plqy/k):
        fp = [smiles_fp(1024) | extra(144)]  (no separate solvent branch)

    Assumption: extra features = columns 8:152 of main_df (144 cols total:
    8 numeric + 136 scaffold flags). The first 8 of those are MinMaxScaled.

    Args:
        target: Property name (abs, em, plqy, k).
        data_dir: Directory with training CSV files.
        active_dir: Directory for checkpoints and best model.
        config: Model config dict (from MODEL_REGISTRY[target]).
        epochs: Maximum training epochs.
        patience: Early stopping patience.
        learning_rate: Initial Adam learning rate.
        device: "cpu" or "cuda".

    Returns:
        Dict with keys: target, mae, rmse, r2, best_val_loss,
        epochs_trained, history.
    """
    os.makedirs(active_dir, exist_ok=True)
    data_dir_path = Path(data_dir)

    # --- Load CSVs ---
    train_df = pd.read_csv(data_dir_path / f"train_{target}.csv")
    valid_df = pd.read_csv(data_dir_path / f"valid_{target}.csv")
    logger.info("Loaded %d train / %d valid rows for %s", len(train_df), len(valid_df), target)

    # --- Fit scalers ---
    label_scaler = StandardScaler()
    label_scaler.fit(train_df[[target]].values)

    num_scaler = MinMaxScaler()
    num_scaler.fit(train_df.iloc[:, 8:16].values)

    # --- Build fingerprint tensors ---
    train_sol = _load_fp_csv(str(data_dir_path / f"train_sol_{target}.csv"))
    valid_sol = _load_fp_csv(str(data_dir_path / f"valid_sol_{target}.csv"))
    train_smi = _load_fp_csv(str(data_dir_path / f"train_smiles_{target}.csv"))
    valid_smi = _load_fp_csv(str(data_dir_path / f"valid_smiles_{target}.csv"))

    # Extra features: cols 8:152 (8 numeric + 136 scaffold flags)
    def _build_extra(df: pd.DataFrame) -> torch.Tensor:
        num_raw = df.iloc[:, 8:16].values
        rest = df.iloc[:, 16:152].values
        num_scaled = num_scaler.transform(num_raw)
        extra = np.concatenate([num_scaled, rest], axis=1)
        return torch.tensor(extra, dtype=torch.float32)

    train_extra = _build_extra(train_df)
    valid_extra = _build_extra(valid_df)

    model_class = config.get("model_class", "GraphFingerprintsModel")
    if model_class == "GraphFingerprintsModel":
        # sol(1024) + smiles(1024) + extra(144) = 2192
        train_fp = torch.cat([train_sol, train_smi, train_extra], dim=1)
        valid_fp = torch.cat([valid_sol, valid_smi, valid_extra], dim=1)
        solvent_dim = train_sol.shape[1]
        smiles_extra_dim = train_smi.shape[1] + train_extra.shape[1]
        fp_size = train_fp.shape[1]
    else:
        # plqy/k: smiles(1024) + extra(144) = 1168 (no separate solvent)
        train_fp = torch.cat([train_smi, train_extra], dim=1)
        valid_fp = torch.cat([valid_smi, valid_extra], dim=1)
        solvent_dim = MORGAN_NBITS
        smiles_extra_dim = train_smi.shape[1] + train_extra.shape[1]
        fp_size = train_fp.shape[1]

    # --- LDS weights ---
    alpha = config.get("alpha", 0.0)
    lds_weights = compute_lds_weights(train_df[target].values, alpha=alpha)

    # --- Build datasets and loaders ---
    batch_size = config.get("batch_size", 32)
    train_ds = _build_dataset(train_df, train_fp, target, label_scaler, lds_weights)
    valid_ds = _build_dataset(valid_df, valid_fp, target, label_scaler, None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # --- Build model ---
    node_feat_size = _ATOM_FEATURIZER.feat_size("hv")
    edge_feat_size = _BOND_FEATURIZER.feat_size("he")
    model = build_model(
        target=target,
        node_feat_size=node_feat_size,
        edge_feat_size=edge_feat_size,
        solvent_dim=solvent_dim,
        smiles_extra_dim=smiles_extra_dim,
        fp_size=fp_size,
        graph_feat_size=GRAPH_FEAT_SIZE,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=10, min_lr=1e-6
    )
    criterion = nn.MSELoss(reduction="none")

    # --- Resume from checkpoint if available ---
    checkpoint_path = os.path.join(active_dir, f"checkpoint_{target}.pth")
    best_val_loss = float("inf")
    best_model_state: dict = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history: dict = {"train_loss": [], "val_loss": []}
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        state = load_checkpoint(checkpoint_path, model, optimizer, device)
        if state:
            start_epoch = state["epoch"]
            best_val_loss = state["best_val_loss"]
            epochs_without_improvement = state["epochs_without_improvement"]
            history = state["history"]
            best_model_state = copy.deepcopy(model.state_dict())
            logger.info("Resuming %s from epoch %d", target, start_epoch)

    # --- Training loop ---
    use_lds = alpha > 0
    final_epoch = start_epoch

    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_batches = 0

        for graphs, fps, labels, masks, weights in train_loader:
            graphs = graphs.to(device)
            fps = fps.to(device)
            labels = labels.to(device)
            node_feats = graphs.ndata["hv"]
            edge_feats = graphs.edata.get("he", None)

            optimizer.zero_grad()
            preds = model(graphs, node_feats, edge_feats, fps)
            loss_per_sample = criterion(preds, labels)

            if use_lds and weights is not None:
                loss = (loss_per_sample * weights.to(device)).mean()
            else:
                loss = loss_per_sample.mean()

            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()
            train_batches += 1

        train_loss = train_loss_total / max(train_batches, 1)

        # --- Validation ---
        model.eval()
        val_loss_total = 0.0
        val_batches = 0
        all_preds: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        with torch.no_grad():
            for graphs, fps, labels, masks, _ in valid_loader:
                graphs = graphs.to(device)
                fps = fps.to(device)
                labels = labels.to(device)
                node_feats = graphs.ndata["hv"]
                edge_feats = graphs.edata.get("he", None)
                preds = model(graphs, node_feats, edge_feats, fps)
                val_loss_total += criterion(preds, labels).mean().item()
                val_batches += 1
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        val_loss = val_loss_total / max(val_batches, 1)
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        final_epoch = epoch

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            # Save best model
            torch.save(best_model_state, os.path.join(active_dir, f"Model_{target}.pth"))
        else:
            epochs_without_improvement += 1

        logger.info(
            "Epoch %d/%d | train=%.4f val=%.4f | no_improve=%d",
            epoch, epochs, train_loss, val_loss, epochs_without_improvement,
        )

        # Save checkpoint after every epoch
        save_checkpoint(
            path=checkpoint_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            best_val_loss=best_val_loss,
            best_model_state=best_model_state,
            history=history,
            epochs_without_improvement=epochs_without_improvement,
            scaler=label_scaler,
            config=config,
        )

        if epochs_without_improvement >= patience:
            logger.info("Early stopping at epoch %d for %s", epoch, target)
            break

    # --- Final metrics on validation set (original scale) ---
    model.load_state_dict(best_model_state)
    model.eval()
    all_preds_final: list[np.ndarray] = []
    all_labels_final: list[np.ndarray] = []

    with torch.no_grad():
        for graphs, fps, labels, masks, _ in valid_loader:
            graphs = graphs.to(device)
            fps = fps.to(device)
            node_feats = graphs.ndata["hv"]
            edge_feats = graphs.edata.get("he", None)
            preds = model(graphs, node_feats, edge_feats, fps)
            all_preds_final.append(preds.cpu().numpy())
            all_labels_final.append(labels.cpu().numpy())

    preds_scaled = np.vstack(all_preds_final)
    labels_scaled = np.vstack(all_labels_final)
    preds_orig = label_scaler.inverse_transform(preds_scaled)
    labels_orig = label_scaler.inverse_transform(labels_scaled)

    mae = float(mean_absolute_error(labels_orig, preds_orig))
    rmse = float(np.sqrt(mean_squared_error(labels_orig, preds_orig)))
    r2 = float(r2_score(labels_orig, preds_orig))

    logger.info("Final metrics for %s: MAE=%.4f RMSE=%.4f R2=%.4f", target, mae, rmse, r2)

    return {
        "target": target,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "best_val_loss": best_val_loss,
        "epochs_trained": final_epoch,
        "history": history,
    }


def archive_completed_run(
    active_dir: str,
    completed_dir: str,
    targets: list[str],
) -> str:
    """Move final models to a timestamped subfolder in completed_dir.

    Steps:
    1. Create YYYY-MM-DD_HH-MM-SS subfolder in completed_dir.
    2. Copy Model_{target}.pth files from active_dir to the new subfolder.
    3. Copy training_config.json to the new subfolder.
    4. Remove checkpoint_{target}.pth files from active_dir.

    Args:
        active_dir: Path to the active training session directory.
        completed_dir: Path to the completed runs directory.
        targets: List of property names that were trained.

    Returns:
        Path to the newly created archive subfolder.
    """
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    archive_path = os.path.join(completed_dir, timestamp)
    os.makedirs(archive_path, exist_ok=True)

    # Copy best model files
    for target in targets:
        src = os.path.join(active_dir, f"Model_{target}.pth")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(archive_path, f"Model_{target}.pth"))
            logger.info("Archived Model_%s.pth to %s", target, archive_path)
        else:
            logger.warning("Model_%s.pth not found in %s, skipping.", target, active_dir)

    # Copy training config
    config_src = os.path.join(active_dir, "training_config.json")
    if os.path.exists(config_src):
        shutil.copy2(config_src, os.path.join(archive_path, "training_config.json"))

    # Clean up checkpoint files from active_dir
    for target in targets:
        ckpt = os.path.join(active_dir, f"checkpoint_{target}.pth")
        if os.path.exists(ckpt):
            os.remove(ckpt)
            logger.debug("Removed checkpoint %s", ckpt)

    logger.info("Run archived to %s", archive_path)
    return archive_path
