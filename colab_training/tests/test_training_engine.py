"""Tests for training_engine.py.

Covers:
- Property 16: LDS uniform weights at alpha zero
- Property 17: TrainingConfig serialization round trip
- Property 19: Checkpoint save/load round trip
- Unit tests: LDS, TrainingConfig, checkpoint, session management, archive
"""

import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from hypothesis import given, settings
from hypothesis.strategies import (
    composite,
    dictionaries,
    floats,
    integers,
    lists,
    sampled_from,
    text,
)

from fluor_modules.training_engine import (
    TrainingConfig,
    archive_completed_run,
    check_existing_session,
    collate_fn,
    compute_lds_weights,
    load_checkpoint,
    save_checkpoint,
)
from fluor_modules.config import PROPERTIES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides: Any) -> TrainingConfig:
    """Build a minimal TrainingConfig for testing."""
    defaults = dict(
        targets=["abs", "em"],
        epochs=10,
        patience=5,
        learning_rate=5e-4,
        batch_size=32,
        model_configs={"abs": {"num_layers": 2}, "em": {"num_layers": 3}},
        data_source="/tmp/data",
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=10,
        lr_scheduler_min=1e-6,
        run_id=str(uuid.uuid4()),
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def _make_tiny_model() -> nn.Module:
    """Return a tiny linear model for checkpoint tests (no DGL dependency)."""
    return nn.Linear(4, 1)


def _make_optimizer(model: nn.Module) -> optim.Optimizer:
    """Return an Adam optimizer for the given model."""
    return optim.Adam(model.parameters(), lr=1e-3)


def _make_scaler(values: list[float] | None = None) -> Any:
    """Return a fitted StandardScaler."""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data = np.array(values or [1.0, 2.0, 3.0, 4.0, 5.0]).reshape(-1, 1)
    scaler.fit(data)
    return scaler


# ---------------------------------------------------------------------------
# Property 16: LDS uniform weights at alpha zero
# Validates: Requirements 12.7
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    labels=lists(
        floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=50,
    )
)
def test_lds_uniform_weights_at_alpha_zero(labels: list[float]) -> None:
    """For any numpy array of float labels, compute_lds_weights with alpha=0
    returns an array of all 1.0 values with the same length as the input.

    **Validates: Requirements 12.7**
    """
    arr = np.array(labels, dtype=np.float32)
    weights = compute_lds_weights(arr, alpha=0)
    assert len(weights) == len(arr), "Weight array length must match input length"
    np.testing.assert_array_equal(
        weights,
        np.ones(len(arr), dtype=np.float32),
        err_msg="alpha=0 must return all 1.0 weights",
    )


# ---------------------------------------------------------------------------
# Property 17: TrainingConfig serialization round trip
# Validates: Requirements 12.8
# ---------------------------------------------------------------------------


@composite
def training_configs(draw: Any) -> TrainingConfig:
    """Hypothesis strategy to generate random TrainingConfig instances."""
    all_props = ["abs", "em", "plqy", "k"]
    n = draw(integers(min_value=1, max_value=4))
    targets = draw(
        lists(sampled_from(all_props), min_size=n, max_size=n, unique=True)
    )
    model_configs = {t: {"num_layers": draw(integers(1, 4))} for t in targets}
    return TrainingConfig(
        targets=targets,
        epochs=draw(integers(min_value=1, max_value=500)),
        patience=draw(integers(min_value=1, max_value=100)),
        learning_rate=draw(floats(min_value=1e-6, max_value=1e-1, allow_nan=False)),
        batch_size=draw(integers(min_value=1, max_value=128)),
        model_configs=model_configs,
        data_source=draw(text(min_size=1, max_size=50, alphabet="abcdefghijklmnopqrstuvwxyz/_-")),
        lr_scheduler_factor=draw(floats(min_value=0.1, max_value=0.9, allow_nan=False)),
        lr_scheduler_patience=draw(integers(min_value=1, max_value=50)),
        lr_scheduler_min=draw(floats(min_value=1e-8, max_value=1e-4, allow_nan=False)),
        run_id=draw(text(min_size=1, max_size=36, alphabet="abcdef0123456789-")),
    )


@settings(max_examples=100)
@given(config=training_configs())
def test_training_config_serialization_round_trip(config: TrainingConfig) -> None:
    """For any valid TrainingConfig, to_json() then from_json() produces a
    config that matches() the original.

    **Validates: Requirements 12.8**
    """
    json_str = config.to_json()
    restored = TrainingConfig.from_json(json_str)
    assert config.matches(restored), (
        f"Round-trip failed.\nOriginal: {config}\nRestored: {restored}"
    )


# ---------------------------------------------------------------------------
# Property 19: Checkpoint save/load round trip
# Validates: Requirements 7.6
# ---------------------------------------------------------------------------


@composite
def training_states(draw: Any) -> dict:
    """Hypothesis strategy to generate random training state dicts."""
    n_epochs = draw(integers(min_value=1, max_value=20))
    train_losses = draw(
        lists(floats(min_value=0.0, max_value=10.0, allow_nan=False), min_size=n_epochs, max_size=n_epochs)
    )
    val_losses = draw(
        lists(floats(min_value=0.0, max_value=10.0, allow_nan=False), min_size=n_epochs, max_size=n_epochs)
    )
    return {
        "epoch": n_epochs,
        "best_val_loss": draw(floats(min_value=0.0, max_value=10.0, allow_nan=False)),
        "epochs_without_improvement": draw(integers(min_value=0, max_value=20)),
        "history": {"train_loss": train_losses, "val_loss": val_losses},
    }


@settings(max_examples=50, deadline=None)
@given(state=training_states())
def test_checkpoint_save_load_round_trip(state: dict) -> None:
    """For any training state, saving a checkpoint and loading it back recovers
    epoch, best_val_loss, epochs_without_improvement, and history lists.

    **Validates: Requirements 7.6**
    """
    import tempfile
    model = _make_tiny_model()
    optimizer = _make_optimizer(model)
    scaler = _make_scaler()
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_checkpoint.pth")

        save_checkpoint(
            path=ckpt_path,
            epoch=state["epoch"],
            model=model,
            optimizer=optimizer,
            best_val_loss=state["best_val_loss"],
            best_model_state=dict(model.state_dict()),
            history=state["history"],
            epochs_without_improvement=state["epochs_without_improvement"],
            scaler=scaler,
            config={"num_layers": 2},
        )

        # Load into a fresh model/optimizer
        model2 = _make_tiny_model()
        optimizer2 = _make_optimizer(model2)
        loaded = load_checkpoint(ckpt_path, model2, optimizer2, device="cpu")

        assert loaded["epoch"] == state["epoch"], "epoch mismatch"
        assert abs(loaded["best_val_loss"] - state["best_val_loss"]) < 1e-6, "best_val_loss mismatch"
        assert loaded["epochs_without_improvement"] == state["epochs_without_improvement"]
        assert loaded["history"]["train_loss"] == state["history"]["train_loss"]
        assert loaded["history"]["val_loss"] == state["history"]["val_loss"]


# ===========================================================================
# Unit tests (Task 9.8)
# ===========================================================================


class TestLDSWeights:
    """Unit tests for compute_lds_weights."""

    def test_alpha_zero_returns_ones(self) -> None:
        """alpha=0 returns all 1.0 weights."""
        labels = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = compute_lds_weights(labels, alpha=0)
        np.testing.assert_array_equal(weights, np.ones(5, dtype=np.float32))

    def test_alpha_positive_returns_positive_values(self) -> None:
        """alpha>0 returns positive weights."""
        labels = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        weights = compute_lds_weights(labels, alpha=0.5)
        assert np.all(weights > 0), "All weights must be positive"

    def test_alpha_positive_mean_is_one(self) -> None:
        """alpha>0 weights are normalized to mean=1."""
        labels = np.linspace(0, 10, 50)
        weights = compute_lds_weights(labels, alpha=0.3)
        assert abs(np.mean(weights) - 1.0) < 1e-5, "Mean weight should be ~1.0"

    def test_output_length_matches_input(self) -> None:
        """Output length equals input length for both alpha=0 and alpha>0."""
        for alpha in [0.0, 0.5]:
            labels = np.array([1.0, 2.0, 3.0])
            weights = compute_lds_weights(labels, alpha=alpha)
            assert len(weights) == 3

    def test_empty_array_returns_empty(self) -> None:
        """Empty input returns empty output."""
        weights = compute_lds_weights(np.array([]), alpha=0)
        assert len(weights) == 0


class TestTrainingConfigJSON:
    """Unit tests for TrainingConfig JSON serialization."""

    def test_round_trip_with_known_config(self) -> None:
        """to_json() then from_json() recovers the original config."""
        cfg = _make_config()
        restored = TrainingConfig.from_json(cfg.to_json())
        assert cfg.matches(restored)

    def test_matches_returns_false_for_different_config(self) -> None:
        """matches() returns False when configs differ."""
        cfg1 = _make_config(epochs=10)
        cfg2 = _make_config(epochs=20)
        assert not cfg1.matches(cfg2)

    def test_matches_returns_true_for_identical(self) -> None:
        """matches() returns True for identical configs."""
        cfg = _make_config()
        assert cfg.matches(cfg)

    def test_json_contains_all_fields(self) -> None:
        """JSON output contains all required fields."""
        cfg = _make_config()
        import json
        data = json.loads(cfg.to_json())
        required = [
            "targets", "epochs", "patience", "learning_rate", "batch_size",
            "model_configs", "data_source", "lr_scheduler_factor",
            "lr_scheduler_patience", "lr_scheduler_min", "run_id",
        ]
        for field in required:
            assert field in data, f"Missing field: {field}"


class TestCheckpointSaveLoad:
    """Unit tests for save_checkpoint and load_checkpoint."""

    def test_save_then_load_recovers_epoch(self, tmp_path: Path) -> None:
        """Saved epoch is recovered after load."""
        model = _make_tiny_model()
        optimizer = _make_optimizer(model)
        scaler = _make_scaler()
        path = str(tmp_path / "ckpt.pth")

        save_checkpoint(
            path=path, epoch=42, model=model, optimizer=optimizer,
            best_val_loss=0.123, best_model_state=dict(model.state_dict()),
            history={"train_loss": [0.5], "val_loss": [0.4]},
            epochs_without_improvement=3, scaler=scaler, config={"num_layers": 2},
        )

        model2 = _make_tiny_model()
        optimizer2 = _make_optimizer(model2)
        loaded = load_checkpoint(path, model2, optimizer2, device="cpu")

        assert loaded["epoch"] == 42
        assert abs(loaded["best_val_loss"] - 0.123) < 1e-6
        assert loaded["epochs_without_improvement"] == 3
        assert loaded["history"]["train_loss"] == [0.5]
        assert loaded["history"]["val_loss"] == [0.4]

    def test_load_corrupt_checkpoint_returns_empty_dict(self, tmp_path: Path) -> None:
        """Loading a corrupt checkpoint returns empty dict without raising."""
        bad_path = str(tmp_path / "bad.pth")
        with open(bad_path, "w") as f:
            f.write("not a valid checkpoint")

        model = _make_tiny_model()
        optimizer = _make_optimizer(model)
        result = load_checkpoint(bad_path, model, optimizer, device="cpu")
        assert result == {}

    def test_load_nonexistent_checkpoint_returns_empty_dict(self, tmp_path: Path) -> None:
        """Loading a non-existent checkpoint returns empty dict."""
        model = _make_tiny_model()
        optimizer = _make_optimizer(model)
        result = load_checkpoint(str(tmp_path / "missing.pth"), model, optimizer, "cpu")
        assert result == {}

    def test_scaler_state_is_saved(self, tmp_path: Path) -> None:
        """Checkpoint includes scaler_mean and scaler_scale."""
        model = _make_tiny_model()
        optimizer = _make_optimizer(model)
        scaler = _make_scaler([10.0, 20.0, 30.0])
        path = str(tmp_path / "ckpt.pth")

        save_checkpoint(
            path=path, epoch=1, model=model, optimizer=optimizer,
            best_val_loss=0.5, best_model_state=dict(model.state_dict()),
            history={"train_loss": [], "val_loss": []},
            epochs_without_improvement=0, scaler=scaler, config={},
        )

        model2 = _make_tiny_model()
        optimizer2 = _make_optimizer(model2)
        loaded = load_checkpoint(path, model2, optimizer2, "cpu")
        assert loaded["scaler_mean"] is not None
        assert loaded["scaler_scale"] is not None


class TestCheckExistingSession:
    """Unit tests for check_existing_session."""

    def test_returns_none_for_empty_dir(self, tmp_path: Path) -> None:
        """Returns 'none' when no training_config.json exists."""
        result = check_existing_session(str(tmp_path), _make_config())
        assert result == "none"

    def test_returns_match_for_matching_config(self, tmp_path: Path) -> None:
        """Returns 'match' when existing config matches requested config."""
        cfg = _make_config()
        config_path = tmp_path / "training_config.json"
        config_path.write_text(cfg.to_json(), encoding="utf-8")

        result = check_existing_session(str(tmp_path), cfg)
        assert result == "match"

    def test_returns_mismatch_for_different_config(self, tmp_path: Path) -> None:
        """Returns 'mismatch' when existing config differs from requested."""
        cfg1 = _make_config(epochs=10)
        cfg2 = _make_config(epochs=99)
        config_path = tmp_path / "training_config.json"
        config_path.write_text(cfg1.to_json(), encoding="utf-8")

        result = check_existing_session(str(tmp_path), cfg2)
        assert result == "mismatch"

    def test_returns_none_for_corrupt_config(self, tmp_path: Path) -> None:
        """Returns 'none' when training_config.json is corrupt."""
        config_path = tmp_path / "training_config.json"
        config_path.write_text("not valid json", encoding="utf-8")

        result = check_existing_session(str(tmp_path), _make_config())
        assert result == "none"


class TestArchiveCompletedRun:
    """Unit tests for archive_completed_run."""

    def test_creates_timestamped_subfolder(self, tmp_path: Path) -> None:
        """archive_completed_run creates a subfolder in completed_dir."""
        active_dir = tmp_path / "active"
        completed_dir = tmp_path / "completed"
        active_dir.mkdir()
        completed_dir.mkdir()

        # Create dummy model files
        targets = ["abs", "em"]
        for t in targets:
            (active_dir / f"Model_{t}.pth").write_bytes(b"dummy")

        archive_path = archive_completed_run(
            str(active_dir), str(completed_dir), targets
        )

        assert os.path.isdir(archive_path), "Archive path must be a directory"
        assert Path(archive_path).parent == completed_dir

    def test_copies_model_files(self, tmp_path: Path) -> None:
        """Model files are copied to the archive subfolder."""
        active_dir = tmp_path / "active"
        completed_dir = tmp_path / "completed"
        active_dir.mkdir()
        completed_dir.mkdir()

        targets = ["abs"]
        (active_dir / "Model_abs.pth").write_bytes(b"model_data")

        archive_path = archive_completed_run(str(active_dir), str(completed_dir), targets)

        assert (Path(archive_path) / "Model_abs.pth").exists()

    def test_copies_training_config(self, tmp_path: Path) -> None:
        """training_config.json is copied to the archive subfolder."""
        active_dir = tmp_path / "active"
        completed_dir = tmp_path / "completed"
        active_dir.mkdir()
        completed_dir.mkdir()

        cfg = _make_config()
        (active_dir / "training_config.json").write_text(cfg.to_json(), encoding="utf-8")
        (active_dir / "Model_abs.pth").write_bytes(b"dummy")

        archive_path = archive_completed_run(str(active_dir), str(completed_dir), ["abs"])

        assert (Path(archive_path) / "training_config.json").exists()

    def test_removes_checkpoint_files(self, tmp_path: Path) -> None:
        """Checkpoint files are removed from active_dir after archiving."""
        active_dir = tmp_path / "active"
        completed_dir = tmp_path / "completed"
        active_dir.mkdir()
        completed_dir.mkdir()

        targets = ["abs"]
        (active_dir / "Model_abs.pth").write_bytes(b"model")
        ckpt = active_dir / "checkpoint_abs.pth"
        ckpt.write_bytes(b"checkpoint")

        archive_completed_run(str(active_dir), str(completed_dir), targets)

        assert not ckpt.exists(), "Checkpoint file should be removed after archiving"

    def test_archive_path_has_timestamp_format(self, tmp_path: Path) -> None:
        """Archive subfolder name matches YYYY-MM-DD_HH-MM-SS format."""
        import re
        active_dir = tmp_path / "active"
        completed_dir = tmp_path / "completed"
        active_dir.mkdir()
        completed_dir.mkdir()

        archive_path = archive_completed_run(str(active_dir), str(completed_dir), [])
        folder_name = Path(archive_path).name
        pattern = r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$"
        assert re.match(pattern, folder_name), (
            f"Archive folder name '{folder_name}' does not match timestamp format"
        )
