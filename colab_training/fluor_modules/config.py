"""Configuration constants and model registry for fluorescent molecule property prediction.

This is a pure-data module -- no heavy imports (no torch, dgl, rdkit).
All model hyperparameters, path definitions, and shared constants live here.

Reference: design.md section "config.py"
"""

# Per-property model architecture hyperparameters.
# Keys: num_layers, num_timesteps, dropout, alpha (LDS), model_class name.
MODEL_REGISTRY: dict[str, dict] = {
    "abs":  {"num_layers": 2, "num_timesteps": 2, "dropout": 0.3, "alpha": 0.1, "model_class": "GraphFingerprintsModel"},
    "em":   {"num_layers": 3, "num_timesteps": 1, "dropout": 0.3, "alpha": 0.0, "model_class": "GraphFingerprintsModel"},
    "plqy": {"num_layers": 2, "num_timesteps": 3, "dropout": 0.4, "alpha": 0.2, "model_class": "GraphFingerprintsModelFC"},
    "k":    {"num_layers": 3, "num_timesteps": 1, "dropout": 0.3, "alpha": 0.6, "model_class": "GraphFingerprintsModelFC"},
}

# The four predicted fluorescent properties.
PROPERTIES: list[str] = ["abs", "em", "plqy", "k"]

# Model and feature dimensions.
GRAPH_FEAT_SIZE: int = 256
BATCH_SIZE: int = 32
LEARNING_RATE: float = 5e-4
MORGAN_RADIUS: int = 2
MORGAN_NBITS: int = 1024
NUM_SCAFFOLD_FLAGS: int = 136
FEATURE_VECTOR_DIM: int = 2192  # 1024 + 1024 + 8 + 136
NUMERIC_FEATURE_COUNT: int = 8

# Google Drive paths (used in Colab runtime).
DRIVE_ROOT: str = "/content/drive/MyDrive/fluor-tools"
DATASETS_DIR: str = f"{DRIVE_ROOT}/datasets"
ACTIVE_RUN_DIR: str = f"{DRIVE_ROOT}/training-runs/active"
COMPLETED_RUNS_DIR: str = f"{DRIVE_ROOT}/training-runs/completed"

# Every MODEL_REGISTRY entry must contain these keys.
REQUIRED_MODEL_REGISTRY_KEYS: list[str] = [
    "num_layers", "num_timesteps", "dropout", "alpha", "model_class"
]
