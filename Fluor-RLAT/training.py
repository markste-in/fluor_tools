#!/usr/bin/env python3
"""
Fluor-RLAT Training Script
===========================
Train property prediction models for fluorescent dyes from scratch.

This script trains 4 separate single-task models:
- Model_abs.pth: Absorption wavelength (nm)
- Model_em.pth: Emission wavelength (nm)  
- Model_plqy.pth: Photoluminescence quantum yield (0-1)
- Model_k.pth: Log10 molar absorptivity

Usage:
    python training.py --target abs          # Train absorption model only
    python training.py --target all          # Train all 4 models (default)
    python training.py --target abs --epochs 100 --patience 30

Default Parameters:
    --target all         Train all 4 models
    --epochs 200         Maximum training epochs
    --patience 20        Early stopping patience
    --data-dir ./data    Training data directory
    --output-dir .       Save models to current directory
    --device auto        CUDA if available, else CPU (DGL doesn't support MPS)

Requirements:
    conda activate fluor
    # Requires: torch, dgl, dgllife, pandas, numpy, scikit-learn

Model Architecture Details (reverse-engineered from pretrained .pth files):
============================================================================

Two architecture variants exist:

1. AttentionCNN Architecture (abs, em models):
   - Graph: AttentiveFPGNN → AttentiveFPReadout → 256-dim
   - Solvent: FC(1024→256→256) → 256-dim
   - Molecule+Extra: FingerprintAttentionCNN(2192→512) → 512-dim
   - Fusion: concat → 1024-dim → predict(128→1)
   - Params: ~2.3M

2. SimpleFC Architecture (plqy, k models):
   - Graph: AttentiveFPGNN → AttentiveFPReadout → 256-dim
   - All FP: FC(2192→256→256) → 256-dim
   - Fusion: concat → 512-dim → predict(128→1)
   - Params: ~2.5-3.0M

Per-model hyperparameters:
   abs:  num_layers=2, num_timesteps=2, dropout=0.3, LDS α=0.1
   em:   num_layers=2, num_timesteps=1, dropout=0.3, LDS α=0.0
   plqy: num_layers=2, num_timesteps=3, dropout=0.4, LDS α=0.2
   k:    num_layers=3, num_timesteps=1, dropout=0.3, LDS α=0.6

Data Requirements:
==================
Training data should be in ./data/ directory with files:
- train_{target}.csv, valid_{target}.csv, test_{target}.csv
- train_sol_{target}.csv, valid_sol_{target}.csv (solvent Morgan FP, 1024-bit)
- train_smiles_{target}.csv, valid_smiles_{target}.csv (molecule Morgan FP, 2048-bit)

Each train_{target}.csv should have columns:
- smiles: SMILES string
- {target}: target value (abs/em/plqy/k)
- Columns 8-15: numeric descriptors (MW, LogP, TPSA, etc.)
- Columns 16-151: scaffold binary flags (fragment_1..fragment_136)

Expected Performance (validation set):
======================================
After ~50-100 epochs with early stopping:
- abs:  MAE ~15-20 nm, R² ~0.90-0.95
- em:   MAE ~20-25 nm, R² ~0.88-0.93
- plqy: MAE ~0.10-0.15, R² ~0.70-0.80
- k:    MAE ~0.15-0.20, R² ~0.75-0.85

Deployment to Web UI:
=====================
After training, copy the model files to use them in the web interface:

1. For standalone Fluor-RLAT predictions:
   Models are saved in the current directory (or --output-dir) by default.
   No copying needed if running from Fluor-RLAT/ folder.

2. For the web UI (Flask app at NIRFluor-opt and web/):
   Copy all 4 model files to the predict/ subfolder:
   
   cp Model_abs.pth  "../NIRFluor-opt and web/predict/"
   cp Model_em.pth   "../NIRFluor-opt and web/predict/"
   cp Model_plqy.pth "../NIRFluor-opt and web/predict/"
   cp Model_k.pth    "../NIRFluor-opt and web/predict/"
   
   Or use the deploy command:
   cp Model_*.pth "../NIRFluor-opt and web/predict/"

3. Verify deployment:
   ls -la "../NIRFluor-opt and web/predict/"*.pth
   
   Then restart the Flask app to load the new models.

Author: Fluor-tools project
Date: February 2026
"""

import argparse
import copy
import os
import random
import sys
import time
from pathlib import Path

from tqdm import tqdm

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from dgllife.data import MoleculeCSVDataset
from dgllife.model.gnn import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout
from dgllife.utils import (
    AttentiveFPAtomFeaturizer,
    AttentiveFPBondFeaturizer,
    smiles_to_bigraph,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset

# ============================================================================
# Configuration for each target property
# ============================================================================
# These hyperparameters were determined by analyzing the pretrained .pth files:
# - num_layers: counted from gnn.gnn_layers.{N} keys in state_dict
# - num_timesteps: counted from readout.readouts.{N} keys in state_dict
# - model_type: determined by presence of fp_extractor vs fp_fc keys
# - solvent_dim/fp_size: extracted from weight tensor shapes

MODEL_CONFIGS = {
    'abs': {
        'num_layers': 2,        # GNN layers (from gnn.gnn_layers count)
        'num_timesteps': 2,     # Readout timesteps (from readout.readouts count)
        'dropout': 0.3,
        'alpha': 0.1,           # LDS weight (label distribution smoothing)
        'model_type': 'attention_cnn',  # Uses FingerprintAttentionCNN
        'solvent_dim': 1024,    # From solvent_extractor.0.weight shape [256, 1024]
    },
    'em': {
        'num_layers': 2,        # GNN layers
        'num_timesteps': 1,     # From readout.readouts count (only readouts.0)
        'dropout': 0.3,
        'alpha': 0.0,           # No LDS for emission
        'model_type': 'attention_cnn',
        'solvent_dim': 1024,
    },
    'plqy': {
        'num_layers': 2,        # From gnn.gnn_layers count
        'num_timesteps': 3,     # From readout.readouts count (0, 1, 2)
        'dropout': 0.4,
        'alpha': 0.2,
        'model_type': 'simple_fc',  # Uses fp_fc (simpler architecture)
        'fp_size': 2192,        # From fp_fc.0.weight shape [256, 2192]
    },
    'k': {
        'num_layers': 3,        # From gnn.gnn_layers count (0, 1)
        'num_timesteps': 1,     # From readout.readouts count (only readouts.0)
        'dropout': 0.3,
        'alpha': 0.6,           # Higher LDS weight for molar absorptivity
        'model_type': 'simple_fc',
        'fp_size': 2192,
    },
}

# Fixed hyperparameters (same across all models)
GRAPH_FEAT_SIZE = 256
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
SEED = 42


# ============================================================================
# Model Architectures
# ============================================================================

class FingerprintAttentionCNN(nn.Module):
    """CNN with attention mechanism for fingerprint feature extraction.
    
    Used by abs and em models. Applies 1D convolution with learned attention
    weights to extract informative features from concatenated fingerprints.
    
    Architecture:
        Input [B, D] → unsqueeze → [B, 1, D]
        → conv_feat [B, C, D] (feature extraction)
        → conv_attn [B, C, D] (attention weights)
        → softmax attention → weighted sum [B, C]
        → concat with max pool [B, C] → [B, 2C]
    
    Input: [B, D] where D = smiles_fp (2048) + extra_features (144) = 2192
    Output: [B, 2*conv_channels] = [B, 512] for default settings
    
    This architecture was reverse-engineered from Model_abs.pth which contains:
        fp_extractor.conv_feat.weight: [256, 1, 3]
        fp_extractor.conv_attn.weight: [256, 1, 3]
    """
    def __init__(self, input_dim, conv_channels=256):
        super(FingerprintAttentionCNN, self).__init__()
        self.conv_feat = nn.Conv1d(1, conv_channels, kernel_size=3, padding=1)
        self.conv_attn = nn.Conv1d(1, conv_channels, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=-1)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, D]
        feat_map = self.conv_feat(x)         # [B, C, D]
        attn_map = self.conv_attn(x)         # [B, C, D]
        attn_weights = self.softmax(attn_map)
        attn_out = torch.sum(feat_map * attn_weights, dim=-1)  # [B, C]
        pooled = self.pool(feat_map).squeeze(-1)               # [B, C]
        return torch.cat([attn_out, pooled], dim=1)            # [B, 2C]


class GraphFingerprintsModelAttentionCNN(nn.Module):
    """Model architecture for abs and em prediction.
    
    Combines:
    - AttentiveFP GNN for molecular graph encoding
    - FingerprintAttentionCNN for molecule fingerprint + descriptors
    - FC network for solvent fingerprint
    
    This architecture was reverse-engineered from Model_abs.pth state_dict:
        gnn.init_context.project_node.0.weight: [256, 39]  → node_feat_size=39
        gnn.init_context.project_edge1.0.weight: [256, 49] → edge_feat_size=10 (49=39+10)
        gnn.gnn_layers.0.* exists, gnn.gnn_layers.1.* doesn't → num_layers=2
        readout.readouts.0.*, readout.readouts.1.* → num_timesteps=2
        solvent_extractor.0.weight: [256, 1024] → solvent_dim=1024
        fp_extractor.conv_feat.weight: [256, 1, 3] → uses AttentionCNN
        predict.1.weight: [128, 1024] → fusion_dim=1024 (256+256+512)
    
    Total input to predict layer: 256 (graph) + 256 (solvent) + 512 (fp) = 1024
    """
    def __init__(self, node_feat_size, edge_feat_size,
                 solvent_dim, smiles_extra_dim,
                 graph_feat_size=256, num_layers=2, num_timesteps=2,
                 n_tasks=1, dropout=0.3):
        super(GraphFingerprintsModelAttentionCNN, self).__init__()

        # Graph neural network for dye molecule
        self.gnn = AttentiveFPGNN(
            node_feat_size=node_feat_size,
            edge_feat_size=edge_feat_size,
            num_layers=num_layers,
            graph_feat_size=graph_feat_size,
            dropout=dropout
        )
        self.readout = AttentiveFPReadout(
            feat_size=graph_feat_size,
            num_timesteps=num_timesteps,
            dropout=dropout
        )

        # Fingerprint extractor with attention CNN
        self.fp_extractor = FingerprintAttentionCNN(
            smiles_extra_dim, 
            conv_channels=graph_feat_size
        )

        # Solvent fingerprint extractor (simple FC)
        self.solvent_extractor = nn.Sequential(
            nn.Linear(solvent_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, graph_feat_size)
        )

        # Final prediction head
        # Input: graph (256) + solvent (256) + fp_extractor (512) = 1024
        total_input_dim = graph_feat_size + graph_feat_size + 2 * graph_feat_size
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_tasks)
        )
        
        # Store dimensions for forward pass
        self.solvent_dim = solvent_dim

    def forward(self, g, node_feats, edge_feats, fingerprints):
        # Handle missing edge features
        if edge_feats is None or edge_feats.size(0) == 0:
            num_edges = g.number_of_edges()
            edge_feats = torch.zeros((num_edges, 10), device=g.device)

        # Graph encoding
        node_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_feats)  # [B, 256]

        # Split fingerprints: solvent | smiles+extra
        solvent_feat = fingerprints[:, :self.solvent_dim]  # [B, 1024]
        smiles_extra_feat = fingerprints[:, self.solvent_dim:]  # [B, 2192]

        # Extract features
        solvent_out = self.solvent_extractor(solvent_feat)  # [B, 256]
        smiles_extra_out = self.fp_extractor(smiles_extra_feat)  # [B, 512]

        # Concatenate and predict
        combined_feats = torch.cat([graph_feats, solvent_out, smiles_extra_out], dim=1)
        return self.predict(combined_feats)


class GraphFingerprintsModelSimpleFC(nn.Module):
    """Model architecture for plqy and k prediction.
    
    Simpler architecture that combines:
    - AttentiveFP GNN for molecular graph encoding  
    - Simple FC network for all fingerprints combined
    
    This architecture was reverse-engineered from Model_plqy.pth state_dict:
        gnn.gnn_layers.0.* exists → num_layers=2
        readout.readouts.0/1/2.* → num_timesteps=3
        fp_fc.0.weight: [256, 2192] → fp_size=2192, no separate solvent branch
        predict.1.weight: [128, 512] → fusion_dim=512 (256+256)
    
    And from Model_k.pth:
        gnn.gnn_layers.0/1.* → num_layers=3
        readout.readouts.0.* only → num_timesteps=1
        fp_fc.0.weight: [256, 2192] → same fp_size
    
    Total input to predict layer: 256 (graph) + 256 (fp) = 512
    """
    def __init__(self, node_feat_size, edge_feat_size, fp_size,
                 graph_feat_size=256, num_layers=2, num_timesteps=2,
                 n_tasks=1, dropout=0.3):
        super(GraphFingerprintsModelSimpleFC, self).__init__()
        
        # Graph neural network
        self.gnn = AttentiveFPGNN(
            node_feat_size=node_feat_size,
            edge_feat_size=edge_feat_size,
            num_layers=num_layers,
            graph_feat_size=graph_feat_size,
            dropout=dropout
        )
        self.readout = AttentiveFPReadout(
            feat_size=graph_feat_size,
            num_timesteps=num_timesteps,
            dropout=dropout
        )
        
        # Simple FC for fingerprints
        self.fp_fc = nn.Sequential(
            nn.Linear(fp_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, graph_feat_size)
        )
        
        # Prediction head
        # Input: graph (256) + fp (256) = 512
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, n_tasks)
        )

    def forward(self, g, node_feats, edge_feats, fingerprints):
        # Handle missing edge features
        if edge_feats is None or edge_feats.size(0) == 0:
            num_edges = g.number_of_edges()
            edge_feats = torch.zeros((num_edges, 10), device=g.device)
            
        node_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_feats)
        fp_feats = self.fp_fc(fingerprints)
        combined_feats = torch.cat([graph_feats, fp_feats], dim=1)
        return self.predict(combined_feats)


# ============================================================================
# Dataset and DataLoader utilities
# ============================================================================

class MolecularDataset(Dataset):
    """Custom dataset for molecular data with fingerprints."""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """Collate function that handles optional LDS weights."""
    if len(batch[0]) == 5:
        graphs, fps, labels, masks, weights = zip(*batch)
        weights = torch.stack(weights)
    else:
        graphs, fps, labels, masks = zip(*batch)
        weights = None
    
    graphs = dgl.batch(graphs)
    fps = torch.stack(fps)
    labels = torch.stack(labels)
    masks = torch.stack(masks) if masks[0] is not None else None
    
    return graphs, fps, labels, masks, weights


def compute_lds_weights(targets, alpha=0.1, sigma=5):
    """Compute Label Distribution Smoothing weights.
    
    Uses kernel density estimation to identify under-represented
    regions of the target distribution and upweight those samples.
    This helps the model learn better on rare/extreme values.
    
    The alpha parameter controls the strength of reweighting:
    - alpha=0: no reweighting (all weights = 1)
    - alpha=0.1: light reweighting (abs model)
    - alpha=0.2: moderate reweighting (plqy model)
    - alpha=0.6: strong reweighting (k model - many sparse high-k regions)
    
    Args:
        targets: Array of target values
        alpha: Weight exponent (higher = more reweighting)
        sigma: KDE bandwidth (controls smoothness)
    
    Returns:
        Tensor of sample weights normalized to mean=1
    """
    if alpha == 0:
        return torch.ones(len(targets), dtype=torch.float32)
    
    targets = np.array(targets).reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=sigma).fit(targets)
    log_densities = kde.score_samples(targets)
    densities = np.exp(log_densities)
    weights = 1. / (densities ** alpha)
    weights = weights / np.mean(weights)  # Normalize to mean=1
    return torch.tensor(weights, dtype=torch.float32)


def load_fingerprints(fp_file):
    """Load fingerprint CSV file as tensor."""
    df = pd.read_csv(fp_file)
    return torch.tensor(df.values, dtype=torch.float32)


# ============================================================================
# Data Loading
# ============================================================================

def load_data_for_target(target, data_dir, atom_featurizer, bond_featurizer, use_cache=True):
    """Load and prepare training/validation data for a target property.
    
    Args:
        target: One of 'abs', 'em', 'plqy', 'k'
        data_dir: Path to data directory
        atom_featurizer: DGL atom featurizer
        bond_featurizer: DGL bond featurizer
        use_cache: Whether to use cached graph data
    
    Returns:
        Dictionary containing all data needed for training
    """
    data_dir = Path(data_dir)
    config = MODEL_CONFIGS[target]
    
    print(f"\n📂 Loading data for {target}...")
    
    # Load CSV data
    train_df = pd.read_csv(data_dir / f'train_{target}.csv')
    valid_df = pd.read_csv(data_dir / f'valid_{target}.csv')
    
    print(f"   Train samples: {len(train_df)}")
    print(f"   Valid samples: {len(valid_df)}")
    
    # Standardize target values
    scaler = StandardScaler()
    train_df[[target]] = scaler.fit_transform(train_df[[target]])
    valid_df[[target]] = scaler.transform(valid_df[[target]])
    
    # Load fingerprints
    train_fp_solvent = load_fingerprints(data_dir / f'train_sol_{target}.csv')
    valid_fp_solvent = load_fingerprints(data_dir / f'valid_sol_{target}.csv')
    train_fp_smiles = load_fingerprints(data_dir / f'train_smiles_{target}.csv')
    valid_fp_smiles = load_fingerprints(data_dir / f'valid_smiles_{target}.csv')
    
    # Extract extra features (columns 8:152 contain descriptors + scaffold flags)
    train_fp_extra = torch.tensor(train_df.iloc[:, 8:152].values, dtype=torch.float32)
    valid_fp_extra = torch.tensor(valid_df.iloc[:, 8:152].values, dtype=torch.float32)
    
    # Normalize numeric features (first 8 columns)
    scaler_num = MinMaxScaler()
    train_num = train_fp_extra[:, :8].numpy()
    valid_num = valid_fp_extra[:, :8].numpy()
    train_rest = train_fp_extra[:, 8:]
    valid_rest = valid_fp_extra[:, 8:]
    
    train_num_scaled = scaler_num.fit_transform(train_num)
    valid_num_scaled = scaler_num.transform(valid_num)
    
    train_fp_extra = torch.cat([
        torch.tensor(train_num_scaled, dtype=torch.float32), 
        train_rest
    ], dim=1)
    valid_fp_extra = torch.cat([
        torch.tensor(valid_num_scaled, dtype=torch.float32), 
        valid_rest
    ], dim=1)
    
    # Concatenate all fingerprints based on model type
    if config['model_type'] == 'attention_cnn':
        # For abs/em: keep solvent separate, combine smiles+extra
        train_fp = torch.cat([train_fp_solvent, train_fp_smiles, train_fp_extra], dim=1)
        valid_fp = torch.cat([valid_fp_solvent, valid_fp_smiles, valid_fp_extra], dim=1)
        solvent_dim = train_fp_solvent.shape[1]
        smiles_extra_dim = train_fp_smiles.shape[1] + train_fp_extra.shape[1]
        fp_size = None
    else:
        # For plqy/k: combine all into single fingerprint
        train_fp = torch.cat([train_fp_smiles, train_fp_extra], dim=1)
        valid_fp = torch.cat([valid_fp_smiles, valid_fp_extra], dim=1)
        solvent_dim = None
        smiles_extra_dim = None
        fp_size = train_fp.shape[1]
    
    print(f"   Total fingerprint dimensions: {train_fp.shape[1]}")
    
    # Compute LDS weights
    alpha = config['alpha']
    lds_weights = compute_lds_weights(
        train_df[[target]].values.flatten(), 
        alpha=alpha
    )
    print(f"   LDS alpha: {alpha}, weight range: [{lds_weights.min():.2f}, {lds_weights.max():.2f}]")
    
    # Create graph datasets
    def create_dataset(df, fp_data, name, weights=None):
        cache_path = data_dir / f'{name}_dataset_{target}.bin'
        
        dataset = MoleculeCSVDataset(
            df,
            smiles_to_graph=smiles_to_bigraph,
            node_featurizer=atom_featurizer,
            edge_featurizer=bond_featurizer,
            smiles_column='smiles',
            cache_file_path=str(cache_path),
            task_names=[target],
            load=use_cache,
            init_mask=True,
            n_jobs=1
        )
        
        combined_data = []
        for i, data_tuple in enumerate(dataset):
            if len(data_tuple) == 3:
                smiles, graph, label = data_tuple
                mask = None
            else:
                smiles, graph, label, mask = data_tuple
            
            fp = fp_data[i]
            
            if weights is not None:
                combined_data.append((graph, fp, label, mask, weights[i:i+1]))
            else:
                combined_data.append((graph, fp, label, mask))
        
        return combined_data
    
    train_data = create_dataset(train_df, train_fp, 'train', lds_weights)
    valid_data = create_dataset(valid_df, valid_fp, 'valid', None)
    
    return {
        'train_data': train_data,
        'valid_data': valid_data,
        'scaler': scaler,
        'scaler_num': scaler_num,
        'config': config,
        'solvent_dim': solvent_dim,
        'smiles_extra_dim': smiles_extra_dim,
        'fp_size': fp_size,
    }


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device, use_lds=True, pbar=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        graphs, fps, labels, masks, weights = batch
        
        graphs = graphs.to(device)
        fps = fps.to(device)
        labels = labels.to(device)
        
        node_feats = graphs.ndata['hv']
        edge_feats = graphs.edata.get('he', None)
        
        optimizer.zero_grad()
        predictions = model(graphs, node_feats, edge_feats, fps)
        
        # Compute loss with optional LDS weighting
        if use_lds and weights is not None:
            weights = weights.to(device)
            loss = (criterion(predictions, labels) * weights).mean()
        elif masks is not None:
            masks = masks.to(device)
            loss = (criterion(predictions, labels) * masks).mean()
        else:
            loss = criterion(predictions, labels).mean()
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if pbar is not None:
            pbar.update(1)
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """Validate model on validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            graphs, fps, labels, masks, _ = batch
            
            graphs = graphs.to(device)
            fps = fps.to(device)
            labels = labels.to(device)
            
            node_feats = graphs.ndata['hv']
            edge_feats = graphs.edata.get('he', None)
            
            predictions = model(graphs, node_feats, edge_feats, fps)
            
            if masks is not None:
                masks = masks.to(device)
                loss = (criterion(predictions, labels) * masks).mean()
            else:
                loss = criterion(predictions, labels).mean()
            
            total_loss += loss.item()
            num_batches += 1
            
            all_preds.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    return total_loss / num_batches, all_preds, all_labels


def train_model(target, data_dir, output_dir, epochs=200, patience=20, 
                device='cuda', use_cache=True):
    """Train a single property prediction model.
    
    Args:
        target: Property to predict ('abs', 'em', 'plqy', 'k')
        data_dir: Directory containing training data
        output_dir: Directory to save trained model
        epochs: Maximum training epochs
        patience: Early stopping patience
        device: 'cuda' or 'cpu'
        use_cache: Use cached graph data if available
    
    Returns:
        Dictionary with training results and metrics
    """
    print(f"\n{'='*60}")
    print(f"🚀 Training model for: {target.upper()}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Set random seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    # Initialize featurizers
    atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='hv')
    bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='he')
    n_feats = atom_featurizer.feat_size('hv')
    e_feats = bond_featurizer.feat_size('he')
    
    print(f"   Node features: {n_feats}, Edge features: {e_feats}")
    
    # Load data
    data = load_data_for_target(
        target, data_dir, atom_featurizer, bond_featurizer, use_cache
    )
    
    config = data['config']
    
    # Create data loaders
    train_dataset = MolecularDataset(data['train_data'])
    valid_dataset = MolecularDataset(data['valid_data'])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Initialize model based on architecture type
    if config['model_type'] == 'attention_cnn':
        model = GraphFingerprintsModelAttentionCNN(
            node_feat_size=n_feats,
            edge_feat_size=e_feats,
            solvent_dim=data['solvent_dim'],
            smiles_extra_dim=data['smiles_extra_dim'],
            graph_feat_size=GRAPH_FEAT_SIZE,
            num_layers=config['num_layers'],
            num_timesteps=config['num_timesteps'],
            n_tasks=1,
            dropout=config['dropout']
        ).to(device)
    else:
        model = GraphFingerprintsModelSimpleFC(
            node_feat_size=n_feats,
            edge_feat_size=e_feats,
            fp_size=data['fp_size'],
            graph_feat_size=GRAPH_FEAT_SIZE,
            num_layers=config['num_layers'],
            num_timesteps=config['num_timesteps'],
            n_tasks=1,
            dropout=config['dropout']
        ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Model parameters: {total_params:,} ({trainable_params:,} trainable)")
    
    # Loss and optimizer
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    
    use_lds = config['alpha'] > 0
    
    print(f"\n📈 Starting training (max {epochs} epochs, patience {patience})...")
    print(f"   Using LDS: {use_lds}\n")
    
    history = {'train_loss': [], 'val_loss': []}
    num_batches = len(train_loader)
    
    # Create epoch progress bar
    epoch_pbar = tqdm(range(1, epochs + 1), desc="Epochs", unit="epoch", position=0)
    
    for epoch in epoch_pbar:
        # Create batch progress bar for this epoch
        batch_pbar = tqdm(total=num_batches, desc=f"  Epoch {epoch}", 
                          unit="batch", position=1, leave=False)
        
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, use_lds, pbar=batch_pbar
        )
        batch_pbar.close()
        
        val_loss, val_preds, val_labels = validate(
            model, valid_loader, criterion, device
        )
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            status = "✓ best"
        else:
            epochs_without_improvement += 1
            status = ""
        
        # Update epoch progress bar with metrics
        epoch_pbar.set_postfix({
            'train': f'{train_loss:.4f}',
            'val': f'{val_loss:.4f}',
            'best': f'{best_val_loss:.4f}',
            'status': status
        })
        
        # Early stopping
        if epochs_without_improvement >= patience:
            epoch_pbar.close()
            print(f"\n⏹️  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break
    
    else:
        epoch_pbar.close()
    
    # Load best model and evaluate
    model.load_state_dict(best_model_state)
    
    # Final validation metrics (in original scale)
    _, val_preds, val_labels = validate(model, valid_loader, criterion, device)
    val_preds_orig = data['scaler'].inverse_transform(val_preds)
    val_labels_orig = data['scaler'].inverse_transform(val_labels)
    
    mae = mean_absolute_error(val_labels_orig, val_preds_orig)
    rmse = np.sqrt(mean_squared_error(val_labels_orig, val_preds_orig))
    r2 = r2_score(val_labels_orig, val_preds_orig)
    
    print(f"\n📊 Final Validation Metrics ({target}):")
    print(f"   MAE:  {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R²:   {r2:.4f}")
    
    # Save model
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f'Model_{target}.pth'
    torch.save(best_model_state, model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Calculate training time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    print(f"⏱️  Training time: {time_str}")
    
    return {
        'target': target,
        'best_val_loss': best_val_loss,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'epochs_trained': epoch,
        'training_time': elapsed_time,
        'history': history,
    }


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train Fluor-RLAT property prediction models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python training.py --target abs                    # Train absorption model
  python training.py --target all                    # Train all 4 models
  python training.py --target em --epochs 100        # Train emission with 100 epochs
  python training.py --target plqy --no-cache        # Train without cached graphs
        """
    )
    
    parser.add_argument(
        '--target', 
        type=str, 
        default='all',
        choices=['abs', 'em', 'plqy', 'k', 'all'],
        help='Target property to train (default: all)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Directory containing training data (default: ./data)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Directory to save trained models (default: current directory)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='Maximum training epochs (default: 200)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=20,
        help='Early stopping patience (default: 20)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (default: auto-detect cuda/cpu)'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Do not use cached graph data'
    )
    
    args = parser.parse_args()
    
    # Auto-detect device
    # Note: DGL does not support MPS (Apple Silicon GPU), so we fall back to CPU
    if args.device is None:
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"🖥️  Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("🖥️  Using CPU (DGL does not support Apple MPS)")
            else:
                print("🖥️  Using CPU")
    else:
        device = args.device
        print(f"🖥️  Using device: {device}")
    
    # Determine which targets to train
    if args.target == 'all':
        targets = ['abs', 'em', 'plqy', 'k']
    else:
        targets = [args.target]
    
    print(f"\n🎯 Training targets: {targets}")
    print(f"📁 Data directory: {args.data_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    
    # Train models
    total_start_time = time.time()
    results = []
    for target in targets:
        result = train_model(
            target=target,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            patience=args.patience,
            device=device,
            use_cache=not args.no_cache
        )
        results.append(result)
    
    total_elapsed = time.time() - total_start_time
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 TRAINING SUMMARY")
    print(f"{'='*60}")
    for r in results:
        # Format individual training time
        t = r['training_time']
        h, rem = divmod(t, 3600)
        m, s = divmod(rem, 60)
        time_str = f"{int(h)}h {int(m)}m {int(s)}s"
        
        print(f"\n{r['target'].upper()}:")
        print(f"   Epochs: {r['epochs_trained']}, Time: {time_str}")
        print(f"   MAE: {r['mae']:.4f}, RMSE: {r['rmse']:.4f}, R²: {r['r2']:.4f}")
    
    # Total time
    total_h, total_rem = divmod(total_elapsed, 3600)
    total_m, total_s = divmod(total_rem, 60)
    total_time_str = f"{int(total_h)}h {int(total_m)}m {int(total_s)}s"
    
    print(f"\n{'='*60}")
    print(f"⏱️  Total training time: {total_time_str}")
    print(f"✅ Training complete!")
    return results


if __name__ == '__main__':
    main()
