# Copilot Instructions for Fluor-tools

## Project Map

### Fluor-RLAT (Property Prediction - Standalone)
Predicts absorption, emission, quantum yield, and molar absorptivity.
- **Entry**: `Fluor-RLAT/run.py`
- **Inference**: `Fluor-RLAT/02_property_prediction.py` (loads pretrained models, runs forward pass)
- **Models**: `Model_abs.pth`, `Model_em.pth`, `Model_k.pth`, `Model_plqy.pth`
- **Data**: `Fluor-RLAT/data/*`, `Fluor-RLAT/input/*`
- **Outputs**: `Fluor-RLAT/result/target_predictions*.csv`
- **Scripts**: `01_data_preprocessing.py`, `02_property_prediction.py`, `03_file_merge.py` (preprocessing → inference → merge/cleanup)
- **Training Notebook**: `colab_training/Fluor_RLAT_Training.ipynb` (GPU-accelerated training with checkpointing)
- **Prediction Notebook**: `colab_training/Fluor_RLAT_Predict.ipynb` (lightweight inference-only notebook)

### Google Colab Paths (Google Drive)
When running notebooks in Google Colab, files are stored on Google Drive:
- **Trained Models**: `/content/drive/MyDrive/fluor_models/` (Model_abs.pth, Model_em.pth, etc.)
- **Checkpoints**: `/content/drive/MyDrive/fluor_checkpoints/` (checkpoint_abs.pth, etc. with training state)
- **Merged Training Data**: `/content/drive/MyDrive/fluor_models/merged_training_data/` (expanded/cleaned datasets; auto-detected by training notebook)
- **Pretrained Models**: `./fluor_tools/Fluor-RLAT/` (cloned from repo)

### Training Data Priority
The training notebook (`Fluor_RLAT_Training.ipynb`) automatically checks for newer training data:
1. **First choice**: `/content/drive/MyDrive/fluor_models/merged_training_data/` — if this directory exists and contains train files, they are **copied into the cloned repo's data folder**, replacing the originals. The repo is temporary (cloned fresh each session), so overwriting is safe.
2. **Fallback**: `./fluor_tools/Fluor-RLAT/data` — original repository data used as-is when no merged data exists on Drive.
This allows retraining with expanded or cleaned datasets by simply uploading new files to the Google Drive `merged_training_data` folder. The `Create_Training_Data.ipynb` notebook's deploy cell (Cell 25) copies merged files there with standard `train_*.csv` naming.

### NIRFluor-opt and web (⚠️ Contains BOTH modules)

#### Module 1: NIRFluor-opt (Structure Generation)
Rule-based pipeline for generating NIR dye candidates.
- **Entry**: `run.py`, `processing.py`
- **Web**: `app.py` (Flask)
- **Input**: `input/target_m.csv`
- **Outputs**: `results/new_molecules.csv` (with `predicted_label` for NIR classification), fragment CSVs, images

#### Module 2: Fluor-RLAT Copy (Property Prediction for Web)
Duplicate of standalone Fluor-RLAT for single-target web predictions.
- **Entry**: `predict/run.py`
- **Scripts**: `predict/01_data_preprocessing.py`, `predict/02_property_prediction.py`, `predict/03_file_merge.py`
- **Input**: `predict/input/target.csv`
- **Outputs**: `predict/result/target_predictions.csv`

### figure_code
Plotting notebooks and source data for manuscript figures; not part of runtime.

## Why Two Modules
- NIRFluor-opt targets structure generation/optimization: starting from a supplied SMILES, it fragments the molecule, selects transformation rules by MACCS similarity, applies substitutions, and surfaces candidate near-infrared dyes (predicted_label==1) for user review. It is rule-based and focuses on proposing new molecules.
- Fluor-RLAT (a.k.a. Fluor-pred) handles property prediction: given SMILES and solvent, it computes descriptors/fingerprints, runs pretrained AttentiveFP + fingerprint models, and outputs four photophysical properties. It does not generate structures—only evaluates them.
- Together: run NIRFluor-opt to generate candidate NIR dyes, then run Fluor-RLAT (or predict/) to score their absorption/emission/yield/extinction; the web app wires these flows for interactive use.

## Naming Note
- Fluor-opt was renamed to NIRFluor-opt; Fluor-pred was renamed to Fluor-RLAT. Code and docs may use either pair interchangeably.
- Confusing folder structure: "NIRFluor-opt and web/" contains BOTH the NIRFluor-opt pipeline (at root) AND a copy of Fluor-RLAT (in predict/ subfolder). Running predict/run.py executes Fluor-RLAT property prediction, NOT NIRFluor-opt structure generation.

---

## ⚠️ CRITICAL: Two Different Model Architectures

**The four Fluor-RLAT models use TWO DIFFERENT architectures. This is the most common source of bugs.**

### Architecture 1: GraphFingerprintsModel (for abs and em)
Uses CNN attention for fingerprints + separate solvent extractor.

**Layer structure (verified from state_dict):**
```
fp_extractor.conv_feat: Conv1d(1, 256, kernel_size=3)
fp_extractor.conv_attn: Conv1d(1, 256, kernel_size=3)
solvent_extractor.0: Linear(1024, 256)
solvent_extractor.3: Linear(256, 256)
predict.1: Linear(1024, 128)  ← input is 1024!
predict.3: Linear(128, 1)
```

**Fingerprint processing:**
- Splits input: `solvent_feat = fp[:, :1024]`, `smiles_extra_feat = fp[:, 1024:]`
- `solvent_extractor(solvent_feat)` → 256-dim
- `fp_extractor(smiles_extra_feat)` → 512-dim (2×256 from attention + pooling)
- Concatenate: `[graph(256) + solvent(256) + fp_extractor(512)] = 1024-dim`

### Architecture 2: GraphFingerprintsModelFC (for plqy and k)
Uses simple FC for ALL fingerprints combined (no separate solvent processing).

**Layer structure (verified from state_dict):**
```
fp_fc.0: Linear(2192, 256)
fp_fc.3: Linear(256, 256)
predict.1: Linear(512, 128)  ← input is 512!
predict.3: Linear(128, 1)
```

**Fingerprint processing:**
- Processes ALL fingerprints together: `fp_fc(fingerprints)` → 256-dim
- Concatenate: `[graph(256) + fp_fc(256)] = 512-dim`

### Model Hyperparameters (EXACT values from 02_property_prediction.py)

| Target | num_layers | num_timesteps | dropout | alpha (LDS) | Architecture |
|--------|------------|---------------|---------|-------------|--------------|
| **abs** | 2 | 2 | 0.3 | 0.1 | GraphFingerprintsModel |
| **em** | 3 | 1 | 0.3 | 0.0 | GraphFingerprintsModel |
| **plqy** | 2 | 3 | 0.4 | 0.2 | GraphFingerprintsModelFC |
| **k** | 3 | 1 | 0.3 | 0.6 | GraphFingerprintsModelFC |

### State Dict Key Differences

**Model_abs.pth / Model_em.pth contain:**
- `fp_extractor.conv_feat.*`, `fp_extractor.conv_attn.*`
- `solvent_extractor.0.*`, `solvent_extractor.3.*`
- `predict.1.weight` shape: `[128, 1024]`

**Model_plqy.pth / Model_k.pth contain:**
- `fp_fc.0.*`, `fp_fc.3.*` (NO fp_extractor or solvent_extractor!)
- `predict.1.weight` shape: `[128, 512]`

---

## Fingerprint Structure and Order

**CRITICAL: Fingerprint order matters! Must be exactly:**
```
[solvent_fp(1024), smiles_fp(1024), numeric_scaled(8), scaffold_flags(136)] = 2192 total
```

### Data Files
- **Solvent fingerprints**: `train_sol_*.csv` (1024 columns) - Morgan FP radius=2, nBits=1024
- **Molecule fingerprints**: `train_smiles_*.csv` (1024 columns) - Morgan FP radius=2, nBits=1024
- **Extra features**: columns 8-151 of `train_*.csv` (144 columns = 8 numeric + 136 scaffold)

### Column Structure in train_abs.csv (152 total columns)
```
Columns 0-7:   split, smiles, solvent, abs, em, plqy, k, tag_name (metadata)
Columns 8-15:  solvent_num, tag, Molecular_Weight, LogP, TPSA, Double_Bond_Count, Ring_Count, unimol_plus (8 numeric)
Columns 16-151: fragment_1 to fragment_136 (136 scaffold binary flags)
```

### Preprocessing Steps
1. **Numeric features** (columns 8:16): Apply `MinMaxScaler` fit on training data
2. **Scaffold flags** (columns 16:152): Use as-is (binary 0/1)
3. **Concatenate**: `[solvent_fp, smiles_fp, numeric_scaled, scaffold_flags]`
4. **Labels**: Apply `StandardScaler` fit on training data, inverse transform predictions

---

## Model Class Definitions (for training/inference)

### FingerprintAttentionCNN (used by abs/em)
```python
class FingerprintAttentionCNN(nn.Module):
    def __init__(self, input_dim, conv_channels=256):
        super().__init__()
        self.conv_feat = nn.Conv1d(1, conv_channels, kernel_size=3, padding=1)
        self.conv_attn = nn.Conv1d(1, conv_channels, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=-1)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, D]
        feat_map = self.conv_feat(x)
        attn_map = self.conv_attn(x)
        attn_weights = self.softmax(attn_map)
        attn_out = torch.sum(feat_map * attn_weights, dim=-1)  # [B, C]
        pooled = self.pool(feat_map).squeeze(-1)               # [B, C]
        return torch.cat([attn_out, pooled], dim=1)            # [B, 2C]
```

### GraphFingerprintsModel (for abs/em)
```python
class GraphFingerprintsModel(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, solvent_dim, smiles_extra_dim,
                 graph_feat_size=256, num_layers=2, num_timesteps=2, n_tasks=1, dropout=0.3):
        super().__init__()
        self.solvent_dim = solvent_dim  # Must store for forward pass!
        self.gnn = AttentiveFPGNN(...)
        self.readout = AttentiveFPReadout(...)
        self.fp_extractor = FingerprintAttentionCNN(smiles_extra_dim, conv_channels=graph_feat_size)
        self.solvent_extractor = nn.Sequential(
            nn.Linear(solvent_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, graph_feat_size)
        )
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size * 4, 128),  # 256+256+512=1024
            nn.ReLU(),
            nn.Linear(128, n_tasks)
        )
```

### GraphFingerprintsModelFC (for plqy/k)
```python
class GraphFingerprintsModelFC(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, fp_size,
                 graph_feat_size=256, num_layers=2, num_timesteps=2, n_tasks=1, dropout=0.3):
        super().__init__()
        self.gnn = AttentiveFPGNN(...)
        self.readout = AttentiveFPReadout(...)
        self.fp_fc = nn.Sequential(
            nn.Linear(fp_size, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, graph_feat_size)
        )  # Only 4 layers! indices 0,1,2,3
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size * 2, 128),  # 256+256=512
            nn.ReLU(),
            nn.Linear(128, n_tasks)
        )
```

---

## Training Data

### Fluor-RLAT (Property Prediction)
- **Primary source**: 00_FluoDB.csv (49,832 total records)
- **Split**: train (~22k), valid (~5k), test (~5k) per property
- **Features per sample**: 152 columns in train_abs.csv + 1024 solvent FP + 1024 molecule FP = 2192 total fingerprint dims

### Data Files Per Target
```
train_{target}.csv      - main data with labels and descriptors (152 cols)
train_sol_{target}.csv  - solvent Morgan fingerprints (1024 cols)
train_smiles_{target}.csv - molecule Morgan fingerprints (1024 cols)
valid_*.csv, test_*.csv  - same structure
```

### Targets and Ranges
- **abs**: absorption wavelength (nm), range ~300-900
- **em**: emission wavelength (nm), range ~350-1000
- **plqy**: quantum yield, range 0.0-1.0
- **k**: log₁₀(molar absorptivity), range ~3.0-5.5

---

## Colab Training Notebook

**Location**: `colab_training/Fluor_RLAT_Training.ipynb`

### Features
- GPU-accelerated training with CUDA support
- Checkpoint saving/resumption (survives Colab disconnects)
- Automatic early stopping (patience=20)
- LDS (Label Distribution Smoothing) for imbalanced data
- tqdm progress bars
- Comparison cell for pretrained vs newly trained models

### Key Configuration
```python
# Optimizer: AdamW (weight_decay=5e-5) — decoupled L2 regularization
# Gradient clipping: max_norm=5.0
# LR scheduler: CosineAnnealingWarmRestarts (T_0=20, T_mult=2, eta_min=1e-6)

MODEL_CONFIGS = {
    'abs':  {'num_layers': 2, 'num_timesteps': 2, 'dropout': 0.3, 'alpha': 0.1, 'model_class': 'GraphFingerprintsModel'},
    'em':   {'num_layers': 3, 'num_timesteps': 1, 'dropout': 0.3, 'alpha': 0.0, 'model_class': 'GraphFingerprintsModel'},
    'plqy': {'num_layers': 2, 'num_timesteps': 3, 'dropout': 0.4, 'alpha': 0.2, 'model_class': 'GraphFingerprintsModelFC'},
    'k':    {'num_layers': 3, 'num_timesteps': 1, 'dropout': 0.3, 'alpha': 0.6, 'model_class': 'GraphFingerprintsModelFC'},
}
```

### Expected Predictions (test molecule: BODIPY in toluene)
```
SMILES: C2=C1C7=C(C(=[N+]1[B-]([N]3C2=C5C(=C3C4=CC=CC=C4)C=CC=C5)(F)F)C6=CC=CC=C6)C=CC=C7
Solvent: CC1=CC=CC=C1 (toluene)

Expected results:
- Absorption: ~639.89 nm
- Emission: ~660.38 nm
- Quantum Yield: ~0.76
- Log ε: ~5.0
```

---

## Common Pitfalls and Bugs

### 1. Wrong Model Architecture Selection
**Bug**: Using GraphFingerprintsModel for plqy/k or vice versa
**Symptom**: State dict loading fails with mismatched keys
**Fix**: Check `model_class` in config, use correct class

### 2. Wrong Fingerprint Order
**Bug**: `[smiles_fp, solvent_fp, extra]` instead of `[solvent_fp, smiles_fp, extra]`
**Symptom**: Predictions are wildly wrong (1000+ nm absorption)
**Fix**: Always use order: `[solvent(1024), smiles(1024), numeric(8), scaffold(136)]`

### 3. Missing MinMaxScaler on Numeric Features
**Bug**: Using raw numeric features without scaling
**Symptom**: Predictions are wrong
**Fix**: Fit MinMaxScaler on train_df.iloc[:, 8:16], transform inference data

### 4. Missing StandardScaler Inverse Transform
**Bug**: Returning raw model output without inverse transform
**Symptom**: Predictions are normalized values near 0
**Fix**: `label_scaler.inverse_transform([[raw_pred]])[0, 0]`

### 5. fp_fc Layer Count Mismatch (plqy/k models)
**Bug**: Defining fp_fc with 6 layers instead of 4
**Symptom**: State dict keys don't match (fp_fc.4, fp_fc.5 not found)
**Fix**: fp_fc should be: `Linear→ReLU→Dropout→Linear` (indices 0,1,2,3 only)

### 6. Wrong predict Layer Input Dimension
**Bug**: Using 1024 input for plqy/k predict layer
**Symptom**: Size mismatch error
**Fix**: abs/em use 1024 (graph+solvent+fp_extractor), plqy/k use 512 (graph+fp_fc)

---

## Detailed Pipeline & Normalization

### Property Prediction Pipeline (Fluor-RLAT) - Step by Step

#### Step 1: Preprocessing (`01_data_preprocessing.py`)

**Input**: `input/target.csv` with columns `smiles, solvent`

**Operations**:
1. **Solvent Mapping**: Maps solvent SMILES → integer `solvent_num` using `data/00_solvent_mapping.csv`
2. **RDKit Descriptors**: Computes 6 molecular descriptors:
   - `Molecular_Weight`: `Descriptors.MolWt(mol)`
   - `LogP`: `Descriptors.MolLogP(mol)`
   - `TPSA`: `Descriptors.TPSA(mol)`
   - `Double_Bond_Count`: count of double + aromatic bonds via RDKit `sum(1 for bond in mol.GetBonds() if bond.GetBondType() == DOUBLE or bond.GetIsAromatic())`
   - `Ring_Count`: `mol.GetRingInfo().NumRings()`
   - `unimol_plus`: placeholder (set to 0)
3. **Scaffold Tags**: Checks 136 substructures from `data/00_mmp_substructure.csv`, creates binary flags `fragment_1..fragment_136`
4. **Morgan Fingerprints**: Generates 1024-bit fingerprints for both molecule and solvent:
   ```python
   AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
   ```

**Outputs**:
- `input/input.csv` - 152 columns (metadata + 8 numeric + 136 scaffold)
- `input/target_smiles_morgan.csv` - 1024 molecule fingerprint columns
- `input/target_sol_morgan.csv` - 1024 solvent fingerprint columns

#### Step 2: Inference (`02_property_prediction.py`)

**For each target (abs, em, plqy, k):**

1. **Load Training Data** (for scaler fitting):
   ```python
   train_df = pd.read_csv(f'data/train_{target}.csv')
   train_sol = pd.read_csv(f'data/train_sol_{target}.csv')
   train_smiles = pd.read_csv(f'data/train_smiles_{target}.csv')
   ```

2. **Fit Feature Scaler** (MinMaxScaler on numeric features only):
   ```python
   feature_scaler = MinMaxScaler()
   feature_scaler.fit(train_df.iloc[:, 8:16])  # 8 numeric columns
   ```

3. **Fit Label Scaler** (StandardScaler on target values):
   ```python
   label_scaler = StandardScaler()
   label_scaler.fit(train_df[[target]].values)
   ```

4. **Prepare Input Features**:
   ```python
   # Scale numeric features (columns 8:16)
   numeric_scaled = feature_scaler.transform(input_df.iloc[:, 8:16])
   
   # Scaffold flags (columns 16:152) - no scaling, use as-is
   scaffold = input_df.iloc[:, 16:152].values
   
   # Concatenate in EXACT order:
   fingerprints = np.hstack([
       solvent_fp,      # 1024 columns from target_sol_morgan.csv
       smiles_fp,       # 1024 columns from target_smiles_morgan.csv  
       numeric_scaled,  # 8 columns, MinMaxScaled
       scaffold         # 136 columns, binary 0/1
   ])  # Total: 2192 dimensions
   ```

5. **Build DGL Graph** (for GNN component):
   ```python
   from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
   
   graph = smiles_to_bigraph(
       smiles,
       node_featurizer=CanonicalAtomFeaturizer(),  # 39 features per atom
       edge_featurizer=CanonicalBondFeaturizer()   # 10 features per bond
   )
   ```

6. **Model Forward Pass**:
   ```python
   # For abs/em (GraphFingerprintsModel):
   #   - Splits fingerprints: solvent[:1024] vs rest[1024:]
   #   - Processes separately, concatenates to 1024-dim
   
   # For plqy/k (GraphFingerprintsModelFC):
   #   - Processes all 2192 fingerprints together
   #   - Concatenates to 512-dim
   
   raw_prediction = model(graph, fingerprints)
   ```

7. **Inverse Transform** (convert normalized output back to real units):
   ```python
   final_prediction = label_scaler.inverse_transform([[raw_prediction]])[0, 0]
   ```

#### Step 3: Merge (`03_file_merge.py`)

Concatenates `target_predictions_abs.csv`, `target_predictions_em.csv`, etc. into single `target_predictions.csv` and cleans up `.bin` cache files.

---

### Normalization Details

#### Feature Normalization (MinMaxScaler)

**Applied to**: 8 numeric descriptor columns (indices 8:16)
**NOT applied to**: Fingerprints (binary 0/1) or scaffold flags (binary 0/1)

```python
# Columns 8-15 in train_*.csv:
# solvent_num, tag, Molecular_Weight, LogP, TPSA, Double_Bond_Count, Ring_Count, unimol_plus

feature_scaler = MinMaxScaler()
feature_scaler.fit(train_df.iloc[:, 8:16])  # Fit on TRAINING data only!

# Transform both train and inference data
train_numeric_scaled = feature_scaler.transform(train_df.iloc[:, 8:16])
input_numeric_scaled = feature_scaler.transform(input_df.iloc[:, 8:16])
```

**Why MinMaxScaler?** Scales values to [0, 1] range, preserving relative differences. Good for neural networks because it keeps gradients stable.

#### Label Normalization (StandardScaler)

**Applied to**: Target property values (abs, em, plqy, k)
**Purpose**: Neural network outputs are trained on normalized labels for stable training

```python
label_scaler = StandardScaler()
label_scaler.fit(train_df[[target]].values)  # Fit on TRAINING data only!

# During training: normalize labels
y_train_normalized = label_scaler.transform(train_df[[target]].values)

# During inference: inverse transform predictions
raw_output = model(graph, fingerprints)  # Returns normalized value
final_prediction = label_scaler.inverse_transform([[raw_output]])[0, 0]
```

**Why StandardScaler?** Converts to zero-mean, unit-variance distribution. Essential because:
- Different targets have vastly different ranges (abs: 300-900 nm vs plqy: 0-1)
- Training loss converges better with normalized targets
- **CRITICAL**: Must inverse transform predictions to get real units!

#### What Gets Normalized vs Not

| Component | Normalization | Reason |
|-----------|---------------|--------|
| Solvent fingerprints (1024) | None | Binary 0/1, already normalized |
| Molecule fingerprints (1024) | None | Binary 0/1, already normalized |
| Numeric descriptors (8) | MinMaxScaler | Continuous values, different scales |
| Scaffold flags (136) | None | Binary 0/1, already normalized |
| Target labels | StandardScaler | Continuous, vastly different scales |

---

### NIR Optimization Pipeline (NIRFluor-opt) - Step by Step

#### Step 1: Fragment Input Molecule
```python
from rdkit.Chem import rdMMPA

# Cut molecule at 1, 2, and 3 bonds
for cut_count in [1, 2, 3]:
    fragments = rdMMPA.FragmentMol(mol, maxCuts=cut_count, ...)
```

#### Step 2: Filter Transformation Rules by Similarity
```python
# Load rules from data/transformation_rules_maccs.csv
# Each rule has: from_smiles, to_smiles, maccs_fingerprint

# Calculate MACCS similarity between input and rules
input_maccs = MACCSkeys.GenMACCSKeys(mol)
for rule in rules:
    similarity = DataStructs.TanimotoSimilarity(input_maccs, rule.maccs)
    if similarity > threshold:
        applicable_rules.append(rule)
```

#### Step 3: Apply Substitutions
```python
# Replace fragments with transformation rule outputs
new_molecules = []
for rule in applicable_rules:
    new_mol = apply_substitution(input_mol, rule)
    new_molecules.append(new_mol)
```

#### Step 4: NIR Classification
```python
# Generate 2048-bit Morgan fingerprint
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

# Load stacking classifier (REQUIRES scikit-learn==1.0.2!)
model = joblib.load('stacking_model_full.pkl')

# Predict: 1 = NIR dye candidate, 0 = not NIR
predicted_label = model.predict([fp])[0]
probability = model.predict_proba([fp])[0, 1]
```

#### Step 5: Output
- `results/new_molecules.csv` with `predicted_label` column
- Only molecules with `predicted_label == 1` are NIR candidates

---

## Environment Setup

### Critical Version Constraints
- **scikit-learn==1.0.2** (stacking model pickled with this version; 1.7+ breaks imports)
- Python 3.7-3.10 recommended
- PyTorch 1.13.1+ with CUDA (optional but recommended)
- DGL 1.1.2+cu117, dgllife 0.2.8

### Required Dependencies
```
torch, dgl, dgllife, rdkit, pandas, numpy, scikit-learn==1.0.2, lightgbm, xgboost, tqdm
```

### Install Order
1. PyTorch with CUDA (if GPU available)
2. DGL matching CUDA version
3. dgllife
4. Other deps

---

## FLAME Package (Upstream Dependency — Unavailable)

The preprocessing scripts (`01_data_preprocessing.py`) contain a `try: from FLAME.flsf.scaffold import scaffold` import wrapped in a `try/except ImportError` block. This always falls through to the hardcoded fallback dictionary.

**What FLAME is**: "FLuorophore design Acceleration ModulE" — a modular AI framework for fluorophore design from Zhu, Y. et al. (2025), *Nature Communications*, 16, 3598. **FLSF** ("FLuorescence prediction with FluoroScaFfold-driven model") is its sub-module for scaffold-based fingerprints.

**What `scaffold` contains**: A Python dictionary mapping 16 fluorescent scaffold class names (`'SquaricAcid'`, `'Coumarin'`, `'BODIPY'`, `'Cyanine'`, etc.) to lists of SMILES substructure patterns. These are used for RDKit substructure matching to produce the 136 binary `fragment_1..fragment_136` columns.

**Why it can't be installed**: The source code was never publicly released — not on PyPI, not on GitHub. The Zenodo deposit for the paper only contains model weights (950 MB), not Python source. The authors' code remains internal to their research group.

**What we did**: Removed the dead `try/except` import from the training notebook (`Fluor_RLAT_Training.ipynb`) and kept only the hardcoded dictionary. The standalone scripts (`Fluor-RLAT/01_data_preprocessing.py`, `predict/01_data_preprocessing.py`) still have the `try/except` wrapper for compatibility with the original authors' environment.

---

## NIRFluor-opt NIR Classification

### Stacking Ensemble Model
- **Input**: 2048-bit Morgan fingerprint (radius=2)
- **Model**: StackingClassifier (GradientBoosting, LogisticRegression, LightGBM, XGBoost)
- **Output**: binary label (0/1) + probability
- **File**: stacking_model_full.pkl
- **Critical**: Must use scikit-learn==1.0.2

---

## File Responsibilities

| File | Purpose |
|------|---------|
| `Fluor-RLAT/run.py` | Sample driver for property prediction |
| `Fluor-RLAT/02_property_prediction.py` | Defines models, loads weights, runs inference |
| `NIRFluor-opt and web/processing.py` | NIR optimization pipeline |
| `NIRFluor-opt and web/app.py` | Flask web interface |
| `colab_training/Fluor_RLAT_Training.ipynb` | GPU training notebook |

---

## Repository Change Log

### February 2026 - Training Improvements (Batch 1)
- Replaced `optim.Adam` with `optim.AdamW` (weight_decay=5e-5) for proper L2 regularization
- Added gradient clipping (`clip_grad_norm_`, max_norm=5.0) in `train_epoch` for training stability
- Replaced `ReduceLROnPlateau` with `CosineAnnealingWarmRestarts` (T_0=20, T_mult=2, eta_min=1e-6)
- Created `colab_training/Model_Improvement_Research.md` with full references and future plans
- References: Loshchilov & Hutter (2019, AdamW), Loshchilov & Hutter (2017, SGDR), Xiong et al. (2020, AttentiveFP)

### February 2026 - Training Notebook & Architecture Documentation
- Created `colab_training/Fluor_RLAT_Training.ipynb` with full training pipeline
- Discovered and documented TWO DIFFERENT model architectures (abs/em vs plqy/k)
- Verified exact state_dict keys and layer shapes for all 4 models
- Fixed fingerprint order: must be `[solvent, smiles, numeric_scaled, scaffold]`
- Documented MinMaxScaler requirement for numeric features (columns 8:16)
- Added comprehensive model class definitions matching pretrained weights

### February 2026 - Initial Setup & Cleanup
- Fixed path resolution issues: anchored all run.py scripts to their own directories
- Padded fragment processing in processing.py for molecules with fewer than 4 parts
- Fixed predict/ folder scripts: converted `./predict/` paths to `./`
- Environment version constraints documented: scikit-learn==1.0.2 required
- Translated all Chinese filenames and content to English

---

## Quick Reference: Model Loading

```python
# For abs/em
model = GraphFingerprintsModel(
    node_feat_size=39, edge_feat_size=10,
    solvent_dim=1024, smiles_extra_dim=1168,  # 1024+144
    graph_feat_size=256, num_layers=2, num_timesteps=2, dropout=0.3
)
model.load_state_dict(torch.load('Model_abs.pth'))

# For plqy/k
model = GraphFingerprintsModelFC(
    node_feat_size=39, edge_feat_size=10,
    fp_size=2192,  # 1024+1024+144
    graph_feat_size=256, num_layers=2, num_timesteps=3, dropout=0.4
)
model.load_state_dict(torch.load('Model_plqy.pth'))
```
