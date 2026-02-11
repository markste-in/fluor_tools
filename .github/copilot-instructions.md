# Copilot Instructions for Fluor-tools

## Project Map

### Fluor-RLAT (Property Prediction - Standalone)
Predicts absorption, emission, quantum yield, and molar absorptivity.
- **Entry**: `Fluor-RLAT/run.py`
- **Models**: `Model_abs.pth`, `Model_em.pth`, `Model_k.pth`, `Model_plqy.pth`
- **Data**: `Fluor-RLAT/data/*`, `Fluor-RLAT/input/*`
- **Outputs**: `Fluor-RLAT/result/target_predictions*.csv`
- **Scripts**: `01_data_preprocessing.py`, `02_property_prediction.py`, `03_file_merge.py` (preprocessing → inference → merge/cleanup)

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
- Fluor-RLAT (a.k.a. Fluor-pred) handles property prediction: given SMILES and solvent, it computes descriptors/fingerprints, runs pretrained AttentiveFP + CNN models, and outputs four photophysical properties. It does not generate structures—only evaluates them.
- Together: run NIRFluor-opt to generate candidate NIR dyes, then run Fluor-RLAT (or predict/) to score their absorption/emission/yield/extinction; the web app wires these flows for interactive use.

## Naming Note
- Fluor-opt was renamed to NIRFluor-opt; Fluor-pred was renamed to Fluor-RLAT. Code and docs may use either pair interchangeably.
- Confusing folder structure: "NIRFluor-opt and web/" contains BOTH the NIRFluor-opt pipeline (at root) AND a copy of Fluor-RLAT (in predict/ subfolder). Running predict/run.py executes Fluor-RLAT property prediction, NOT NIRFluor-opt structure generation.

## High-Level Data Flow
- Property prediction (Fluor-RLAT and predict/):
  1) 01_data_preprocessing.py maps solvent names to numeric ids, computes RDKit descriptors, scaffold tags, and writes input/input.csv plus derived fingerprints (target_smiles_morgan.csv, target_sol_morgan.csv).
  2) 02_property_prediction.py loads pretrained graph+fingerprint models (AttentiveFP with CNN over fingerprints) for abs/em/plqy/k. It standardizes numeric features, runs inference, and saves per-target CSVs under result/.
  3) 03_file_merge.py concatenates the four prediction CSVs into result/target_predictions.csv and deletes .bin cache files.
- NIR optimization (processing.py):
  1) Cleans previous results (results/*.csv, results/molecule_images/*.png, results/rules_images/*.png).
  2) Reads input/target_m.csv, fragments the SMILES via rdMMPA across cut counts (1-3), saves fragments to results/target_fragment.csv; aborts if molecule cannot be fragmented.
  3) Computes MACCS for target, filters transformation rules from data/transformation_rules_maccs.csv by Tanimoto similarity > user-supplied similarity_value; writes results/target_similary_rules.csv, expands to results/target_rules.csv (and target_rules_replace.csv when H replacements appear).
  4) Applies mapped node replacements to fragment combinations to generate candidate motifs; outputs results/new_m_replace.csv plus used_mapping_pairs.csv.
  5) Post-processes new_m_replace.csv (column reassignment by marker [*:1-3], pruning sparse rows, filling missing substitutions) then saves candidate molecules.
  6) Merges generated fragments with rule mappings (using data/transformation_rules.csv and other rule sources) to build results/new_molecules.csv, results/merged_file.csv, results/merged_file_pred.csv, etc.; selects predicted_label==1 rows for UI display. Images for molecules and rules are drawn to results/molecule_images and results/rules_images.
- Flask app (app.py):
  - Route /run_model accepts SMILES + Similarity Value, writes input/target_m.csv, calls processing.process(similarity_value), then surfaces up to 20 passing SMILES (predicted_label==1) from results/new_molecules.csv.
  - Route /prediction rewrites predict/input/target.csv with SMILES + solvent, runs predict/01..03 to produce predict/result/target_predictions.csv, and renders the table.

## File Responsibilities and Entry Points
- Fluor-RLAT/run.py: sample driver that sets similarity_value=0.1 and calls processing.process for an example SMILES; adjust or import as needed for batch runs.
- Fluor-RLAT/01_data_preprocessing.py: multi-step feature generation; heavy scaffold list inside; ensure RDKit available.
- Fluor-RLAT/02_property_prediction.py: defines GraphFingerprintsModel and fingerprint CNN; loads .pth weights; assumes StandardScaler/MinMaxScaler fit to training data; device auto-selects CUDA if available.
- Fluor-RLAT/03_file_merge.py: merges prediction outputs and removes *.bin caches.
- NIRFluor-opt and web/app.py: Flask UI; uses templates under templates/ and static assets under static/.
- NIRFluor-opt and web/processing.py: entire optimization pipeline described above; controls rule filtering and molecule assembly. Relies on data/transformation_rules*.csv and results/ staging.
- NIRFluor-opt and web/run.py: simple CLI wrapper that seeds input/target_m.csv and calls processing.process with similarity_value=0.1.
- predict/ within NIRFluor-opt and web: copy of Fluor-RLAT pipeline for single-target prediction used by the web route.
- figure_code/*: notebooks (Fluor-opt_figures_code.ipynb, Fluor-pred_figures_code.ipynb) with associated CSV data for plotting figures.

## Data Locations and Persistence
- Inputs: Fluor-RLAT/input/target.csv (property prediction), NIRFluor-opt and web/input/target_m.csv (optimization), predict/input/target.csv (web prediction). Training/reference rule tables under NIRFluor-opt and web/data/* and Fluor-RLAT/data/*.
- Outputs: Fluor-RLAT/result/target_predictions*.csv; NIRFluor-opt and web/results/*.csv plus images; predict/result/target_predictions.csv for UI.
- Scripts aggressively delete previous outputs (processing.py removes prior results CSV/PNG; 03_文件组合.py removes .bin caches). Preserve artifacts elsewhere if needed.

## Training Data

### Fluor-RLAT (Property Prediction)
- **Primary source**: 00_FluoDB.csv (49,832 total records)
  - Columns: abs, em, plqy, k (log-transformed), smiles, solvent, solvent_num, tag, tag_name, Molecular_Weight, LogP, TPSA, Avg_Gasteiger_Charge, Double_Bond_Count, Ring_Count, unimol_plus, split
  - Split into train (21,949), valid (~5,000), test (~5,000) across four properties (abs/em/plqy/k)
- **Features** (total 152 columns per sample in train_abs.csv, etc.):
  1. **Target labels** (1 column): abs (nm), em (nm), plqy (0-1), or k (log₁₀ molar absorptivity)
  2. **Molecular descriptors** (8 columns): Molecular_Weight, LogP, TPSA, Double_Bond_Count, Ring_Count, unimol_plus, tag (scaffold numeric ID), solvent_num (mapped from solvent SMILES)
  3. **Scaffold fingerprints** (136 columns, fragment_1..fragment_136): binary indicators for 136 predefined substructures (from 00_mmp_substructure.csv) including SquaricAid, Naphthalimide, Coumarin, BODIPY, Cyanine, Rhodamine, etc.
  4. **Solvent fingerprints** (2048-bit Morgan, stored in train_sol_abs.csv): radius=2 Morgan fingerprint of solvent SMILES (mapped via 00_solvent_mapping.csv: 75 solvents including ClCCl, CO, CCO, CC#N, etc.)
  5. **Molecule fingerprints** (2048-bit Morgan, stored in train_smiles_abs.csv): radius=2 Morgan fingerprint of target molecule SMILES
- **Preprocessing** (01_data_preprocessing.py):
  - Maps solvent names → solvent_num (0-74) via 00_solvent_mapping.csv
  - Computes RDKit descriptors: MW, logP, TPSA, count_double_bonds (includes aromatics), ring count
  - Assigns scaffold tags from 136-member substructure dictionary (binary match to fragment_1..fragment_136)
  - Generates Morgan fingerprints (radius=2, nBits=2048) for both solvent and molecule SMILES
- **Targets**:
  - abs: absorption wavelength in nm (range ~300-900)
  - em: emission wavelength in nm (range ~350-1000)
  - plqy: photoluminescence quantum yield (0-1 scale)
  - k: log₁₀(molar absorptivity) (range ~3-5.5)

### NIRFluor-opt (NIR Classification)
- **Training data**: Morgan_train_set.csv (13,792 samples)
  - Columns: smiles, abs (nm), label (0=non-NIR, 1=NIR), Bit_1..Bit_2048 (Morgan fingerprint)
  - Binary classification: label=1 indicates near-infrared fluorophore (typically abs >650 nm)
  - Features: 2048-bit Morgan fingerprint (radius=2) derived from SMILES
- **Transformation rules**: transformation_rules.csv (10,355 rules)
  - Columns: element_tran (transformation description), public (example SMILES with markers [*:*]), public_num (example with markers [*:1], [*:2], etc.)
  - Rules sourced from literature MMP (matched molecular pair) transformations
  - MACCS and Morgan fingerprints precomputed for similarity filtering (transformation_rules_maccs.csv, transformation_rules_morgan.csv)
- **No retraining scripts**: Both NIR classifier (stacking_model_full.pkl) and property models (Model_abs.pth, etc.) are **pretrained only**; no training code exists in repository

## Model Architecture

### Fluor-RLAT Property Prediction (GraphFingerprintsModel)
**Multimodal deep learning** combining graph neural networks with fingerprint CNNs:
- **Graph branch** (AttentiveFPGNN + AttentiveFPReadout):
  - Input: molecular graph with atom features (39-dim) and bond features (10-dim) from AttentiveFP featurizers
  - Architecture: num_layers=2 (abs), 3 (em/k), 1 (plqy); graph_feat_size=256; dropout=0.3
  - Output: 256-dim graph-level embedding
- **Fingerprint branches** (parallel processing):
  1. **Solvent extractor** (FC network): solvent Morgan fingerprint (2048-bit) → 256-dim via 2-layer MLP
  2. **Molecule+descriptor extractor** (FingerprintAttentionCNN): concatenated [molecule Morgan (2048-bit) + 8 numeric descriptors + 136 scaffold flags] → 512-dim (2×256) via Conv1d attention mechanism
- **Feature fusion**: concatenate [graph_feats (256) + solvent_out (256) + smiles_extra_out (512)] → 1024-dim
- **Prediction head**: 1024 → 128 (ReLU) → 1 (final property value)
- **Training details** (inferred from code, no training script):
  - Loss: MSE with LDS (Label Distribution Smoothing) reweighting via KernelDensity (bandwidth=5, alpha=0.1 for abs, 0 for others)
  - Optimizer: Adam, lr=1e-3
  - Epochs: 3 (for validation), patience=20 (early stopping)
  - Batch size: 32
  - Normalization: StandardScaler for labels (abs/em/plqy/k), MinMaxScaler for 8 numeric descriptors
- **Four separate models**: Model_abs.pth, Model_em.pth, Model_plqy.pth, Model_k.pth (each ~10-50MB .pth files containing state_dict)

### NIRFluor-opt NIR Classification (Stacking Ensemble)
**Stacking classifier** trained on Morgan fingerprints:
- **Input**: 2048-bit Morgan fingerprint (radius=2) from SMILES
- **Architecture**: StackingClassifier with multiple base learners (GradientBoosting, LogisticRegression, LightGBM, XGBoost) + meta-learner
- **Output**: binary label (0/1) + probability score for NIR class
- **Model file**: stacking_model_full.pkl (pickled with scikit-learn 1.0.2)
- **Critical constraint**: requires scikit-learn==1.0.2 (1.7+ breaks sklearn.ensemble._gb_losses import)
- **No retraining**: model is pretrained; code only loads and runs inference via joblib.load + predict_proba/predict

## Model Differences: Fluor-RLAT vs NIRFluor-opt
| Aspect | Fluor-RLAT (Property Prediction) | NIRFluor-opt (NIR Classification) |
|--------|----------------------------------|-----------------------------------|
| **Task** | Regression (4 properties: abs/em/plqy/k) | Binary classification (NIR yes/no) |
| **Input** | SMILES + solvent (mapped to ID) | SMILES only |
| **Model type** | Deep learning (GNN + CNN fusion) | Classical ML (stacking ensemble) |
| **Features** | Graph + Morgan FP + descriptors + scaffold flags + solvent FP | Morgan FP (2048-bit) only |
| **Training data** | 00_FluoDB.csv (49,832 records) | Morgan_train_set.csv (13,792 records) |
| **Data overlap** | Some shared molecules, different targets | Independent NIR label annotation |
| **Output** | Four continuous values (wavelengths/yield/extinction) | Binary label + probability |
| **Model files** | Model_abs.pth (25MB), Model_em.pth, Model_plqy.pth, Model_k.pth | stacking_model_full.pkl (10MB) |
| **Framework** | PyTorch + DGL + dgllife | scikit-learn + lightgbm + xgboost |
| **Inference** | 02_property_prediction.py (loads .pth, runs forward pass) | processing.py (joblib.load, predict_proba) |
| **Retraining** | No training script (pretrained only) | No training script (pretrained only) |

## Environment Setup Issues
- Python 3.7-3.10 recommended; critical version constraint: scikit-learn==1.0.2 (stacking model pickled with this version; 1.7+ breaks internal imports like sklearn.ensemble._gb_losses)
- Required deps: RDKit, DGL (1.1.2+cu117), dgllife (0.2.8), PyTorch (1.13.1), pandas (1.3.0), lightgbm, xgboost, scikit-learn==1.0.2
- Install order matters: install PyTorch with CUDA first if GPU available, then dgl matching CUDA version, then dgllife
- Common missing deps during first run: lightgbm, xgboost (needed by stacking model in processing.py)
- GPU optional but recommended for 02_性质预测.py; falls back to CPU automatically
- MorganGenerator deprecation warnings from RDKit are harmless (old API still functional)
## Training / Retraining
**No training scripts exist in this repository**—all models are pretrained only:
- **Fluor-RLAT models** (Model_abs.pth, Model_em.pth, Model_plqy.pth, Model_k.pth): PyTorch state_dicts saved after training; no training code provided
  - 02_property_prediction.py contains inference-only pipeline: loads pretrained models via `model.load_state_dict(torch.load('Model_*.pth'))`, runs forward pass, inverse-transforms predictions
  - To retrain: would need to implement training loop with MSE+LDS loss, Adam optimizer (lr=1e-3), early stopping (patience=20), data loaders for train/valid sets
  - Training data available in data/train_*.csv, data/valid_*.csv, data/test_*.csv (split already performed in 00_FluoDB.csv)
- **NIR classifier** (stacking_model_full.pkl): pickled scikit-learn StackingClassifier; no training code provided
  - processing.py loads via `joblib.load('./data/stacking_model_full.pkl')` for inference only
  - To retrain: would need to implement stacking ensemble with base learners (GradientBoosting, LogisticRegression, LightGBM, XGBoost), fit on Morgan_train_set.csv, pickle with scikit-learn==1.0.2
  - **Critical**: must use scikit-learn==1.0.2 for pickle compatibility; retrain would require same version
- **Retraining not required for normal usage**: existing models cover intended use cases (property prediction and NIR classification)
- **If retraining needed**: refer to 02_property_prediction.py model architecture (GraphFingerprintsModel class) for Fluor-RLAT; no reference architecture exists for NIR stacking model beyond inference calls in processing.py

### Adding New Training Data for Fluor-RLAT

**Data format required**: To add new training data, you need at minimum:
- `smiles`: molecular SMILES string
- `solvent`: solvent name (must match one of 75 entries in data/00_solvent_mapping.csv: ClCCl, CO, CCO, CC#N, Cc1ccccc1, CS(C)=O, etc.)
- `abs`: absorption wavelength in nm (optional if only training emission model)
- `em`: emission wavelength in nm (optional if only training absorption model)
- `plqy`: quantum yield 0-1 (optional, leave empty/NaN if not available)
- `k`: molar absorptivity (optional, leave empty/NaN if not available)

**Where to put new data**:
1. **Option 1 - Concatenate to existing dataset** (recommended for large additions >5k samples):
   - Add rows to Fluor-RLAT/data/00_FluoDB.csv
   - Required columns: `abs,em,plqy,k,smiles,solvent,solvent_num,tag,tag_name,Molecular_Weight,LogP,TPSA,Avg_Gasteiger_Charge,Double_Bond_Count,Ring_Count,unimol_plus,split`
   - Fill missing columns by running 01_data_preprocessing.py or compute manually:
     - `solvent_num`: map via 00_solvent_mapping.csv (0-74)
     - `tag`, `tag_name`: scaffold assignment (computed by preprocessing)
     - `Molecular_Weight`, `LogP`, `TPSA`, `Double_Bond_Count`, `Ring_Count`: RDKit descriptors
     - `split`: assign 'train', 'valid', or 'test' (stratify by solvent/scaffold if possible)
   - After adding data, re-run preprocessing to regenerate train/valid/test splits and fingerprint files

2. **Option 2 - Simple format (RECOMMENDED if you only have smiles + solvent + abs/em)**:
   - Create CSV with just 4 columns: `smiles,solvent,abs,em`
   - Leave `plqy` and `k` empty/omitted (models train separately)
   - Place file as Fluor-RLAT/input/target.csv or create your own filename
   - Run 01_data_preprocessing.py to automatically compute:
     - RDKit descriptors (MW, LogP, TPSA, double bonds, rings)
     - Scaffold tags (binary match against 136 substructures)
     - Morgan fingerprints (2048-bit for both molecule and solvent)
     - All 152 feature columns
   - Manually split output into train/valid/test CSVs (e.g., 70/15/15 split)
   - Train Model_abs.pth and Model_em.pth (skip plqy and k models)

3. **Option 3 - Separate training dataset** (for advanced users):
   - Create new CSV with columns: `smiles,solvent,abs,em,plqy,k`
   - Run through 01_data_preprocessing.py to generate descriptors and fingerprints
   - Manually split into train/valid/test CSVs
   - Either train standalone model or fine-tune pretrained weights

**Training script requirements** (not included in repo—must implement):
- Data loaders using MolecularDataset class and collate_fn from 02_property_prediction.py
- Training loop with:
  - Forward pass: `predictions = model(graphs, node_feats, edge_feats, fps)`
  - Loss: MSE with optional LDS reweighting `(weights * (predictions - labels)**2).mean()`
  - Optimizer: Adam(lr=1e-3)
  - Early stopping: patience=20 epochs, save best validation loss
  - Checkpoint: `torch.save(model.state_dict(), 'Model_abs.pth')` when validation improves
- Normalization: fit StandardScaler on training labels, MinMaxScaler on 8 numeric descriptors (columns 8-16 in feature matrix)
- Batch size: 32, epochs: typically 50-100 with early stopping

**Solvent constraints**:
- New solvents not in 00_solvent_mapping.csv require:
  - Adding solvent SMILES and assigning new solvent_num
  - Generating 2048-bit Morgan fingerprint for the solvent
  - Retraining on sufficient data covering new solvent (models won't extrapolate well to unseen solvents)

**Data volume recommendations**:
- Full retrain: >10k samples for good generalization
- Fine-tuning pretrained models: >1k samples, use low learning rate (1e-4 to 1e-5)
- Transfer learning: <1k samples, freeze graph branch, only train fingerprint extractors

## Operational Notes
- processing.py assumes rule CSV schemas: columns include element_tran, node1, node2, smiles fingerprints; similarity threshold drives rule selection
- SMILES/solvent validation is minimal; invalid SMILES will raise RuntimeError during fragmentation or model loading
- All Chinese filenames and content have been translated to English (Feb 2026); use ASCII names for new files
- Scripts modified to work from their own directories (pathlib anchoring); all ./predict/ paths converted to ./ in predict/ folder scripts

## Editing Guidelines
- Keep paths relative to their module roots (do not hardcode absolute paths).
- Avoid altering rule CSV schemas or model filenames unless you also update processing.py and prediction scripts accordingly.
- When extending web UI, preserve Flask route behaviors and expected CSV outputs for templates.
- For batch jobs, consider bypassing Flask and calling processing.process or Fluor-RLAT scripts directly; capture outputs before cleanup steps delete intermediates.
- Add brief comments only where logic is non-obvious (e.g., feature splits, rule expansion), stay concise.

## Repository Change Log
**Important**: Document significant changes here so future agents/sessions understand what was modified.

### February 2026 - Initial Setup & Cleanup
- Fixed path resolution issues: anchored all run.py scripts to their own directories using pathlib to avoid cwd errors when launched from different locations or debuggers
- Padded fragment processing in processing.py to handle molecules with fewer than 4 parts (previously crashed on column count mismatch)
- Fixed predict/ folder scripts: converted all `./predict/` paths to `./` so they run correctly when cwd is the predict directory
- Environment version constraints documented: scikit-learn==1.0.2 required for stacking model compatibility
- Comprehensive translation pass: renamed all Chinese filenames and translated all Chinese content to English
  - Files renamed: `打包.ipynb` → `packaging.ipynb`, `基团替换.csv` → `group_replacement.csv`, `H替换.csv` → `H_replacement.csv`, etc.
  - Script names: `01_数据预处理.py` → `01_data_preprocessing.py`, `02_性质预测.py` → `02_property_prediction.py`, `03_文件组合.py` → `03_file_merge.py`
  - All print statements, comments, and error messages translated
- Created translate_all.py for reproducible translation if needed again
- Add brief comments only where logic is non-obvious (e.g., feature splits, rule expansion), stay concise.
