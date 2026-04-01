# How to Use Fluor Unified

## What Is This Tool?

Fluor Unified is a Google Colab notebook that predicts four optical properties of fluorescent molecules:

- **abs** -- absorption wavelength (nm)
- **em** -- emission wavelength (nm)
- **plqy** -- photoluminescence quantum yield (0 to 1)
- **k** -- log10 of the molar absorptivity (log epsilon)

It uses AttentiveFP graph neural network models trained on the FluoDB dataset. You can use the bundled pretrained models or retrain them on your own experimental data.

---

## Three Modes

Open `Fluor_Unified.ipynb` in Google Colab and run the setup cells first. Then choose a mode from the radio button widget.

### 1. Create Training Data

Use this mode to convert your experimental CSV file into the training format and optionally merge it with the existing dataset.

**Steps:**
1. Upload your CSV to Google Drive (e.g. `fluor-tools/datasets/my_data.csv`).
2. Enter the path in the CSV path field.
3. Run the processing cell. Valid rows are featurized; invalid rows are reported with reasons.
4. Run the merge cell to combine your data with the existing training data and save to Drive.

### 2. Predict Properties

Use this mode to predict abs, em, plqy, and k for one or more molecules.

**Steps:**
1. Select model source: **pretrained** (bundled models) or **custom** (your trained models).
2. Enter the solvent name (e.g. `EtOH`, `DMSO`, `Wasser`).
3. Enter one SMILES string per line in the SMILES text area.
4. Run the prediction cell. Results appear as a table.

If you select **custom** and no completed training runs exist, the notebook will tell you and suggest using pretrained models instead.

### 3. Train Models

Use this mode to retrain the models on updated data.

**Steps:**
1. Select which properties to train (any combination of abs, em, plqy, k).
2. Select data source: **original repo data** or **merged Google Drive data**.
3. Run the training cell. Progress is logged per epoch.
4. When training finishes, models are archived to `training-runs/completed/`.

If your Colab session disconnects mid-training, just re-run the training cell. The session will resume from the last checkpoint automatically (as long as the config matches).

---

## Input CSV Format

Your CSV must have these columns:

| Column  | Required | Description                                      |
|---------|----------|--------------------------------------------------|
| name    | yes      | Molecule identifier (any string)                 |
| solvent | yes      | Solvent name (German or English, see list below) |
| smiles  | yes      | SMILES string of the molecule                    |
| abs     | no       | Absorption wavelength in nm                      |
| em      | no       | Emission wavelength in nm                        |
| epsilon | no       | Molar absorptivity (will be converted to k)      |
| mw      | no       | Molecular weight (informational only)            |
| plqy    | no       | Quantum yield (0 to 1)                           |

Rows with missing or empty SMILES are skipped. Rows with unrecognized solvent names are skipped. All other columns are optional -- rows with NaN property values are simply excluded from that property's training split.

---

## Supported Solvent Names

The following names are recognized (German and English):

| German name   | English name       | SMILES       |
|---------------|--------------------|--------------|
| Toluol        | Toluene            | Cc1ccccc1    |
| EtOH          | Ethanol            | CCO          |
| MeOH          | Methanol           | CO           |
| DCM           | Dichloromethane    | ClCCl        |
| Dichlormethan | Dichloromethane    | ClCCl        |
| CHCl3         | Chloroform         | ClC(Cl)Cl    |
| Benzol        | Benzene            | c1ccccc1     |
| Wasser        | Water              | O            |
| Aceton        | Acetone            | CC(C)=O      |
| Cyclohexan    | Cyclohexane        | C1CCCCC1     |
| Hexan         | Hexane             | CCCCCC       |
| Acetonitril   | Acetonitrile       | CC#N         |
| Diethylether  | Diethyl ether      | CCOCC        |
| THF           | Tetrahydrofuran    | C1CCOC1      |
| DMSO          | --                 | CS(C)=O      |
| DMF           | --                 | CN(C)C=O     |

You can also pass a SMILES string directly as the solvent value if it is not in this list and exists in the `00_solvent_mapping.csv` reference file.

---

## Retraining Models

1. Prepare your data using **Create Training Data** mode and save to Drive.
2. Switch to **Train Models** mode.
3. Select the properties you want to retrain.
4. Choose **merged Google Drive data** as the data source.
5. Run the training cell.

**Session resumption after Colab disconnect:**
- Checkpoints are saved after every epoch to `training-runs/active/`.
- When you re-run the training cell with the same settings, training resumes from the last checkpoint.
- If you change any training parameter (epochs, patience, learning rate, selected properties), the notebook detects the mismatch and starts fresh.

**After training completes:**
- Final models are moved to a timestamped folder in `training-runs/completed/`.
- The `active/` folder is cleaned up.

---

## Making Predictions

**Pretrained models:**
- Select **pretrained** in the model source widget.
- Models are loaded from the cloned repository (`Fluor-RLAT/Model_{property}.pth`).

**Custom models:**
- Select **custom** in the model source widget.
- The notebook automatically loads the most recently completed training run.
- You can also manually specify a run folder from `training-runs/completed/`.

**Input options:**
- Type SMILES directly into the text area (one per line).
- For batch predictions from a CSV, load the file in a code cell and iterate over rows.

---

## Google Drive Folder Structure

All persistent data lives under `/content/drive/MyDrive/fluor-tools/`:

```
fluor-tools/
  datasets/                        # Your CSV files and merged training data
    train_{property}.csv           # Merged main training data
    train_smiles_{property}.csv    # Merged molecule fingerprints
    train_sol_{property}.csv       # Merged solvent fingerprints
  training-runs/
    active/                        # Current training session (checkpoints)
      training_config.json         # Run parameters for resume detection
      checkpoint_{property}.pth    # Per-target checkpoint (overwritten each epoch)
      Model_{property}.pth         # Best model so far (overwritten when improved)
    completed/
      YYYY-MM-DD_HH-MM-SS/         # Archived finished run
        Model_{property}.pth       # Final best models
        training_config.json       # Archived config
```

The `active/` folder is temporary. Once training completes, everything is moved to `completed/` and `active/` is cleaned up. If you want to keep a run, do not delete the `completed/` subfolder.

---

## DGL Compatibility Note

DGL 2.4 supports PyTorch up to version 2.4. If you encounter a `Cannot find DGL C++ graphbolt library` error, it means your PyTorch version is newer than what DGL supports. The setup cell in the notebook installs the correct DGL wheel for your PyTorch version automatically. If the issue persists, pin PyTorch to 2.4.x before installing DGL.

Reference: https://github.com/dmlc/dgl/issues/7822
