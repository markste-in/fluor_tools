
import os
from pathlib import Path
import pandas as pd
import subprocess

# Always resolve paths from this script to avoid cwd issues
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
INPUT_DIR = BASE_DIR / "input"
INPUT_DIR.mkdir(exist_ok=True)
csv_file = INPUT_DIR / 'target.csv'

# Manual input values (right column names)
new_smiles = 'C2=C1C7=C(C(=[N+]1[B-]([N]3C2=C5C(=C3C4=CC=CC=C4)C=CC=C5)(F)F)C6=CC=CC=C6)C=CC=C7'  # Example SMILES
new_solvent_name = 'Toluene'  # Example solvent name (right column)

# Solvent name to structure mapping (right column -> left column)
solvent_mapping = {
    'CH2Cl2': 'ClCCl',
    'MeOH': 'CO',
    'EtOH': 'CCO',
    'CHCl3': 'ClC(Cl)Cl',
    'MeCN': 'CC#N',
    'THF': 'C1CCOC1',
    'Toluene': 'Cc1ccccc1',
    'DMSO': 'CS(C)=O',
    'H2O': 'O',
    'Benzene': 'c1ccccc1'
}

# Convert input solvent name (right column) to structure (left column)
if new_solvent_name not in solvent_mapping:
    raise ValueError(f"❌ Input solvent name '{new_solvent_name}' not found in mapping table")
new_solvent = solvent_mapping[new_solvent_name]

# Read original CSV file
df = pd.read_csv(csv_file)

# Replace smiles and solvent in first row
if 'smiles' in df.columns and 'solvent' in df.columns:
    df.at[0, 'smiles'] = new_smiles
    df.at[0, 'solvent'] = new_solvent
else:
    raise ValueError("❌ CSV file does not contain 'smiles' or 'solvent' columns")

# Save updated CSV file
df.to_csv(csv_file, index=False)

print(f"✅ First row 'smiles' and 'solvent' replaced with: {new_smiles}, {new_solvent}, and saved to: {csv_file}")





print("🚀 Running 01_data_preprocessing.py...")
subprocess.run(['python', '01_data_preprocessing.py'], check=True, cwd=BASE_DIR)

print("🚀 Running 02_property_prediction.py...")
subprocess.run(['python', '02_property_prediction.py'], check=True, cwd=BASE_DIR)

print("🚀 Running 03_file_merge.py...")
subprocess.run(['python', '03_file_merge.py'], check=True, cwd=BASE_DIR)