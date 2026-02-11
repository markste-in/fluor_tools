import os
from pathlib import Path
import pandas as pd
from processing import process

# Always resolve paths from this file to avoid cwd surprises (e.g., debug adapters).
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
INPUT_DIR = BASE_DIR / "input"
RESULTS_DIR = BASE_DIR / "results"
INPUT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

smiles = "C2=C1C7=C(C(=[N+]1[B-]([N]3C2=C5C(=C3C4=CC=CC=C4)C=CC=C5)(F)F)C6=CC=CC=C6)C=CC=C7"

df = pd.DataFrame({"smiles": [smiles]})
df.to_csv(INPUT_DIR / "target_m.csv", index=False)

def main():
    # Set hyperparameters
    similarity_value = 0.1
    process(similarity_value)
    
    print("\nAll molecules processed!")

if __name__ == "__main__":
    main()
