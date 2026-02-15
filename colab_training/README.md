# Fluor-RLAT Colab Training

This folder contains a Google Colab notebook for training Fluor-RLAT property prediction models with GPU acceleration.

## Contents

- `Fluor_RLAT_Training.ipynb` - Colab notebook with complete training pipeline

## Quick Start

1. Upload `Fluor_RLAT_Training.ipynb` to Google Colab
2. Enable GPU runtime: Runtime → Change runtime type → T4 GPU
3. Run all cells - the notebook will:
   - Clone the fluor_tools repository automatically
   - Install required dependencies (DGL, dgllife, tqdm)
   - Train all 4 property models with GPU acceleration
4. Download trained models or save checkpoints to Google Drive

## Training Configuration

Edit these variables in cell 4.1 to customize training:

```python
TARGETS = ['abs', 'em', 'plqy', 'k']  # Which models to train
EPOCHS = 200                           # Max training epochs
PATIENCE = 20                          # Early stopping patience
```

## Expected Training Times (Colab T4 GPU)

| Model | ~Samples | ~Time |
|-------|----------|-------|
| abs   | 22k      | 20-30 min |
| em    | 17k      | 15-25 min |
| plqy  | 13k      | 10-20 min |
| k     | 7k       | 5-15 min |
| **Total** | - | **~1-1.5 hours** |

## Output

Trained models are saved to `./models/`:

- `Model_abs.pth` - Absorption wavelength
- `Model_em.pth` - Emission wavelength  
- `Model_plqy.pth` - Quantum yield
- `Model_k.pth` - Molar absorptivity

## Colab Environment Notes

The notebook automatically handles:

- DGL installation with correct CUDA version
- GPU memory management between models
- Progress bars optimized for Colab display

**Tested with:**

- Colab T4 GPU runtime
- PyTorch 2.x + CUDA 12.x
- DGL 2.x

## Troubleshooting

### "CUDA out of memory"

- Restart runtime and try training one model at a time
- Reduce `BATCH_SIZE` from 32 to 16

### DGL installation fails

- Try: `!pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html`
- Check PyTorch CUDA version matches DGL wheel

### Slow graph conversion

- This is normal (~5 min for 22k molecules on first run)
- Subsequent runs use cached graphs
