# U-Net for Document Tampering Detection

A PyTorch implementation of U-Net for detecting tampered regions in documents. This model predicts binary masks that highlight areas of interest (tampered regions) in RGB images.

## Features

### Advanced Loss Function
The model uses a **Combined Loss** that addresses class imbalance (small masks vs. large backgrounds):
- **Binary Cross-Entropy (BCE)**: Standard pixel-wise loss
- **Dice Loss**: Focuses on overlap between prediction and ground truth, robust to class imbalance
- **Focal Loss**: Emphasizes hard examples, reduces weight of easy background pixels

### Model Architecture
- **U-Net** with encoder-decoder structure
- Skip connections for preserving spatial information
- Batch normalization for stable training
- ~31M trainable parameters

### Training Best Practices
- ✅ **Learning Rate Scheduling**: ReduceLROnPlateau (reduces LR when validation loss plateaus)
- ✅ **Early Stopping**: Stops training if no improvement after 15 epochs
- ✅ **Model Checkpointing**: Saves best models based on IoU and loss
- ✅ **Data Normalization**: ImageNet statistics
- ✅ **AdamW Optimizer**: With weight decay for regularization
- ✅ **Comprehensive Metrics**: IoU, Dice coefficient, Precision, Recall, F1

## Dataset Structure

The dataset is stored in LMDB format with the following structure:
- **Images**: Stored with keys `image-%09d` (e.g., `image-000000001`)
- **Masks**: Stored with keys `label-%09d` (e.g., `label-000000001`)
- **Image Format**: RGB, 512×512 pixels
- **Mask Format**: Single channel, binary (0=background, 255=tampered region)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Train with default settings
python train_unet.py
```

**Training Configuration** (modify in `train_unet.py`):
- `batch_size`: 8 (adjust based on GPU memory)
- `num_epochs`: 100
- `learning_rate`: 1e-3
- `train_path`: `DocTamperV1-TrainingSet`
- `val_path`: `DocTamperV1-TestingSet`
- `early_stopping_patience`: 15 epochs

**Training Output**:
- Model checkpoints saved in `checkpoints/`
  - `best_model_iou.pth`: Best model based on IoU
  - `best_model_loss.pth`: Best model based on validation loss
  - `checkpoint_epoch_X.pth`: Periodic checkpoints every 10 epochs
  - `final_model.pth`: Final model state
- Sample predictions saved in `predictions/` (every 5 epochs)
- Training history plot: `training_history.png`

### Inference

```bash
# Test on a single image
python inference.py --checkpoint checkpoints/best_model_iou.pth --index 1

# Test on multiple samples (default: 10 samples)
python inference.py --checkpoint checkpoints/best_model_iou.pth --num_samples 20

# Test with custom threshold
python inference.py --checkpoint checkpoints/best_model_iou.pth --threshold 0.6

# Test on different dataset
python inference.py --checkpoint checkpoints/best_model_iou.pth \
                    --lmdb_path DocTamperV1-TrainingSet \
                    --num_samples 5
```

**Inference Arguments**:
- `--checkpoint`: Path to model checkpoint (default: `checkpoints/best_model_iou.pth`)
- `--lmdb_path`: Path to LMDB dataset (default: `DocTamperV1-TestingSet`)
- `--index`: Index of specific image to test (optional)
- `--num_samples`: Number of samples to test (default: 10)
- `--threshold`: Threshold for binary prediction (default: 0.5)
- `--output_dir`: Directory to save results (default: `test_results/`)
- `--device`: Device to use (default: auto-detect CUDA)

**Inference Output**:
- Visualization images saved in `test_results/`
- Each visualization shows:
  - Original image
  - Ground truth mask
  - Predicted probability mask (colored heatmap)
  - Binary prediction with metrics
- Console output with detailed metrics

## Model Performance Metrics

The training script tracks the following metrics:

- **IoU (Intersection over Union)**: Measures overlap between prediction and ground truth
- **Dice Coefficient**: Similar to IoU but with different weighting (2 × intersection / sum)
- **Precision**: True positive rate among predicted positives
- **Recall**: True positive rate among actual positives
- **F1 Score**: Harmonic mean of precision and recall

## Loss Function Design

### Why Combined Loss?

Document tampering detection presents unique challenges:

1. **Class Imbalance**: Tampered regions are often much smaller than the background
2. **Variable Mask Sizes**: Some images have tiny tampering, others have large regions
3. **Multiple Masks**: Some images contain multiple disconnected tampered regions

### Solution: Combined Loss

```python
Total Loss = 0.5 × BCE + 0.3 × Dice + 0.2 × Focal
```

- **BCE (50%)**: Provides stable gradients for all pixels
- **Dice (30%)**: Focuses on overlap, handles size imbalance
- **Focal (20%)**: Emphasizes hard-to-classify pixels, ignores easy background

This combination ensures:
- ✅ Robust learning even with small masks
- ✅ Better detection of multiple disconnected regions
- ✅ Reduced false positives on background
- ✅ Improved performance on various mask sizes

## Training Tips

### GPU Memory Issues
If you encounter out-of-memory errors:
```python
# In train_unet.py, reduce batch size
config['batch_size'] = 4  # or even 2
```

### Faster Convergence
```python
# Increase learning rate slightly
config['learning_rate'] = 2e-3

# Use Cosine Annealing instead of ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
```

### Fine-tuning
To resume training from a checkpoint:
```python
# In train_unet.py, after model initialization
checkpoint = torch.load('checkpoints/best_model_iou.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

## File Structure

```
.
├── train_unet.py          # Main training script
├── inference.py           # Inference and evaluation script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── vizlmdb.py            # Example LMDB visualization script
├── checkpoints/          # Model checkpoints (created during training)
├── predictions/          # Sample predictions (created during training)
├── test_results/         # Inference results (created during inference)
├── training_history.png  # Training metrics plot
├── DocTamperV1-TrainingSet/    # Training dataset (LMDB)
│   ├── data.mdb
│   └── lock.mdb
└── DocTamperV1-TestingSet/     # Testing dataset (LMDB)
    ├── data.mdb
    └── lock.mdb
```

## Architecture Details

### U-Net Structure
```
Input (3×512×512)
    ↓
Encoder:
    DoubleConv(3→64)
    Down(64→128)
    Down(128→256)
    Down(256→512)
    Down(512→512)
    ↓
Decoder:
    Up(1024→512) + skip
    Up(512→256) + skip
    Up(256→128) + skip
    Up(128→64) + skip
    ↓
Output (1×512×512)
```

### DoubleConv Block
```
Conv2d(3×3) → BatchNorm → ReLU → Conv2d(3×3) → BatchNorm → ReLU
```

## Expected Results

With proper training, you should achieve:
- **Training IoU**: 0.85-0.95
- **Validation IoU**: 0.75-0.88
- **Dice Coefficient**: 0.80-0.92

Training typically converges in 30-50 epochs with early stopping.

## Troubleshooting

### Issue: Model not learning
- Check if masks are properly loaded (should be 0 or 1, not 0 or 255)
- Verify data normalization
- Try reducing learning rate

### Issue: Overfitting
- Reduce model complexity (use `bilinear=True` in UNet)
- Increase weight decay
- Add data augmentation

### Issue: Underfitting
- Increase model capacity
- Train for more epochs
- Reduce weight decay
- Increase learning rate

## Citation

If you use this code, please cite the U-Net paper:
```
@article{ronneberger2015unet,
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  journal={MICCAI},
  year={2015}
}
```

## License

This project is for educational and research purposes.

