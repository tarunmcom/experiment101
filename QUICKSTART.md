# Quick Start Guide

Get started with U-Net training in 3 simple steps!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note for Windows users**: If you encounter issues with `lmdb`, try:
```bash
pip install lmdb-python
```

## Step 2: Verify Dataset

Check that your datasets are properly formatted and loaded:

```bash
python check_dataset.py
```

This will:
- ✓ Verify dataset integrity
- ✓ Show statistics about mask sizes and coverage
- ✓ Generate visualization plots
- ✓ Provide recommendations for training

**Expected Output:**
```
Dataset initialized with XXX samples from DocTamperV1-TrainingSet
Dataset initialized with XXX samples from DocTamperV1-TestingSet
```

## Step 3: Train the Model

```bash
python train_unet.py
```

**Training will:**
- Use combined loss (BCE + Dice + Focal) optimized for imbalanced masks
- Save best models in `checkpoints/`
- Generate sample predictions every 5 epochs
- Plot training curves at the end
- Use learning rate scheduling and early stopping

**Expected Training Time:**
- With GPU (NVIDIA RTX 3080): ~1-2 hours for 50 epochs
- With CPU: ~10-20 hours for 50 epochs

**Monitor Training:**
The progress bar shows:
- `loss`: Combined training loss
- `iou`: Intersection over Union metric
- `dice`: Dice coefficient

Example:
```
Epoch 10 [Train]: 100%|████| loss: 0.1234, iou: 0.8567, dice: 0.8901
```

## Step 4: Evaluate the Model (Optional)

Get comprehensive evaluation metrics:

```bash
python evaluate.py --checkpoint checkpoints/best_model_iou.pth
```

This generates:
- Per-image and global metrics
- ROC and Precision-Recall curves
- Confusion matrix
- Metric distribution plots

## Step 5: Run Inference

Test on specific images or multiple samples:

```bash
# Test on first image
python inference.py --index 1

# Test on 10 random samples
python inference.py --num_samples 10

# Test with custom threshold
python inference.py --threshold 0.6
```

## Common Issues & Solutions

### Issue 1: Out of Memory Error

**Solution:** Reduce batch size in `train_unet.py`:

```python
config['batch_size'] = 4  # or even 2
```

### Issue 2: CUDA Not Available

The script will automatically use CPU if CUDA is not available. To force CPU:

```python
config['device'] = 'cpu'
```

### Issue 3: Dataset Not Found

Ensure your folder structure is:
```
.
├── DocTamperV1-TrainingSet/
│   ├── data.mdb
│   └── lock.mdb
└── DocTamperV1-TestingSet/
    ├── data.mdb
    └── lock.mdb
```

### Issue 4: Training Not Improving

Try these solutions:
1. **Check data loading**: Run `check_dataset.py` to verify masks
2. **Adjust learning rate**: Try `1e-4` instead of `1e-3`
3. **Increase training time**: Remove early stopping or increase patience
4. **Check loss weights**: Modify weights in `CombinedLoss`

## Configuration Options

### Training Configuration (in `train_unet.py`)

```python
config = {
    'batch_size': 8,           # Adjust based on GPU memory
    'num_epochs': 100,         # Maximum epochs
    'learning_rate': 1e-3,     # Initial learning rate
    'weight_decay': 1e-5,      # L2 regularization
    'early_stopping_patience': 15,  # Epochs to wait
}
```

### Loss Function Weights

```python
criterion = CombinedLoss(
    bce_weight=0.5,    # Binary Cross-Entropy
    dice_weight=0.3,   # Dice Loss
    focal_weight=0.2   # Focal Loss
)
```

**Recommendations:**
- **Small masks (<5% coverage)**: Increase `dice_weight` and `focal_weight`
- **Large masks (>30% coverage)**: Increase `bce_weight`
- **Balanced masks**: Keep default weights

## Expected Results

After successful training:

| Metric | Expected Range |
|--------|---------------|
| Training IoU | 0.85 - 0.95 |
| Validation IoU | 0.75 - 0.88 |
| Dice Coefficient | 0.80 - 0.92 |
| Precision | 0.80 - 0.95 |
| Recall | 0.75 - 0.90 |

**Note:** Results depend on dataset quality and training time.

## Output Files

After training, you'll have:

```
.
├── checkpoints/
│   ├── best_model_iou.pth      # Best model by IoU
│   ├── best_model_loss.pth     # Best model by loss
│   ├── final_model.pth         # Final epoch model
│   └── checkpoint_epoch_X.pth  # Periodic checkpoints
├── predictions/
│   ├── epoch_5.png
│   ├── epoch_10.png
│   └── ...
├── training_history.png        # Loss and metrics plots
├── dataset_statistics.png      # From check_dataset.py
├── training_samples.png        # Sample visualizations
└── validation_samples.png      # Sample visualizations
```

## Tips for Better Results

### 1. Data Quality
- Ensure masks are properly aligned with images
- Check for corrupted or missing masks
- Verify mask values (0 and 255)

### 2. Training Strategy
- Start with default settings
- Monitor validation metrics closely
- Save best model based on IoU (not loss)
- Use early stopping to prevent overfitting

### 3. Hyperparameter Tuning
- Learning rate: Try `[1e-4, 5e-4, 1e-3, 2e-3]`
- Batch size: Larger is better (if GPU allows)
- Loss weights: Adjust based on mask size distribution

### 4. Model Selection
- Use `best_model_iou.pth` for final predictions
- Compare with `best_model_loss.pth` for different metrics
- Check epoch predictions to understand model behavior

## Next Steps

1. ✅ Train baseline model with default settings
2. ✅ Evaluate on test set using `evaluate.py`
3. ✅ Analyze failure cases (low IoU samples)
4. ✅ Fine-tune hyperparameters based on results
5. ✅ Consider data augmentation for better generalization

## Need Help?

Check these files for detailed information:
- `README.md`: Comprehensive documentation
- `check_dataset.py`: Dataset verification
- `train_unet.py`: Training implementation
- `inference.py`: Testing and visualization
- `evaluate.py`: Comprehensive evaluation

## Command Cheat Sheet

```bash
# 1. Verify dataset
python check_dataset.py

# 2. Train model
python train_unet.py

# 3. Evaluate model
python evaluate.py

# 4. Test on samples
python inference.py --num_samples 10

# 5. Test specific image
python inference.py --index 5

# 6. Visualize original example
python vizlmdb.py --input DocTamperV1-TrainingSet --i 1
```

Happy training! 🚀

