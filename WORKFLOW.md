# Training Workflow Guide

A visual guide to the complete training and evaluation workflow.

## Complete Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 1: Setup & Verification                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Install Deps    │ → │  Test Setup      │ → │  Check Dataset   │
│                  │    │                  │    │                  │
│ pip install -r   │    │ python           │    │ python           │
│ requirements.txt │    │ test_setup.py    │    │ check_dataset.py │
└──────────────────┘    └──────────────────┘    └──────────────────┘
                              ↓
                     Outputs:                     Outputs:
                     - Dependency check           - Dataset stats
                     - CUDA status                - Sample visualizations
                     - Model test                 - Coverage analysis
                                                  - Recommendations
                              ↓
                              
┌─────────────────────────────────────────────────────────────────┐
│                      STEP 2: Model Training                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│                    python train_unet.py                           │
│                                                                   │
│  Training Loop (for each epoch):                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ 1. Load batch from training set                            │ │
│  │    - Images: RGB 512×512                                   │ │
│  │    - Masks: Binary 512×512                                 │ │
│  │                                                             │ │
│  │ 2. Forward pass through U-Net                              │ │
│  │    Input (3,512,512) → Encoder → Bottleneck →              │ │
│  │    Decoder → Output (1,512,512)                            │ │
│  │                                                             │ │
│  │ 3. Calculate Combined Loss                                 │ │
│  │    Loss = 0.5×BCE + 0.3×Dice + 0.2×Focal                  │ │
│  │                                                             │ │
│  │ 4. Backward pass & update weights                          │ │
│  │    optimizer.zero_grad()                                   │ │
│  │    loss.backward()                                         │ │
│  │    optimizer.step()                                        │ │
│  │                                                             │ │
│  │ 5. Calculate metrics (IoU, Dice, Precision, Recall)       │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              ↓                                    │
│  Validation Loop:                                                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ 1. Load batch from validation set                          │ │
│  │ 2. Forward pass (no gradient)                              │ │
│  │ 3. Calculate loss and metrics                              │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              ↓                                    │
│  Post-Epoch Actions:                                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ - Update learning rate (ReduceLROnPlateau)                 │ │
│  │ - Save best model (if IoU improved)                        │ │
│  │ - Save checkpoint (every 10 epochs)                        │ │
│  │ - Generate predictions (every 5 epochs)                    │ │
│  │ - Check early stopping (patience=15)                       │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
                              ↓
                     Outputs:
                     ┌────────────────────────┐
                     │ checkpoints/           │
                     │  ├─ best_model_iou.pth │
                     │  ├─ best_model_loss.pth│
                     │  ├─ checkpoint_*.pth   │
                     │  └─ final_model.pth    │
                     ├────────────────────────┤
                     │ predictions/           │
                     │  ├─ epoch_5.png        │
                     │  ├─ epoch_10.png       │
                     │  └─ ...                │
                     ├────────────────────────┤
                     │ training_history.png   │
                     └────────────────────────┘
                              ↓

┌─────────────────────────────────────────────────────────────────┐
│                    STEP 3: Model Evaluation                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│              python evaluate.py --checkpoint ...                  │
│                                                                   │
│  Evaluation Process:                                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ For each image in test set:                                │ │
│  │   1. Load image and ground truth mask                      │ │
│  │   2. Run inference (forward pass)                          │ │
│  │   3. Calculate metrics:                                    │ │
│  │      - IoU (Intersection over Union)                       │ │
│  │      - Dice Coefficient                                    │ │
│  │      - Precision, Recall, F1                               │ │
│  │      - Accuracy, Specificity                               │ │
│  │   4. Store predictions and targets                         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              ↓                                    │
│  Analysis:                                                       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ - Calculate per-image statistics (mean, std, min, max)    │ │
│  │ - Calculate global metrics (pixel-level aggregation)      │ │
│  │ - Generate ROC curve and AUC                               │ │
│  │ - Generate Precision-Recall curve                          │ │
│  │ - Generate confusion matrix                                │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
                              ↓
                     Outputs:
                     ┌──────────────────────────────┐
                     │ evaluation_results/          │
                     │  ├─ metric_distributions.png │
                     │  ├─ roc_pr_curves.png        │
                     │  ├─ confusion_matrix.png     │
                     │  └─ evaluation_results.txt   │
                     └──────────────────────────────┘
                              ↓

┌─────────────────────────────────────────────────────────────────┐
│                    STEP 4: Inference & Testing                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│           python inference.py --checkpoint ... --index N          │
│                                                                   │
│  Inference Process:                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ 1. Load trained model                                      │ │
│  │ 2. Load test image                                         │ │
│  │ 3. Preprocess:                                             │ │
│  │    - Convert to tensor                                     │ │
│  │    - Normalize with ImageNet stats                        │ │
│  │ 4. Run inference:                                          │ │
│  │    - Forward pass through model                            │ │
│  │    - Apply sigmoid activation                              │ │
│  │    - Threshold to get binary mask                          │ │
│  │ 5. Post-process:                                           │ │
│  │    - Calculate metrics vs ground truth                     │ │
│  │    - Generate visualizations                               │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
                              ↓
                     Outputs:
                     ┌────────────────────┐
                     │ test_results/      │
                     │  ├─ sample_1.png   │
                     │  ├─ sample_2.png   │
                     │  └─ ...            │
                     └────────────────────┘
                              ↓
                      ✅ COMPLETE
```

## Data Flow Through U-Net

```
                         Input Image (RGB)
                       ┌─────────────────┐
                       │   512 × 512 × 3 │
                       └────────┬────────┘
                                │
                         Normalization
                                │
                                ↓
                    ┌──────────────────────┐
                    │   ENCODER PATH       │
                    │                      │
                    │  ┌───────────────┐  │
                    │  │ Conv Block 1  │  │──┐
                    │  │    64 ch      │  │  │ Skip
                    │  └───────┬───────┘  │  │ Connection
                    │          ↓          │  │
                    │  ┌───────────────┐  │  │
                    │  │   MaxPool     │  │  │
                    │  └───────┬───────┘  │  │
                    │          ↓          │  │
                    │  ┌───────────────┐  │  │
                    │  │ Conv Block 2  │  │──┤ Skip
                    │  │   128 ch      │  │  │ Connection
                    │  └───────┬───────┘  │  │
                    │          ↓          │  │
                    │  ┌───────────────┐  │  │
                    │  │   MaxPool     │  │  │
                    │  └───────┬───────┘  │  │
                    │          ↓          │  │
                    │  ┌───────────────┐  │  │
                    │  │ Conv Block 3  │  │──┤ Skip
                    │  │   256 ch      │  │  │ Connection
                    │  └───────┬───────┘  │  │
                    │          ↓          │  │
                    │  ┌───────────────┐  │  │
                    │  │   MaxPool     │  │  │
                    │  └───────┬───────┘  │  │
                    │          ↓          │  │
                    │  ┌───────────────┐  │  │
                    │  │ Conv Block 4  │  │──┤ Skip
                    │  │   512 ch      │  │  │ Connection
                    │  └───────┬───────┘  │  │
                    └──────────┼──────────┘  │
                               ↓             │
                    ┌──────────────────────┐ │
                    │   BOTTLENECK         │ │
                    │  ┌───────────────┐   │ │
                    │  │ Conv Block 5  │   │ │
                    │  │   512 ch      │   │ │
                    │  └───────┬───────┘   │ │
                    └──────────┼───────────┘ │
                               ↓             │
                    ┌──────────────────────┐ │
                    │   DECODER PATH       │ │
                    │                      │ │
                    │  ┌───────────────┐  │ │
                    │  │   Upsample    │  │ │
                    │  └───────┬───────┘  │ │
                    │          ↓          │ │
                    │  ┌───────────────┐  │ │
                    │  │  Concatenate  │◄─┘─┘
                    │  │  with Skip    │
                    │  └───────┬───────┘
                    │          ↓
                    │  ┌───────────────┐
                    │  │ Conv Block    │
                    │  │   256 ch      │
                    │  └───────┬───────┘
                    │          ↓
                    │      (repeat 3×)
                    │          ↓
                    │  ┌───────────────┐
                    │  │  1×1 Conv     │
                    │  │    1 ch       │
                    │  └───────┬───────┘
                    └──────────┼────────┘
                               ↓
                         Sigmoid
                               ↓
                        Output Mask
                       ┌─────────────┐
                       │ 512 × 512 × 1│
                       │   (0 to 1)   │
                       └──────────────┘
```

## Loss Function Workflow

```
         Predictions              Ground Truth
         (logits)                    (binary)
              │                         │
              ├─────────────┬───────────┤
              ↓             ↓           ↓
    ┌──────────────┐ ┌────────────┐ ┌──────────────┐
    │ BCE Loss     │ │ Dice Loss  │ │ Focal Loss   │
    │              │ │            │ │              │
    │ Standard     │ │ Overlap    │ │ Hard mining  │
    │ pixel-wise   │ │ focused    │ │ focused      │
    └──────┬───────┘ └─────┬──────┘ └──────┬───────┘
           │               │                │
           ↓               ↓                ↓
        × 0.5           × 0.3            × 0.2
           │               │                │
           └───────────────┼────────────────┘
                           ↓
                   Combined Loss
                           ↓
                  Backpropagation
                           ↓
                   Update Weights
```

## Training Progress Timeline

```
Epoch   LR          Training     Validation    Actions
        (×10⁻³)     IoU/Loss     IoU/Loss      
─────────────────────────────────────────────────────────────
  1     1.00        0.45/0.35    0.42/0.38    Initial training
  5     1.00        0.68/0.22    0.64/0.25    Save predictions
 10     1.00        0.75/0.18    0.71/0.21    Save checkpoint
 15     1.00        0.80/0.15    0.74/0.19    
 20     0.50 ⬇      0.83/0.12    0.77/0.17    LR reduced (plateau)
 25     0.50        0.85/0.11    0.79/0.16    Save predictions
 30     0.50        0.87/0.10    0.81/0.15    Save checkpoint
 35     0.25 ⬇      0.88/0.09    0.82/0.14    LR reduced (plateau)
 40     0.25        0.89/0.08    0.83/0.14    Save checkpoint
 45     0.25        0.90/0.08    0.84/0.13    Save predictions
 50     0.125⬇      0.90/0.08    0.84/0.13    Best model! ⭐
 55     0.125       0.90/0.08    0.84/0.13    No improvement
 60     0.125       0.90/0.08    0.84/0.13    No improvement
 65     0.125       0.90/0.08    0.84/0.13    Early stopping! ⛔
─────────────────────────────────────────────────────────────
                                               Training complete
                                               Best Val IoU: 0.84
                                               Total time: 2h 15m
```

## Decision Tree: Which Script to Run?

```
                    Start Here
                        │
                        ↓
              First time using this?
                  /         \
              YES             NO
               ↓               ↓
        test_setup.py    Already trained?
               ↓             /         \
        Everything OK?    YES           NO
          /         \      ↓             ↓
       YES          NO    Want to    check_dataset.py
        ↓            ↓    evaluate?      ↓
check_dataset.py   Fix     /    \    Looks good?
        ↓          issues YES    NO      ↓
   Looks good?            │      │   train_unet.py
    /        \            ↓      ↓       ↓
 YES         NO      evaluate.py  inference.py  Training
  ↓           ↓           ↓           ↓      complete?
train_unet.py Fix         Done!      Test      /    \
              data                   samples YES    NO
                                               ↓     ↓
                                          evaluate.py Wait
                                               ↓     or
                                          inference.py Monitor
                                               ↓
                                            Done! ✅
```

## Metric Interpretation Guide

```
┌──────────────────────────────────────────────────────────┐
│                   Metric Thresholds                       │
├──────────────────────────────────────────────────────────┤
│ IoU (Intersection over Union):                          │
│   0.90 - 1.00  ★★★★★  Excellent                         │
│   0.80 - 0.90  ★★★★   Very Good                         │
│   0.70 - 0.80  ★★★    Good                              │
│   0.60 - 0.70  ★★     Fair                              │
│   0.00 - 0.60  ★      Poor                              │
│                                                          │
│ Dice Coefficient:                                        │
│   0.90 - 1.00  ★★★★★  Excellent                         │
│   0.85 - 0.90  ★★★★   Very Good                         │
│   0.75 - 0.85  ★★★    Good                              │
│   0.65 - 0.75  ★★     Fair                              │
│   0.00 - 0.65  ★      Poor                              │
│                                                          │
│ Precision (How many predictions are correct):           │
│   High (>0.9): Few false alarms ✓                       │
│   Low  (<0.7): Many false positives ✗                   │
│                                                          │
│ Recall (How many actual positives found):               │
│   High (>0.9): Finds most tampering ✓                   │
│   Low  (<0.7): Misses many tampered regions ✗           │
│                                                          │
│ F1 Score (Balance of Precision & Recall):               │
│   High (>0.85): Well-balanced model ✓                   │
│   Low  (<0.70): Imbalanced predictions ✗                │
└──────────────────────────────────────────────────────────┘
```

## Common Training Patterns

### Pattern 1: Normal Training
```
Loss:  ↘↘↘↘↘↘↘ (Smooth decrease)
IoU:   ↗↗↗↗↗↗↗ (Smooth increase)
Status: ✅ Good - Model learning well
```

### Pattern 2: Overfitting
```
Train Loss: ↘↘↘↘↘↘
Val Loss:   ↘↘↗↗↗↗ (Increases after initial decrease)
Status: ⚠️ Warning - Model memorizing training data
Action: Enable early stopping, increase regularization
```

### Pattern 3: Underfitting
```
Train Loss: ↘───── (Plateaus too early)
Val Loss:   ↘───── (Both plateau)
Status: ⚠️ Warning - Model not learning enough
Action: Train longer, increase learning rate, reduce regularization
```

### Pattern 4: Learning Rate Too High
```
Loss: ↘↗↘↗↘↗↘↗ (Oscillating)
Status: ⚠️ Warning - Steps too large
Action: Reduce learning rate
```

### Pattern 5: Learning Rate Too Low
```
Loss: ↘.......... (Very slow decrease)
Status: ⚠️ Warning - Steps too small
Action: Increase learning rate
```

## File Dependency Graph

```
requirements.txt
     │
     ├─► train_unet.py
     │        │
     │        ├─► Dataset (LMDB)
     │        ├─► U-Net Model
     │        ├─► Loss Functions
     │        └─► Training Loop
     │             │
     │             ↓
     │        checkpoints/
     │        predictions/
     │        training_history.png
     │
     ├─► inference.py
     │        │
     │        ├─► train_unet.py (imports UNet)
     │        ├─► checkpoints/best_model_iou.pth
     │        └─► Dataset (LMDB)
     │             │
     │             ↓
     │        test_results/
     │
     ├─► evaluate.py
     │        │
     │        ├─► train_unet.py (imports UNet)
     │        ├─► checkpoints/best_model_iou.pth
     │        └─► Dataset (LMDB)
     │             │
     │             ↓
     │        evaluation_results/
     │
     ├─► check_dataset.py
     │        │
     │        └─► Dataset (LMDB)
     │             │
     │             ↓
     │        dataset_statistics.png
     │        training_samples.png
     │        validation_samples.png
     │
     └─► test_setup.py
              │
              └─► Verifies all dependencies
```

## Quick Reference Commands

```bash
# Setup Phase
pip install -r requirements.txt      # Install dependencies
python test_setup.py                 # Verify installation
python check_dataset.py              # Check dataset

# Training Phase
python train_unet.py                 # Train model (main task)

# Evaluation Phase
python evaluate.py                   # Comprehensive evaluation
python inference.py --num_samples 10 # Test on samples

# Monitoring
# Watch training_history.png for progress
# Check predictions/ folder for visual results
# Monitor checkpoints/ for saved models
```

---

**For detailed documentation**: See README.md  
**For quick start**: See QUICKSTART.md  
**For complete overview**: See PROJECT_SUMMARY.md

