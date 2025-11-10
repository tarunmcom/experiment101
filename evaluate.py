"""
Comprehensive evaluation script for trained U-Net model
"""
import os
import cv2
import six
import lmdb
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from train_unet import UNet
import argparse
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    model = UNet(n_channels=3, n_classes=1, bilinear=True).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"Trained for {checkpoint['epoch']} epochs")
    if 'val_iou' in checkpoint:
        print(f"Checkpoint Val IoU: {checkpoint['val_iou']:.4f}")
    
    return model


def evaluate_dataset(model, lmdb_path, device, threshold=0.5):
    """Evaluate model on entire dataset"""
    env = lmdb.open(lmdb_path, readonly=True, lock=False, 
                   readahead=False, meminit=False)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    with env.begin(write=False) as txn:
        num_samples = txn.stat()['entries'] // 2
    
    print(f"\nEvaluating on {num_samples} samples from {lmdb_path}")
    print("="*60)
    
    all_metrics = []
    all_predictions = []
    all_targets = []
    
    model.eval()
    
    with torch.no_grad():
        for i in tqdm(range(1, num_samples + 1), desc="Evaluating"):
            with env.begin(write=False) as txn:
                # Load image
                img_key = 'image-%09d' % i
                imgbuf = txn.get(img_key.encode('utf-8'))
                
                if imgbuf is None:
                    continue
                
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                image = Image.open(buf).convert('RGB')
                
                # Load mask
                lbl_key = 'label-%09d' % i
                lblbuf = txn.get(lbl_key.encode('utf-8'))
                
                if lblbuf is None:
                    continue
                
                mask = cv2.imdecode(np.frombuffer(lblbuf, dtype=np.uint8), 0)
                if mask.max() == 1:
                    mask = mask * 255
            
            # Preprocess
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            # Predict
            output = model(img_tensor)
            pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()
            pred_binary = (pred_prob > threshold).astype(np.uint8)
            
            # Ground truth
            gt_binary = (mask > 127).astype(np.uint8)
            
            # Store for ROC/PR curves
            all_predictions.append(pred_prob.flatten())
            all_targets.append(gt_binary.flatten())
            
            # Calculate metrics
            metrics = calculate_metrics(pred_binary, gt_binary, pred_prob)
            all_metrics.append(metrics)
    
    env.close()
    
    return all_metrics, all_predictions, all_targets


def calculate_metrics(pred_binary, gt_binary, pred_prob=None):
    """Calculate comprehensive evaluation metrics"""
    pred_flat = pred_binary.flatten().astype(np.float32)
    gt_flat = gt_binary.flatten().astype(np.float32)
    
    # Basic metrics
    true_positive = np.sum((pred_flat == 1) & (gt_flat == 1))
    true_negative = np.sum((pred_flat == 0) & (gt_flat == 0))
    false_positive = np.sum((pred_flat == 1) & (gt_flat == 0))
    false_negative = np.sum((pred_flat == 0) & (gt_flat == 1))
    
    # IoU
    intersection = true_positive
    union = true_positive + false_positive + false_negative
    iou = intersection / (union + 1e-8)
    
    # Dice
    dice = (2 * intersection) / (pred_flat.sum() + gt_flat.sum() + 1e-8)
    
    # Precision, Recall, F1
    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Accuracy
    accuracy = (true_positive + true_negative) / (pred_flat.size + 1e-8)
    
    # Specificity
    specificity = true_negative / (true_negative + false_positive + 1e-8)
    
    return {
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'specificity': specificity,
        'tp': true_positive,
        'tn': true_negative,
        'fp': false_positive,
        'fn': false_negative
    }


def print_evaluation_results(all_metrics):
    """Print comprehensive evaluation results"""
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    
    # Calculate mean and std for each metric
    metrics_summary = {}
    for key in all_metrics[0].keys():
        if key not in ['tp', 'tn', 'fp', 'fn']:
            values = [m[key] for m in all_metrics]
            metrics_summary[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
    
    # Print results
    print("\nPer-Image Metrics (mean ± std):")
    print("-"*60)
    for metric, stats in metrics_summary.items():
        print(f"{metric.upper():15s}: {stats['mean']:.4f} ± {stats['std']:.4f} "
              f"[{stats['min']:.4f}, {stats['max']:.4f}] (median: {stats['median']:.4f})")
    
    # Print confusion matrix statistics
    print("\nConfusion Matrix (Total):")
    print("-"*60)
    total_tp = sum(m['tp'] for m in all_metrics)
    total_tn = sum(m['tn'] for m in all_metrics)
    total_fp = sum(m['fp'] for m in all_metrics)
    total_fn = sum(m['fn'] for m in all_metrics)
    
    print(f"True Positives:  {total_tp:,}")
    print(f"True Negatives:  {total_tn:,}")
    print(f"False Positives: {total_fp:,}")
    print(f"False Negatives: {total_fn:,}")
    
    # Global metrics
    print("\nGlobal Metrics:")
    print("-"*60)
    global_iou = total_tp / (total_tp + total_fp + total_fn + 1e-8)
    global_dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + 1e-8)
    global_precision = total_tp / (total_tp + total_fp + 1e-8)
    global_recall = total_tp / (total_tp + total_fn + 1e-8)
    global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall + 1e-8)
    global_accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + 1e-8)
    
    print(f"IoU:        {global_iou:.4f}")
    print(f"Dice:       {global_dice:.4f}")
    print(f"Precision:  {global_precision:.4f}")
    print(f"Recall:     {global_recall:.4f}")
    print(f"F1 Score:   {global_f1:.4f}")
    print(f"Accuracy:   {global_accuracy:.4f}")
    
    print("="*60)
    
    return metrics_summary


def plot_metric_distributions(all_metrics, save_path='metric_distributions.png'):
    """Plot distributions of evaluation metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    metrics_to_plot = ['iou', 'dice', 'precision', 'recall', 'f1', 'accuracy']
    
    for idx, metric in enumerate(metrics_to_plot):
        row = idx // 3
        col = idx % 3
        
        values = [m[metric] for m in all_metrics]
        
        axes[row, col].hist(values, bins=50, edgecolor='black', alpha=0.7)
        axes[row, col].axvline(np.mean(values), color='red', linestyle='--', 
                              linewidth=2, label=f'Mean: {np.mean(values):.3f}')
        axes[row, col].axvline(np.median(values), color='green', linestyle='--', 
                              linewidth=2, label=f'Median: {np.median(values):.3f}')
        axes[row, col].set_xlabel(metric.upper())
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].set_title(f'Distribution of {metric.upper()}')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nMetric distributions saved to {save_path}")
    plt.close()


def plot_roc_pr_curves(all_predictions, all_targets, save_path='roc_pr_curves.png'):
    """Plot ROC and Precision-Recall curves"""
    # Flatten all predictions and targets
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_predictions)
    
    # Sample for faster computation if dataset is very large
    if len(y_true) > 10000000:  # 10M pixels
        indices = np.random.choice(len(y_true), 10000000, replace=False)
        y_true = y_true[indices]
        y_pred = y_pred[indices]
    
    print("\nCalculating ROC and PR curves...")
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # ROC curve
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)
    
    # Precision-Recall curve
    axes[1].plot(recall, precision, color='green', lw=2, 
                label=f'PR curve (AUC = {pr_auc:.4f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"ROC and PR curves saved to {save_path}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  PR AUC:  {pr_auc:.4f}")
    plt.close()


def plot_confusion_matrix(all_metrics, save_path='confusion_matrix.png'):
    """Plot normalized confusion matrix"""
    total_tp = sum(m['tp'] for m in all_metrics)
    total_tn = sum(m['tn'] for m in all_metrics)
    total_fp = sum(m['fp'] for m in all_metrics)
    total_fn = sum(m['fn'] for m in all_metrics)
    
    cm = np.array([[total_tn, total_fp],
                   [total_fn, total_tp]])
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Raw confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title('Confusion Matrix (Raw Counts)')
    
    # Normalized confusion matrix
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues', ax=axes[1],
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_title('Confusion Matrix (Normalized)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate U-Net model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model_iou.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--lmdb_path', type=str, default='DocTamperV1-TestingSet',
                       help='Path to LMDB dataset')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary prediction')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for evaluation')
    
    args = parser.parse_args()
    
    print("="*60)
    print("U-Net Model Evaluation")
    print("="*60)
    print(f"Device: {args.device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"LMDB Path: {args.lmdb_path}")
    print(f"Threshold: {args.threshold}")
    print("="*60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    
    # Evaluate
    all_metrics, all_predictions, all_targets = evaluate_dataset(
        model, args.lmdb_path, device, args.threshold
    )
    
    # Print results
    metrics_summary = print_evaluation_results(all_metrics)
    
    # Plot metric distributions
    plot_metric_distributions(all_metrics, 
                             os.path.join(args.output_dir, 'metric_distributions.png'))
    
    # Plot ROC and PR curves
    plot_roc_pr_curves(all_predictions, all_targets,
                      os.path.join(args.output_dir, 'roc_pr_curves.png'))
    
    # Plot confusion matrix
    plot_confusion_matrix(all_metrics,
                         os.path.join(args.output_dir, 'confusion_matrix.png'))
    
    # Save metrics to file
    results_file = os.path.join(args.output_dir, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("U-Net Model Evaluation Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset: {args.lmdb_path}\n")
        f.write(f"Threshold: {args.threshold}\n")
        f.write(f"Number of samples: {len(all_metrics)}\n\n")
        
        f.write("Per-Image Metrics (mean ± std):\n")
        f.write("-"*60 + "\n")
        for metric, stats in metrics_summary.items():
            f.write(f"{metric.upper():15s}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                   f"[{stats['min']:.4f}, {stats['max']:.4f}] (median: {stats['median']:.4f})\n")
    
    print(f"\nEvaluation results saved to {results_file}")
    print("\n" + "="*60)
    print("Evaluation completed!")
    print("="*60)


if __name__ == '__main__':
    main()

