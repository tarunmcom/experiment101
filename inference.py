import os
import cv2
import six
import lmdb
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from train_unet import UNet
import argparse


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
        print(f"Validation IoU: {checkpoint['val_iou']:.4f}")
    
    return model


def load_image_from_lmdb(lmdb_path, index):
    """Load image and mask from LMDB"""
    env = lmdb.open(lmdb_path, readonly=True, lock=False, 
                   readahead=False, meminit=False)
    
    with env.begin(write=False) as txn:
        # Load image
        img_key = 'image-%09d' % index
        imgbuf = txn.get(img_key.encode('utf-8'))
        
        if imgbuf is None:
            raise ValueError(f"Image not found for key: {img_key}")
        
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        image = Image.open(buf).convert('RGB')
        
        # Load mask
        lbl_key = 'label-%09d' % index
        lblbuf = txn.get(lbl_key.encode('utf-8'))
        
        if lblbuf is None:
            mask = None
        else:
            mask = cv2.imdecode(np.frombuffer(lblbuf, dtype=np.uint8), 0)
            if mask.max() == 1:
                mask = mask * 255
    
    env.close()
    return image, mask


def predict(model, image, device, threshold=0.5):
    """Predict mask for a given image"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.sigmoid(output).squeeze().cpu().numpy()
    
    # Apply threshold
    binary_mask = (prediction > threshold).astype(np.uint8) * 255
    
    return prediction, binary_mask


def calculate_metrics(pred_mask, gt_mask, threshold=0.5):
    """Calculate evaluation metrics"""
    pred_binary = (pred_mask > threshold).astype(np.float32)
    gt_binary = (gt_mask > 127).astype(np.float32)
    
    # IoU
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    iou = intersection / (union + 1e-8)
    
    # Dice
    dice = (2 * intersection) / (pred_binary.sum() + gt_binary.sum() + 1e-8)
    
    # Precision and Recall
    true_positive = intersection
    false_positive = pred_binary.sum() - true_positive
    false_negative = gt_binary.sum() - true_positive
    
    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return {
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def visualize_prediction(image, gt_mask, pred_mask, binary_mask, metrics=None, save_path=None):
    """Visualize prediction results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Ground truth mask
    if gt_mask is not None:
        axes[0, 1].imshow(gt_mask, cmap='gray')
        axes[0, 1].set_title('Ground Truth Mask', fontsize=14, fontweight='bold')
    else:
        axes[0, 1].text(0.5, 0.5, 'No Ground Truth', ha='center', va='center', fontsize=16)
        axes[0, 1].set_title('Ground Truth Mask', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Predicted mask (probability)
    im = axes[1, 0].imshow(pred_mask, cmap='jet', vmin=0, vmax=1)
    axes[1, 0].set_title('Predicted Mask (Probability)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Binary predicted mask
    axes[1, 1].imshow(binary_mask, cmap='gray')
    title = 'Binary Prediction (threshold=0.5)'
    if metrics:
        title += f"\nIoU: {metrics['iou']:.3f} | Dice: {metrics['dice']:.3f}"
    axes[1, 1].set_title(title, fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def test_on_dataset(model, lmdb_path, device, num_samples=10, save_dir='test_results'):
    """Test model on multiple samples from dataset"""
    os.makedirs(save_dir, exist_ok=True)
    
    env = lmdb.open(lmdb_path, readonly=True, lock=False, 
                   readahead=False, meminit=False)
    
    with env.begin(write=False) as txn:
        num_total_samples = txn.stat()['entries'] // 2
    
    print(f"Testing on {num_samples} samples from {lmdb_path}")
    print(f"Total samples available: {num_total_samples}")
    print("-" * 60)
    
    all_metrics = []
    
    for i in range(1, min(num_samples + 1, num_total_samples + 1)):
        # Load image and mask
        image, gt_mask = load_image_from_lmdb(lmdb_path, i)
        
        # Predict
        pred_mask, binary_mask = predict(model, image, device)
        
        # Calculate metrics
        if gt_mask is not None:
            metrics = calculate_metrics(pred_mask, gt_mask)
            all_metrics.append(metrics)
            
            print(f"Sample {i}:")
            print(f"  IoU:       {metrics['iou']:.4f}")
            print(f"  Dice:      {metrics['dice']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}")
        else:
            metrics = None
            print(f"Sample {i}: No ground truth available")
        
        # Visualize
        save_path = os.path.join(save_dir, f'sample_{i}.png')
        visualize_prediction(image, gt_mask, pred_mask, binary_mask, metrics, save_path)
    
    # Print average metrics
    if all_metrics:
        print("\n" + "=" * 60)
        print("Average Metrics:")
        print("=" * 60)
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }
        for key, value in avg_metrics.items():
            print(f"  {key.capitalize():12s}: {value:.4f}")
        print("=" * 60)
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description='U-Net Inference for Document Tampering Detection')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model_iou.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--lmdb_path', type=str, default='DocTamperV1-TestingSet',
                       help='Path to LMDB dataset')
    parser.add_argument('--index', type=int, default=None,
                       help='Index of specific image to test (if not provided, tests multiple samples)')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to test (when index is not specified)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary mask prediction')
    parser.add_argument('--output_dir', type=str, default='test_results',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("U-Net Inference for Document Tampering Detection")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"LMDB Path: {args.lmdb_path}")
    print("=" * 60)
    
    # Load model
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    
    # Test on single image or multiple samples
    if args.index is not None:
        print(f"\nTesting on sample {args.index}...")
        
        # Load image and mask
        image, gt_mask = load_image_from_lmdb(args.lmdb_path, args.index)
        
        # Predict
        pred_mask, binary_mask = predict(model, image, device, args.threshold)
        
        # Calculate metrics
        if gt_mask is not None:
            metrics = calculate_metrics(pred_mask, gt_mask, args.threshold)
            
            print("\nMetrics:")
            print(f"  IoU:       {metrics['iou']:.4f}")
            print(f"  Dice:      {metrics['dice']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}")
        else:
            metrics = None
            print("\nNo ground truth available for this sample")
        
        # Visualize
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, f'sample_{args.index}.png')
        visualize_prediction(image, gt_mask, pred_mask, binary_mask, metrics, save_path)
    else:
        print(f"\nTesting on {args.num_samples} samples...")
        test_on_dataset(model, args.lmdb_path, device, args.num_samples, args.output_dir)
    
    print("\nInference completed!")


if __name__ == '__main__':
    main()

