"""
Script to check dataset integrity and visualize statistics
"""
import os
import cv2
import six
import lmdb
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict


def check_lmdb_dataset(lmdb_path):
    """Check LMDB dataset and return statistics"""
    print(f"\nChecking dataset: {lmdb_path}")
    print("="*60)
    
    if not os.path.exists(lmdb_path):
        print(f"❌ Error: Dataset not found at {lmdb_path}")
        return None
    
    env = lmdb.open(lmdb_path, readonly=True, lock=False, 
                   readahead=False, meminit=False)
    
    stats = {
        'num_samples': 0,
        'image_sizes': [],
        'mask_sizes': [],
        'mask_ratios': [],
        'mask_counts': [],
        'has_mask': 0,
        'no_mask': 0,
        'errors': []
    }
    
    with env.begin(write=False) as txn:
        total_entries = txn.stat()['entries']
        num_samples = total_entries // 2
        stats['num_samples'] = num_samples
        
        print(f"Total entries in LMDB: {total_entries}")
        print(f"Number of samples (image+mask pairs): {num_samples}")
        print("-"*60)
        
        # Check first few samples
        samples_to_check = min(num_samples, 10)
        print(f"\nChecking first {samples_to_check} samples...")
        
        for i in range(1, num_samples + 1):
            try:
                # Load image
                img_key = 'image-%09d' % i
                imgbuf = txn.get(img_key.encode('utf-8'))
                
                if imgbuf is None:
                    stats['errors'].append(f"Sample {i}: Image not found")
                    continue
                
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                image = Image.open(buf).convert('RGB')
                img_array = np.array(image)
                stats['image_sizes'].append(img_array.shape)
                
                # Load mask
                lbl_key = 'label-%09d' % i
                lblbuf = txn.get(lbl_key.encode('utf-8'))
                
                if lblbuf is None:
                    stats['no_mask'] += 1
                    if i <= samples_to_check:
                        print(f"  Sample {i}: ⚠️  No mask found")
                    continue
                
                mask = cv2.imdecode(np.frombuffer(lblbuf, dtype=np.uint8), 0)
                
                if mask is None:
                    stats['errors'].append(f"Sample {i}: Failed to decode mask")
                    continue
                
                stats['mask_sizes'].append(mask.shape)
                
                # Normalize mask
                if mask.max() == 1:
                    mask = mask * 255
                
                # Calculate mask statistics
                mask_binary = (mask > 127).astype(np.uint8)
                mask_pixels = np.sum(mask_binary)
                total_pixels = mask.shape[0] * mask.shape[1]
                mask_ratio = mask_pixels / total_pixels
                
                stats['mask_ratios'].append(mask_ratio)
                
                # Count connected components (number of separate masks)
                num_labels, labels = cv2.connectedComponents(mask_binary)
                num_masks = num_labels - 1  # Subtract background
                stats['mask_counts'].append(num_masks)
                
                if mask_pixels > 0:
                    stats['has_mask'] += 1
                else:
                    stats['no_mask'] += 1
                
                if i <= samples_to_check:
                    status = "✓" if mask_pixels > 0 else "○"
                    print(f"  Sample {i}: {status} Image {img_array.shape}, Mask {mask.shape}, "
                          f"Coverage: {mask_ratio*100:.2f}%, Regions: {num_masks}")
                
            except Exception as e:
                stats['errors'].append(f"Sample {i}: {str(e)}")
                if i <= samples_to_check:
                    print(f"  Sample {i}: ❌ Error: {str(e)}")
    
    env.close()
    return stats


def print_statistics(stats):
    """Print dataset statistics"""
    if stats is None:
        return
    
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    
    print(f"Total samples: {stats['num_samples']}")
    print(f"Samples with masks: {stats['has_mask']}")
    print(f"Samples without masks: {stats['no_mask']}")
    
    if stats['image_sizes']:
        unique_sizes = set(stats['image_sizes'])
        print(f"\nImage sizes: {unique_sizes}")
    
    if stats['mask_ratios']:
        mask_ratios = np.array(stats['mask_ratios'])
        print(f"\nMask Coverage Statistics:")
        print(f"  Mean:   {mask_ratios.mean()*100:.2f}%")
        print(f"  Median: {np.median(mask_ratios)*100:.2f}%")
        print(f"  Min:    {mask_ratios.min()*100:.2f}%")
        print(f"  Max:    {mask_ratios.max()*100:.2f}%")
        print(f"  Std:    {mask_ratios.std()*100:.2f}%")
    
    if stats['mask_counts']:
        mask_counts = np.array(stats['mask_counts'])
        print(f"\nNumber of Mask Regions per Image:")
        print(f"  Mean:   {mask_counts.mean():.2f}")
        print(f"  Median: {np.median(mask_counts):.0f}")
        print(f"  Min:    {mask_counts.min():.0f}")
        print(f"  Max:    {mask_counts.max():.0f}")
    
    if stats['errors']:
        print(f"\n⚠️  Errors found: {len(stats['errors'])}")
        for error in stats['errors'][:5]:
            print(f"  - {error}")
        if len(stats['errors']) > 5:
            print(f"  ... and {len(stats['errors']) - 5} more")
    
    print("="*60)


def plot_statistics(train_stats, val_stats, save_path='dataset_statistics.png'):
    """Plot dataset statistics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Mask coverage histogram
    if train_stats['mask_ratios']:
        axes[0, 0].hist(np.array(train_stats['mask_ratios'])*100, bins=50, 
                       alpha=0.7, label='Training', edgecolor='black')
    if val_stats['mask_ratios']:
        axes[0, 0].hist(np.array(val_stats['mask_ratios'])*100, bins=50, 
                       alpha=0.7, label='Validation', edgecolor='black')
    axes[0, 0].set_xlabel('Mask Coverage (%)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Mask Coverage')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Number of mask regions histogram
    if train_stats['mask_counts']:
        axes[0, 1].hist(train_stats['mask_counts'], bins=range(0, max(train_stats['mask_counts'])+2), 
                       alpha=0.7, label='Training', edgecolor='black')
    if val_stats['mask_counts']:
        axes[0, 1].hist(val_stats['mask_counts'], bins=range(0, max(val_stats['mask_counts'])+2), 
                       alpha=0.7, label='Validation', edgecolor='black')
    axes[0, 1].set_xlabel('Number of Mask Regions')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Number of Mask Regions')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Box plot of mask coverage
    if train_stats['mask_ratios'] and val_stats['mask_ratios']:
        data = [np.array(train_stats['mask_ratios'])*100, 
                np.array(val_stats['mask_ratios'])*100]
        axes[1, 0].boxplot(data, labels=['Training', 'Validation'])
        axes[1, 0].set_ylabel('Mask Coverage (%)')
        axes[1, 0].set_title('Mask Coverage Comparison')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Dataset size comparison
    categories = ['Training', 'Validation']
    samples = [train_stats['num_samples'], val_stats['num_samples']]
    with_masks = [train_stats['has_mask'], val_stats['has_mask']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, samples, width, label='Total Samples', alpha=0.8)
    axes[1, 1].bar(x + width/2, with_masks, width, label='With Masks', alpha=0.8)
    axes[1, 1].set_xlabel('Dataset')
    axes[1, 1].set_ylabel('Number of Samples')
    axes[1, 1].set_title('Dataset Size Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(categories)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nStatistics plot saved to {save_path}")
    plt.close()


def visualize_samples(lmdb_path, num_samples=4, save_path='dataset_samples.png'):
    """Visualize sample images and masks from dataset"""
    env = lmdb.open(lmdb_path, readonly=True, lock=False, 
                   readahead=False, meminit=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    with env.begin(write=False) as txn:
        for i in range(num_samples):
            idx = i + 1
            
            # Load image
            img_key = 'image-%09d' % idx
            imgbuf = txn.get(img_key.encode('utf-8'))
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            image = Image.open(buf).convert('RGB')
            
            # Load mask
            lbl_key = 'label-%09d' % idx
            lblbuf = txn.get(lbl_key.encode('utf-8'))
            mask = cv2.imdecode(np.frombuffer(lblbuf, dtype=np.uint8), 0)
            
            if mask.max() == 1:
                mask = mask * 255
            
            # Calculate statistics
            mask_binary = (mask > 127).astype(np.uint8)
            mask_ratio = np.sum(mask_binary) / mask_binary.size
            num_labels, _ = cv2.connectedComponents(mask_binary)
            num_regions = num_labels - 1
            
            # Plot
            if num_samples == 1:
                axes[0].imshow(image)
                axes[0].set_title(f'Sample {idx}: Original Image')
                axes[0].axis('off')
                
                axes[1].imshow(mask, cmap='gray')
                axes[1].set_title(f'Mask (Coverage: {mask_ratio*100:.1f}%)')
                axes[1].axis('off')
                
                # Overlay
                overlay = np.array(image).copy()
                overlay[mask > 127] = [255, 0, 0]  # Red overlay
                axes[2].imshow(overlay)
                axes[2].set_title(f'Overlay ({num_regions} regions)')
                axes[2].axis('off')
            else:
                axes[i, 0].imshow(image)
                axes[i, 0].set_title(f'Sample {idx}: Original Image')
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(mask, cmap='gray')
                axes[i, 1].set_title(f'Mask (Coverage: {mask_ratio*100:.1f}%)')
                axes[i, 1].axis('off')
                
                # Overlay
                overlay = np.array(image).copy()
                overlay[mask > 127] = [255, 0, 0]  # Red overlay
                axes[i, 2].imshow(overlay)
                axes[i, 2].set_title(f'Overlay ({num_regions} regions)')
                axes[i, 2].axis('off')
    
    env.close()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Sample visualization saved to {save_path}")
    plt.close()


def main():
    print("="*60)
    print("Dataset Checker for Document Tampering Detection")
    print("="*60)
    
    # Check training dataset
    train_stats = check_lmdb_dataset('DocTamperV1-TrainingSet')
    if train_stats:
        print_statistics(train_stats)
    
    # Check validation dataset
    val_stats = check_lmdb_dataset('DocTamperV1-TestingSet')
    if val_stats:
        print_statistics(val_stats)
    
    # Plot statistics
    if train_stats and val_stats:
        plot_statistics(train_stats, val_stats)
    
    # Visualize samples
    print("\n" + "="*60)
    print("Generating sample visualizations...")
    print("="*60)
    
    if train_stats:
        visualize_samples('DocTamperV1-TrainingSet', num_samples=4, 
                         save_path='training_samples.png')
    
    if val_stats:
        visualize_samples('DocTamperV1-TestingSet', num_samples=4, 
                         save_path='validation_samples.png')
    
    print("\n" + "="*60)
    print("Dataset check completed!")
    print("="*60)
    
    # Recommendations
    print("\n📋 Recommendations:")
    if train_stats:
        avg_coverage = np.mean(train_stats['mask_ratios']) * 100 if train_stats['mask_ratios'] else 0
        if avg_coverage < 5:
            print("  ⚠️  Average mask coverage is very low (<5%).")
            print("     Consider using Focal Loss or weighted BCE to handle class imbalance.")
        if avg_coverage > 50:
            print("  ℹ️  Average mask coverage is high (>50%).")
            print("     Standard BCE might work well.")
    
    print("  ✓ The training script uses Combined Loss (BCE + Dice + Focal)")
    print("    which handles various mask sizes and imbalance effectively.")
    print("\nYou can now run: python train_unet.py")


if __name__ == '__main__':
    main()

