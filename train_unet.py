import os
import cv2
import six
import lmdb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import requests
import json
import warnings
warnings.filterwarnings('ignore')


# ==================== Slack Notifications ====================
def send_slack(msg, webhook_url=None):
    """Send a message to Slack webhook"""
    if webhook_url is None:
        return  # Skip if no webhook configured
    
    try:
        data = {"text": msg}
        response = requests.post(webhook_url, json.dumps(data), timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"⚠️  Failed to send Slack notification: {e}")
        return False


# ==================== Dataset Class ====================
class LMDBDataset(Dataset):
    """Custom Dataset for loading images and masks from LMDB format"""
    
    def __init__(self, lmdb_path, transform=None, augment=False):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, 
                            readahead=False, meminit=False)
        self.transform = transform
        self.augment = augment
        
        # Get the number of samples
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries'] // 2  # Divided by 2 (image + label pairs)
        
        print(f"Dataset initialized with {self.length} samples from {lmdb_path}")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with self.env.begin(write=False) as txn:
            # Load image
            img_key = 'image-%09d' % (idx + 1)  # LMDB indices start from 1
            imgbuf = txn.get(img_key.encode('utf-8'))
            
            if imgbuf is None:
                raise ValueError(f"Image not found for key: {img_key}")
            
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            image = Image.open(buf).convert('RGB')
            
            # Load mask
            lbl_key = 'label-%09d' % (idx + 1)
            lblbuf = txn.get(lbl_key.encode('utf-8'))
            
            if lblbuf is None:
                raise ValueError(f"Label not found for key: {lbl_key}")
            
            mask = cv2.imdecode(np.frombuffer(lblbuf, dtype=np.uint8), 0)
            
            # Normalize mask to 0-255 if needed
            if mask.max() == 1:
                mask = mask * 255
            
            # Convert to PIL for consistent transforms
            mask = Image.fromarray(mask)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask)
            mask = (mask > 0.5).float()  # Binary mask (0 or 1)
        
        return image, mask


# ==================== U-Net Model ====================
class DoubleConv(nn.Module):
    """Double Convolution block: (Conv -> BN -> ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Concatenate with skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net Architecture"""
    
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# ==================== Loss Functions ====================
class DiceLoss(nn.Module):
    """Dice Loss - Good for imbalanced segmentation tasks"""
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class FocalLoss(nn.Module):
    """Focal Loss - Focuses on hard examples, good for imbalanced data"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(predictions, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Combined Loss: BCE + Dice + Focal for robust training"""
    
    def __init__(self, bce_weight=0.5, dice_weight=0.3, focal_weight=0.2):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
    
    def forward(self, predictions, targets):
        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        focal_loss = self.focal(predictions, targets)
        
        total_loss = (self.bce_weight * bce_loss + 
                     self.dice_weight * dice_loss + 
                     self.focal_weight * focal_loss)
        
        return total_loss, bce_loss, dice_loss, focal_loss


# ==================== Metrics ====================
def calculate_iou(predictions, targets, threshold=0.5):
    """Calculate Intersection over Union (IoU)"""
    predictions = (torch.sigmoid(predictions) > threshold).float()
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    return iou.item()


def calculate_dice(predictions, targets, threshold=0.5):
    """Calculate Dice Coefficient"""
    predictions = (torch.sigmoid(predictions) > threshold).float()
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    intersection = (predictions * targets).sum()
    dice = (2. * intersection) / (predictions.sum() + targets.sum() + 1e-8)
    
    return dice.item()


# ==================== Training and Validation ====================
def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_bce = 0.0
    running_dice = 0.0
    running_focal = 0.0
    running_iou = 0.0
    running_dice_metric = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Calculate loss
        total_loss, bce_loss, dice_loss, focal_loss = criterion(outputs, masks)
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Calculate metrics
        iou = calculate_iou(outputs, masks)
        dice_metric = calculate_dice(outputs, masks)
        
        # Update running metrics
        running_loss += total_loss.item()
        running_bce += bce_loss.item()
        running_dice += dice_loss.item()
        running_focal += focal_loss.item()
        running_iou += iou
        running_dice_metric += dice_metric
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            'iou': f'{iou:.4f}',
            'dice': f'{dice_metric:.4f}'
        })
    
    num_batches = len(dataloader)
    return {
        'loss': running_loss / num_batches,
        'bce': running_bce / num_batches,
        'dice_loss': running_dice / num_batches,
        'focal': running_focal / num_batches,
        'iou': running_iou / num_batches,
        'dice': running_dice_metric / num_batches
    }


def validate_epoch(model, dataloader, criterion, device, epoch):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    running_bce = 0.0
    running_dice = 0.0
    running_focal = 0.0
    running_iou = 0.0
    running_dice_metric = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            total_loss, bce_loss, dice_loss, focal_loss = criterion(outputs, masks)
            
            # Calculate metrics
            iou = calculate_iou(outputs, masks)
            dice_metric = calculate_dice(outputs, masks)
            
            # Update running metrics
            running_loss += total_loss.item()
            running_bce += bce_loss.item()
            running_dice += dice_loss.item()
            running_focal += focal_loss.item()
            running_iou += iou
            running_dice_metric += dice_metric
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'iou': f'{iou:.4f}',
                'dice': f'{dice_metric:.4f}'
            })
    
    num_batches = len(dataloader)
    return {
        'loss': running_loss / num_batches,
        'bce': running_bce / num_batches,
        'dice_loss': running_dice / num_batches,
        'focal': running_focal / num_batches,
        'iou': running_iou / num_batches,
        'dice': running_dice_metric / num_batches
    }


def save_sample_predictions(model, dataloader, device, epoch, save_dir='predictions'):
    """Save sample predictions for visualization"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # Get first batch
    images, masks = next(iter(dataloader))
    images = images.to(device)
    masks = masks.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        predictions = torch.sigmoid(outputs) > 0.5
    
    # Save first 4 samples
    num_samples = min(4, images.size(0))
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    for i in range(num_samples):
        # Original image
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        # Ground truth mask
        gt_mask = masks[i, 0].cpu().numpy()
        
        # Predicted mask
        pred_mask = predictions[i, 0].cpu().numpy()
        
        if num_samples == 1:
            axes[0].imshow(img)
            axes[0].set_title('Image')
            axes[0].axis('off')
            
            axes[1].imshow(gt_mask, cmap='gray')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            axes[2].imshow(pred_mask, cmap='gray')
            axes[2].set_title('Prediction')
            axes[2].axis('off')
        else:
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(gt_mask, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_mask, cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_history(history, save_path='training_history.png'):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # IoU
    axes[0, 1].plot(history['train_iou'], label='Train IoU', marker='o')
    axes[0, 1].plot(history['val_iou'], label='Val IoU', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].set_title('Training and Validation IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Dice
    axes[1, 0].plot(history['train_dice'], label='Train Dice', marker='o')
    axes[1, 0].plot(history['val_dice'], label='Val Dice', marker='s')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice Coefficient')
    axes[1, 0].set_title('Training and Validation Dice')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning Rate
    axes[1, 1].plot(history['learning_rate'], label='Learning Rate', marker='o', color='green')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ==================== Main Training Function ====================
def main():
    # Configuration
    config = {
        'train_path': 'DocTamperV1-TrainingSet',
        'val_path': 'DocTamperV1-TestingSet',
        'batch_size': 8,
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'img_size': 512,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'checkpoints',
        'early_stopping_patience': 15,
        'min_delta': 1e-4,
        'compile_model': False,  # Set to True for ~20-50% speedup (requires PyTorch 2.0+)
        
        # Slack notifications
        'slack_webhook': 'https://hooks.slack.com/services/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',
        'slack_enabled': True,  # Set to False to disable notifications
        'slack_update_frequency': 5,  # Send update every N epochs
    }
    
    print("="*60)
    print("U-Net Training for Document Tampering Detection")
    print("="*60)
    print(f"Device: {config['device']}")
    print(f"Training path: {config['train_path']}")
    print(f"Validation path: {config['val_path']}")
    print("="*60)
    
    # Create directories
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs('predictions', exist_ok=True)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = LMDBDataset(config['train_path'], transform=train_transform)
    val_dataset = LMDBDataset(config['val_path'], transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: {config['batch_size']}")
    print("="*60)
    
    # Initialize model
    device = torch.device(config['device'])
    model = UNet(n_channels=3, n_classes=1, bilinear=True).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Compile model for faster training (PyTorch 2.0+)
    if config.get('compile_model', False):
        try:
            print("Compiling model with torch.compile()...")
            model = torch.compile(model, mode='default')
            print("✓ Model compiled successfully!")
        except Exception as e:
            print(f"⚠️  Model compilation not available: {e}")
            print("   Continuing with uncompiled model...")
    
    print("="*60)
    
    # Loss function with weights to handle imbalance
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.3, focal_weight=0.2)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], 
                           weight_decay=config['weight_decay'])
    
    # Learning rate scheduler - ReduceLROnPlateau based on validation loss
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, 
                                 verbose=True, min_lr=1e-7)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou': [], 'val_iou': [],
        'train_dice': [], 'val_dice': [],
        'learning_rate': []
    }
    
    # Early stopping
    best_val_loss = float('inf')
    best_val_iou = 0.0
    patience_counter = 0
    
    # Training loop
    print("Starting training...")
    print("="*60)
    
    # Send training start notification
    if config.get('slack_enabled', False):
        slack_webhook = config.get('slack_webhook')
        msg = f"🚀 *U-Net Training Started*\n" \
              f"• Device: {config['device']}\n" \
              f"• Training samples: {len(train_dataset)}\n" \
              f"• Validation samples: {len(val_dataset)}\n" \
              f"• Batch size: {config['batch_size']}\n" \
              f"• Max epochs: {config['num_epochs']}\n" \
              f"• Learning rate: {config['learning_rate']}\n" \
              f"• Model parameters: {num_params:,}"
        send_slack(msg, slack_webhook)
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        print("-"*60)
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, epoch)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_iou'].append(train_metrics['iou'])
        history['val_iou'].append(val_metrics['iou'])
        history['train_dice'].append(train_metrics['dice'])
        history['val_dice'].append(val_metrics['dice'])
        history['learning_rate'].append(current_lr)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Train IoU:  {train_metrics['iou']:.4f} | Val IoU:  {val_metrics['iou']:.4f}")
        print(f"  Train Dice: {train_metrics['dice']:.4f} | Val Dice: {val_metrics['dice']:.4f}")
        print(f"  Learning Rate: {current_lr:.2e}")
        
        # Send periodic Slack updates
        if config.get('slack_enabled', False) and epoch % config.get('slack_update_frequency', 5) == 0:
            slack_webhook = config.get('slack_webhook')
            msg = f"📊 *Epoch {epoch}/{config['num_epochs']} Update*\n" \
                  f"• Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}\n" \
                  f"• Train IoU: {train_metrics['iou']:.4f} | Val IoU: {val_metrics['iou']:.4f}\n" \
                  f"• Train Dice: {train_metrics['dice']:.4f} | Val Dice: {val_metrics['dice']:.4f}\n" \
                  f"• Learning Rate: {current_lr:.2e}"
            send_slack(msg, slack_webhook)
        
        # Update learning rate scheduler
        scheduler.step(val_metrics['loss'])
        
        # Save sample predictions every 5 epochs
        if epoch % 5 == 0:
            save_sample_predictions(model, val_loader, device, epoch)
        
        # Save best model based on validation IoU
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': best_val_iou,
                'val_loss': val_metrics['loss'],
            }, os.path.join(config['save_dir'], 'best_model_iou.pth'))
            print(f"  ✓ Saved best model (IoU: {best_val_iou:.4f})")
            
            # Send Slack notification for new best model
            if config.get('slack_enabled', False):
                slack_webhook = config.get('slack_webhook')
                msg = f"🎯 *New Best Model! (IoU)*\n" \
                      f"• Epoch: {epoch}/{config['num_epochs']}\n" \
                      f"• Val IoU: {best_val_iou:.4f}\n" \
                      f"• Val Loss: {val_metrics['loss']:.4f}\n" \
                      f"• Val Dice: {val_metrics['dice']:.4f}"
                send_slack(msg, slack_webhook)
        
        # Early stopping check
        if val_metrics['loss'] < (best_val_loss - config['min_delta']):
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            # Save best model based on validation loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_iou': val_metrics['iou'],
            }, os.path.join(config['save_dir'], 'best_model_loss.pth'))
            print(f"  ✓ Saved best model (Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  Early stopping counter: {patience_counter}/{config['early_stopping_patience']}")
        
        # Check for early stopping
        if patience_counter >= config['early_stopping_patience']:
            print("\n" + "="*60)
            print("Early stopping triggered!")
            print("="*60)
            
            # Send early stopping notification
            if config.get('slack_enabled', False):
                slack_webhook = config.get('slack_webhook')
                msg = f"⛔ *Early Stopping Triggered*\n" \
                      f"• Stopped at epoch: {epoch}/{config['num_epochs']}\n" \
                      f"• Best Val IoU: {best_val_iou:.4f}\n" \
                      f"• Best Val Loss: {best_val_loss:.4f}\n" \
                      f"• No improvement for {config['early_stopping_patience']} epochs"
                send_slack(msg, slack_webhook)
            
            break
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }, os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch}.pth'))
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, os.path.join(config['save_dir'], 'final_model.pth'))
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best Validation IoU: {best_val_iou:.4f}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print("="*60)
    
    # Send training completion notification
    if config.get('slack_enabled', False):
        slack_webhook = config.get('slack_webhook')
        msg = f"✅ *Training Completed!*\n" \
              f"• Total epochs: {epoch}\n" \
              f"• Best Val IoU: {best_val_iou:.4f}\n" \
              f"• Best Val Loss: {best_val_loss:.4f}\n" \
              f"• Final Train IoU: {history['train_iou'][-1]:.4f}\n" \
              f"• Final Val IoU: {history['val_iou'][-1]:.4f}\n" \
              f"• Model saved in: {config['save_dir']}/\n" \
              f"🎉 Ready for inference!"
        send_slack(msg, slack_webhook)
    
    # Plot training history
    plot_training_history(history)
    print("Training history plot saved to 'training_history.png'")
    
    return model, history


if __name__ == '__main__':
    model, history = main()

