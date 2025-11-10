"""
Quick setup verification script
Tests all dependencies and basic functionality before training
"""

import sys

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    print("-" * 60)
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'lmdb': 'LMDB',
        'six': 'Six',
        'matplotlib': 'Matplotlib',
        'tqdm': 'tqdm',
        'sklearn': 'scikit-learn'
    }
    
    failed = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name:20s} - OK")
        except ImportError as e:
            print(f"✗ {name:20s} - FAILED: {str(e)}")
            failed.append(name)
    
    print("-" * 60)
    
    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("Please install missing packages:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages installed successfully!")
        return True


def test_cuda():
    """Test CUDA availability"""
    import torch
    
    print("\nTesting CUDA...")
    print("-" * 60)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  Device count: {torch.cuda.device_count()}")
        print(f"  Current device: {torch.cuda.current_device()}")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
        print(f"  Device capability: {torch.cuda.get_device_capability(0)}")
        
        # Test memory
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            print(f"  Total memory: {total_memory / 1024**3:.2f} GB")
        except:
            pass
    else:
        print("⚠️  CUDA not available - will use CPU")
        print("  Training will be slower but will work")
    
    print("-" * 60)


def test_datasets():
    """Test dataset loading"""
    import os
    import lmdb
    
    print("\nTesting datasets...")
    print("-" * 60)
    
    datasets = {
        'DocTamperV1-TrainingSet': 'Training',
        'DocTamperV1-TestingSet': 'Testing/Validation'
    }
    
    all_ok = True
    
    for path, name in datasets.items():
        if not os.path.exists(path):
            print(f"✗ {name:20s} - NOT FOUND at {path}")
            all_ok = False
            continue
        
        try:
            env = lmdb.open(path, readonly=True, lock=False, 
                          readahead=False, meminit=False)
            
            with env.begin(write=False) as txn:
                num_entries = txn.stat()['entries']
                num_samples = num_entries // 2
                
                # Try to load first sample
                img_key = 'image-%09d' % 1
                imgbuf = txn.get(img_key.encode('utf-8'))
                
                lbl_key = 'label-%09d' % 1
                lblbuf = txn.get(lbl_key.encode('utf-8'))
                
                if imgbuf is None or lblbuf is None:
                    print(f"⚠️  {name:20s} - Found but data may be corrupted")
                else:
                    print(f"✓ {name:20s} - OK ({num_samples} samples)")
            
            env.close()
            
        except Exception as e:
            print(f"✗ {name:20s} - ERROR: {str(e)}")
            all_ok = False
    
    print("-" * 60)
    
    if not all_ok:
        print("\n⚠️  Dataset issues detected")
        print("Please ensure dataset folders are present:")
        print("  - DocTamperV1-TrainingSet/")
        print("  - DocTamperV1-TestingSet/")
        return False
    else:
        print("\n✅ All datasets accessible!")
        return True


def test_model():
    """Test model creation"""
    import torch
    from train_unet import UNet
    
    print("\nTesting model creation...")
    print("-" * 60)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = UNet(n_channels=3, n_classes=1, bilinear=True).to(device)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 512, 512).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ Model created successfully")
        print(f"  Parameters: {num_params:,}")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Device: {device}")
        
        print("-" * 60)
        print("\n✅ Model working correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {str(e)}")
        print("-" * 60)
        return False


def test_loss_functions():
    """Test loss functions"""
    import torch
    from train_unet import CombinedLoss, DiceLoss, FocalLoss
    
    print("\nTesting loss functions...")
    print("-" * 60)
    
    try:
        # Create dummy data
        pred = torch.randn(2, 1, 512, 512)
        target = torch.randint(0, 2, (2, 1, 512, 512)).float()
        
        # Test Dice Loss
        dice_loss = DiceLoss()
        dice_val = dice_loss(pred, target)
        print(f"✓ Dice Loss: {dice_val.item():.4f}")
        
        # Test Focal Loss
        focal_loss = FocalLoss()
        focal_val = focal_loss(pred, target)
        print(f"✓ Focal Loss: {focal_val.item():.4f}")
        
        # Test Combined Loss
        combined_loss = CombinedLoss()
        total, bce, dice, focal = combined_loss(pred, target)
        print(f"✓ Combined Loss: {total.item():.4f}")
        print(f"  - BCE: {bce.item():.4f}")
        print(f"  - Dice: {dice.item():.4f}")
        print(f"  - Focal: {focal.item():.4f}")
        
        print("-" * 60)
        print("\n✅ Loss functions working correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Loss function test failed: {str(e)}")
        print("-" * 60)
        return False


def main():
    print("=" * 60)
    print("U-Net Setup Verification")
    print("=" * 60)
    print()
    
    results = []
    
    # Run all tests
    results.append(("Imports", test_imports()))
    test_cuda()  # Informational only
    results.append(("Datasets", test_datasets()))
    results.append(("Model", test_model()))
    results.append(("Loss Functions", test_loss_functions()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20s}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All tests passed! You're ready to train.")
        print("\nNext steps:")
        print("  1. Check dataset: python check_dataset.py")
        print("  2. Start training: python train_unet.py")
        print("\nFor more information, see QUICKSTART.md")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Check dataset paths")
        print("  - Verify Python version (3.8+)")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())

