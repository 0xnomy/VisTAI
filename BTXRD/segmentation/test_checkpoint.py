"""
Quick test to verify checkpoint saving and loading works for Swin-UNet
"""
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from segmentation.models import build_model
from common.utils import save_checkpoint

def test_checkpoint_save_load():
    """Test that we can save and load Swin-UNet checkpoints"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Build model
    print("\n1. Building Swin-UNet model...")
    model = build_model('swin_unet', pretrained=False)  # No pretrained for quick test
    model = model.to(device)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')
    
    # Save checkpoint
    test_dir = Path('outputs/test_checkpoint')
    test_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = test_dir / 'test_checkpoint.pth'
    
    print(f"\n2. Saving checkpoint to: {checkpoint_path}")
    save_checkpoint(
        model, optimizer, scheduler, epoch=1,
        metrics={'val_dice': 0.75, 'val_iou': 0.60},
        save_path=checkpoint_path
    )
    
    # Verify file exists
    if checkpoint_path.exists():
        size_mb = checkpoint_path.stat().st_size / (1024**2)
        print(f"   ✓ Checkpoint saved successfully! ({size_mb:.1f} MB)")
    else:
        print(f"   ✗ ERROR: Checkpoint not saved!")
        return False
    
    # Load checkpoint
    print(f"\n3. Loading checkpoint...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        print(f"   ✓ Checkpoint loaded successfully!")
        print(f"   - Epoch: {checkpoint['epoch']}")
        print(f"   - Metrics: {checkpoint['metrics']}")
        
        # Load state into new model
        model_test = build_model('swin_unet', pretrained=False)
        model_test = model_test.to(device)
        model_test.load_state_dict(checkpoint['model_state_dict'])
        print(f"   ✓ Model state loaded successfully!")
        
    except Exception as e:
        print(f"   ✗ ERROR loading checkpoint: {e}")
        return False
    
    # Test inference
    print(f"\n4. Testing inference with loaded model...")
    model_test.eval()
    with torch.no_grad():
        test_input = torch.randn(1, 3, 224, 224).to(device)
        output = model_test(test_input)
        print(f"   ✓ Inference successful! Output shape: {output.shape}")
    
    # Cleanup
    checkpoint_path.unlink()
    test_dir.rmdir()
    
    print(f"\n{'='*60}")
    print(f"✅ All checkpoint tests passed!")
    print(f"{'='*60}\n")
    return True

if __name__ == '__main__':
    success = test_checkpoint_save_load()
    sys.exit(0 if success else 1)
