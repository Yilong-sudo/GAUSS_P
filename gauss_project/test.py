"""
Quick test script to verify GAUSS installation and functionality
"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import GAUSS
from utils import DataLoader, set_seed


def test_gauss():
    """
    Quick test to verify GAUSS works correctly
    """
    print("="*70)
    print("GAUSS Quick Test")
    print("="*70)
    
    # Set seed
    set_seed(42)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    
    # Load data
    print("\n✓ Loading Cora dataset...")
    try:
        data_loader = DataLoader(root='./data')
        data, num_features, num_classes = data_loader.get_data('Cora', seed=42)
        print(f"  - Nodes: {data.num_nodes}, Edges: {data.edge_index.shape[1]}")
        print(f"  - Features: {num_features}, Classes: {num_classes}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False
    
    # Create model
    print("\n✓ Creating GAUSS model...")
    try:
        model = GAUSS(
            in_features=num_features,
            hidden_features=64,  # Smaller for quick test
            out_features=num_classes,
            num_blocks=2,
            lambda_param=10.0,
            gamma=1.0,
            max_iter=5,  # Fewer iterations for quick test
            dropout=0.5,
            device=device
        )
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters())}")
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        return False
    
    # Test forward pass
    print("\n✓ Testing forward pass...")
    try:
        data = data.to(device)
        # Only test on a small subset for speed
        test_indices = torch.arange(min(10, data.num_nodes))
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[test_indices] = True
        
        with torch.no_grad():
            out = model(data.x, data.edge_index, train_mode=False)
            print(f"  - Output shape: {out.shape}")
            print(f"  - Expected shape: ({data.num_nodes}, {num_classes})")
    except Exception as e:
        print(f"✗ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*70)
    print("✓ All tests passed! GAUSS is working correctly.")
    print("="*70)
    print("\nYou can now run:")
    print("  python train.py --dataset Cora --epochs 500")
    print("  python run_experiments.py --dataset Cora --num-runs 10")
    
    return True


if __name__ == '__main__':
    success = test_gauss()
    sys.exit(0 if success else 1)
