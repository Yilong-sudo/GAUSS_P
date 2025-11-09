"""
Quick example script to demonstrate GAUSS usage
"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import GAUSS
from utils import DataLoader, set_seed


def main():
    """
    Simple example of using GAUSS for node classification
    """
    print("="*70)
    print("GAUSS: GrAph-customized Universal Self-Supervised Learning")
    print("Quick Example")
    print("="*70)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load a small dataset for quick demo
    print("\nLoading Cora dataset...")
    data_loader = DataLoader(root='./data')
    data, num_features, num_classes = data_loader.get_data('Cora', seed=42)
    
    print(f"Dataset statistics:")
    print(f"  - Nodes: {data.num_nodes}")
    print(f"  - Edges: {data.edge_index.shape[1]}")
    print(f"  - Features: {num_features}")
    print(f"  - Classes: {num_classes}")
    
    # Compute homophily
    homophily = data_loader.compute_homophily_ratio(data)
    print(f"  - Homophily ratio: {homophily:.4f}")
    
    # Create GAUSS model
    print("\nCreating GAUSS model...")
    model = GAUSS(
        in_features=num_features,
        hidden_features=128,  # Smaller for quick demo
        out_features=num_classes,
        num_blocks=2,
        lambda_param=10.0,
        gamma=1.0,
        max_iter=10,  # Fewer iterations for quick demo
        dropout=0.5,
        device=device
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Move data to device
    data = data.to(device)
    
    print("\nNote: This is just a quick example showing model initialization.")
    print("To train the model, run:")
    print("  python train.py --dataset Cora --epochs 500")
    print("\nOr run full experiments:")
    print("  python run_experiments.py --dataset Cora --num-runs 10")
    
    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)


if __name__ == '__main__':
    main()
