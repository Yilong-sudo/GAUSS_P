"""
Training script for GAUSS model
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import GAUSS
from utils import DataLoader, set_seed, accuracy, EarlyStopping


def train_epoch(model, data, optimizer, device):
    """
    Train for one epoch.
    
    Args:
        model: GAUSS model
        data: PyG Data object
        optimizer: Optimizer
        device: Device to use
        
    Returns:
        loss: Training loss
        acc: Training accuracy
    """
    model.train()
    optimizer.zero_grad()
    
    # Forward pass (GAUSS propagation is always performed)
    out = model(data.x.to(device), data.edge_index.to(device))
    
    # Compute loss on training nodes
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask].to(device))
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Compute accuracy
    acc = accuracy(out[data.train_mask], data.y[data.train_mask].to(device))
    
    return loss.item(), acc


@torch.no_grad()
def evaluate(model, data, device, mask_name='val'):
    """
    Evaluate model.
    
    Args:
        model: GAUSS model
        data: PyG Data object
        device: Device to use
        mask_name: Which mask to use ('val' or 'test')
        
    Returns:
        loss: Validation/test loss
        acc: Validation/test accuracy
    """
    model.eval()
    
    # Forward pass (GAUSS propagation is always performed, same as training)
    out = model(data.x.to(device), data.edge_index.to(device))
    
    # Get mask
    if mask_name == 'val':
        mask = data.val_mask
    else:
        mask = data.test_mask
    
    # Compute loss
    loss = F.cross_entropy(out[mask], data.y[mask].to(device))
    
    # Compute accuracy
    acc = accuracy(out[mask], data.y[mask].to(device))
    
    return loss.item(), acc


def train(args):
    """
    Main training function.
    
    Args:
        args: Arguments
    """
    # Set seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading {args.dataset} dataset...")
    data_loader = DataLoader(root=args.data_root)
    data, num_features, num_classes = data_loader.get_data(
        args.dataset, 
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    
    # Compute homophily ratio
    homophily = data_loader.compute_homophily_ratio(data)
    print(f"Dataset: {args.dataset}")
    print(f"Nodes: {data.num_nodes}, Edges: {data.edge_index.shape[1]}")
    print(f"Features: {num_features}, Classes: {num_classes}")
    print(f"Homophily ratio: {homophily:.4f}")
    
    # Move data to device
    data = data.to(device)
    
    # Create model
    print("\nCreating GAUSS model...")
    model = GAUSS(
        in_features=num_features,
        hidden_features=args.hidden_dim,
        out_features=num_classes,
        num_blocks=args.num_blocks,
        lambda_param=args.lambda_param,
        gamma=args.gamma,
        max_iter=args.max_iter,
        dropout=args.dropout,
        device=device
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    # Training loop
    print("\nStart training...")
    best_val_acc = 0
    best_test_acc = 0
    
    for epoch in tqdm(range(args.epochs), desc="Training"):
        # Train
        train_loss, train_acc = train_epoch(model, data, optimizer, device)
        
        # Evaluate
        val_loss, val_acc = evaluate(model, data, device, 'val')
        test_loss, test_acc = evaluate(model, data, device, 'test')
        
        # Update best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
        
        # Print progress
        if (epoch + 1) % args.print_every == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model
    early_stopping.load_best_model(model)
    
    # Final evaluation
    print("\n" + "="*50)
    print("Final Evaluation")
    print("="*50)
    _, final_val_acc = evaluate(model, data, device, 'val')
    _, final_test_acc = evaluate(model, data, device, 'test')
    
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print(f"Best Test Acc: {best_test_acc:.4f}")
    print(f"Final Val Acc: {final_val_acc:.4f}")
    print(f"Final Test Acc: {final_test_acc:.4f}")
    
    # Save model
    if args.save_model:
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, f'gauss_{args.dataset.lower()}.pt')
        torch.save(model.state_dict(), save_path)
        print(f"\nModel saved to {save_path}")
    
    return final_test_acc


def main():
    parser = argparse.ArgumentParser(description='Train GAUSS model')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='Cora',
                       choices=['Cora', 'CiteSeer', 'PubMed', 'WikiCS', 'Computers', 'Photo',
                               'Chameleon', 'Squirrel', 'Actor', 'Cornell', 'Texas', 'Wisconsin'],
                       help='Dataset name')
    parser.add_argument('--data-root', type=str, default='./data',
                       help='Root directory for datasets')
    parser.add_argument('--train-ratio', type=float, default=0.1,
                       help='Training data ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Validation data ratio')
    
    # Model
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--num-blocks', type=int, default=3,
                       help='Number of blocks k')
    parser.add_argument('--lambda-param', type=float, default=10.0,
                       help='Lambda parameter')
    parser.add_argument('--gamma', type=float, default=1.0,
                       help='Gamma parameter')
    parser.add_argument('--max-iter', type=int, default=20,
                       help='Maximum iterations for optimization')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    
    # Training
    parser.add_argument('--epochs', type=int, default=500,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=200,
                       help='Patience for early stopping')
    parser.add_argument('--print-every', type=int, default=50,
                       help='Print frequency')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--cuda', action='store_true', default=True,
                       help='Use CUDA if available')
    parser.add_argument('--save-model', action='store_true', default=False,
                       help='Save trained model')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                       help='Directory to save models')
    
    args = parser.parse_args()
    
    # Train
    train(args)


if __name__ == '__main__':
    main()
