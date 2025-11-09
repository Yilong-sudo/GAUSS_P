"""
Experiment script to run multiple experiments
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import train


def run_experiments(args):
    """
    Run multiple experiments with different seeds.
    
    Args:
        args: Arguments
    """
    results = []
    
    print(f"Running {args.num_runs} experiments on {args.dataset}...")
    print("="*70)
    
    for run in range(args.num_runs):
        print(f"\nRun {run+1}/{args.num_runs}")
        print("-"*70)
        
        # Set seed for this run
        args.seed = args.base_seed + run
        
        # Train
        test_acc = train(args)
        results.append(test_acc)
        
        print(f"Run {run+1} Test Accuracy: {test_acc:.4f}")
    
    # Compute statistics
    mean_acc = np.mean(results)
    std_acc = np.std(results)
    
    print("\n" + "="*70)
    print("Final Results")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Number of runs: {args.num_runs}")
    print(f"Mean Test Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Individual results: {[f'{r:.4f}' for r in results]}")
    
    # Save results
    if args.save_results:
        os.makedirs(args.results_dir, exist_ok=True)
        results_file = os.path.join(args.results_dir, f'{args.dataset.lower()}_results.txt')
        
        with open(results_file, 'w') as f:
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Number of runs: {args.num_runs}\n")
            f.write(f"Mean Test Accuracy: {mean_acc:.4f} ± {std_acc:.4f}\n")
            f.write(f"Individual results: {results}\n")
        
        print(f"\nResults saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(description='Run GAUSS experiments')
    
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
    
    # Experiment
    parser.add_argument('--num-runs', type=int, default=10,
                       help='Number of runs with different seeds')
    parser.add_argument('--base-seed', type=int, default=42,
                       help='Base random seed')
    
    # Other
    parser.add_argument('--cuda', action='store_true', default=True,
                       help='Use CUDA if available')
    parser.add_argument('--save-model', action='store_true', default=False,
                       help='Save trained model')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                       help='Directory to save models')
    parser.add_argument('--save-results', action='store_true', default=True,
                       help='Save experiment results')
    parser.add_argument('--results-dir', type=str, default='./results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Run experiments
    run_experiments(args)


if __name__ == '__main__':
    main()
