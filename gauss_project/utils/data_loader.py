"""
Data loading utilities for graph datasets
Supports datasets mentioned in the GAUSS paper
"""

import torch
import numpy as np
import os
from typing import Tuple, Optional
from torch_geometric.datasets import Planetoid, WikiCS, Amazon, WebKB, Actor, WikipediaNetwork
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split


class DataLoader:
    """
    Unified data loader for various graph datasets
    """
    
    HOMOPHILIC_DATASETS = ['Cora', 'CiteSeer', 'PubMed', 'WikiCS', 'Computers', 'Photo']
    HETEROPHILIC_DATASETS = ['Chameleon', 'Squirrel', 'Actor', 'Cornell', 'Texas', 'Wisconsin']
    
    def __init__(self, root: str = './data'):
        """
        Args:
            root: Root directory to store datasets
        """
        self.root = root
        os.makedirs(root, exist_ok=True)
    
    def load_dataset(self, name: str) -> Tuple[Data, int, int]:
        """
        Load a dataset by name.
        
        Args:
            name: Name of the dataset
            
        Returns:
            data: PyG Data object
            num_features: Number of node features
            num_classes: Number of classes
        """
        name_lower = name.lower()
        
        # Homophilic datasets
        if name in ['Cora', 'CiteSeer', 'PubMed']:
            dataset = Planetoid(root=self.root, name=name)
            data = dataset[0]
        
        elif name == 'WikiCS':
            dataset = WikiCS(root=os.path.join(self.root, 'WikiCS'))
            data = dataset[0]
        
        elif name in ['Computers', 'Photo']:
            dataset = Amazon(root=self.root, name=name)
            data = dataset[0]
        
        # Heterophilic datasets
        elif name in ['Cornell', 'Texas', 'Wisconsin']:
            dataset = WebKB(root=self.root, name=name)
            data = dataset[0]
        
        elif name == 'Actor':
            dataset = Actor(root=os.path.join(self.root, 'Actor'))
            data = dataset[0]
        
        elif name in ['Chameleon', 'Squirrel']:
            dataset = WikipediaNetwork(root=self.root, name=name.lower())
            data = dataset[0]
        
        else:
            raise ValueError(f"Unknown dataset: {name}")
        
        num_features = data.num_features
        num_classes = dataset.num_classes
        
        return data, num_features, num_classes
    
    def split_data(
        self, 
        data: Data, 
        dataset_name: str,
        train_ratio: float = 0.1,
        val_ratio: float = 0.1,
        seed: int = 42
    ) -> Data:
        """
        Split data into train/val/test sets.
        
        Args:
            data: PyG Data object
            dataset_name: Name of the dataset
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            seed: Random seed
            
        Returns:
            data: Data object with train_mask, val_mask, test_mask
        """
        num_nodes = data.num_nodes
        
        # Check if masks already exist
        if hasattr(data, 'train_mask') and data.train_mask is not None:
            # Some datasets like Cora already have splits
            if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
                return data
        
        # Create new splits
        np.random.seed(seed)
        indices = np.arange(num_nodes)
        
        if dataset_name in self.HETEROPHILIC_DATASETS:
            # For heterophilic datasets, use 48%/32%/20% split as in paper
            train_ratio = 0.48
            val_ratio = 0.32
        
        # Split indices
        train_size = int(train_ratio * num_nodes)
        val_size = int(val_ratio * num_nodes)
        
        train_idx, temp_idx = train_test_split(
            indices, train_size=train_size, random_state=seed, stratify=data.y.cpu().numpy()
        )
        
        val_idx, test_idx = train_test_split(
            temp_idx, train_size=val_size, random_state=seed, 
            stratify=data.y[temp_idx].cpu().numpy()
        )
        
        # Create masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        
        return data
    
    def get_data(
        self, 
        dataset_name: str,
        train_ratio: float = 0.1,
        val_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple[Data, int, int]:
        """
        Load and split dataset.
        
        Args:
            dataset_name: Name of the dataset
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            seed: Random seed
            
        Returns:
            data: PyG Data object with splits
            num_features: Number of node features
            num_classes: Number of classes
        """
        data, num_features, num_classes = self.load_dataset(dataset_name)
        data = self.split_data(data, dataset_name, train_ratio, val_ratio, seed)
        
        return data, num_features, num_classes
    
    def compute_homophily_ratio(self, data: Data) -> float:
        """
        Compute homophily ratio of the graph.
        
        Args:
            data: PyG Data object
            
        Returns:
            homophily_ratio: Ratio of edges connecting same-class nodes
        """
        edge_index = data.edge_index
        labels = data.y
        
        same_class = 0
        total_edges = edge_index.shape[1]
        
        for i in range(total_edges):
            src, dst = edge_index[0, i], edge_index[1, i]
            if labels[src] == labels[dst]:
                same_class += 1
        
        homophily_ratio = same_class / total_edges if total_edges > 0 else 0
        
        return homophily_ratio
    
    @staticmethod
    def add_noise_to_features(data: Data, noise_ratio: float = 0.1, seed: int = 42) -> Data:
        """
        Add Gaussian noise to node features.
        
        Args:
            data: PyG Data object
            noise_ratio: Ratio of noise to add
            seed: Random seed
            
        Returns:
            data: Data with noisy features
        """
        torch.manual_seed(seed)
        noise = torch.randn_like(data.x) * noise_ratio
        data.x = data.x + noise
        return data
    
    @staticmethod
    def add_noise_to_edges(data: Data, noise_ratio: float = 0.1, seed: int = 42) -> Data:
        """
        Add random edges to the graph.
        
        Args:
            data: PyG Data object
            noise_ratio: Ratio of edges to add
            seed: Random seed
            
        Returns:
            data: Data with noisy edges
        """
        torch.manual_seed(seed)
        num_nodes = data.num_nodes
        num_edges = data.edge_index.shape[1]
        num_noise_edges = int(num_edges * noise_ratio)
        
        # Generate random edges
        noise_edges = torch.randint(0, num_nodes, (2, num_noise_edges))
        
        # Combine with original edges
        data.edge_index = torch.cat([data.edge_index, noise_edges], dim=1)
        
        return data
