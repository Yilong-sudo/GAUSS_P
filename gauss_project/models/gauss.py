"""
GAUSS: GrAph-customized Universal Self-Supervised Learning
Implementation based on the WWW 2024 paper
GPU-Accelerated Version
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GAUSS(nn.Module):
    """
    GAUSS Model: GrAph-customized Universal Self-Supervised Learning
    
    Fully GPU-accelerated implementation using PyTorch.
    The main idea is to replace global parameters with locally learnable propagation
    by exploiting local attribute distribution.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_blocks: int = 3,
        lambda_param: float = 10.0,
        gamma: float = 1.0,
        max_iter: int = 20,
        dropout: float = 0.5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            in_features: Input feature dimension
            hidden_features: Hidden layer dimension
            out_features: Output feature dimension (number of classes)
            num_blocks: Number of blocks k for k-block diagonal regularization
            lambda_param: Parameter λ to balance Z and B
            gamma: Parameter γ for block diagonal regularization
            max_iter: Maximum number of iterations for optimization
            dropout: Dropout rate
            device: Device to run the model on
        """
        super(GAUSS, self).__init__()
        
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_blocks = num_blocks
        self.lambda_param = lambda_param
        self.gamma = gamma
        self.max_iter = max_iter
        self.dropout = dropout
        self.device = device
        
        # MLP for final classification
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, out_features)
        )
        
        self.to(device)
    
    def construct_ego_network(self, edge_index: torch.Tensor, num_nodes: int, 
                             node_idx: int, max_neighbors: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct ego-network for a given node.
        
        Args:
            edge_index: Edge index of the graph [2, num_edges]
            num_nodes: Total number of nodes
            node_idx: Index of the center node
            max_neighbors: Maximum number of neighbors to include
            
        Returns:
            ego_nodes: Indices of nodes in the ego-network
            ego_mask: Mask indicating which nodes are in the ego-network
        """
        # Find 1-hop neighbors
        neighbors = edge_index[1][edge_index[0] == node_idx].unique()
        
        # Limit the number of neighbors
        if len(neighbors) > max_neighbors:
            neighbors = neighbors[:max_neighbors]
        
        # Combine center node with neighbors
        ego_nodes = torch.cat([torch.tensor([node_idx], device=self.device), neighbors])
        
        return ego_nodes
    
    def compute_laplacian(self, B: torch.Tensor) -> torch.Tensor:
        """
        Compute Laplacian matrix from affinity matrix B (GPU-accelerated).
        
        Args:
            B: Affinity matrix [n, n] on GPU
            
        Returns:
            L: Laplacian matrix [n, n] on GPU
        """
        # Ensure B is symmetric
        B = (B + B.T) / 2
        B = torch.clamp(B, min=0)  # Ensure non-negative
        
        # Compute degree matrix
        D = torch.diag(B.sum(dim=1))
        
        # Laplacian matrix
        L = D - B
        
        return L
    
    def update_W(self, B: torch.Tensor, k: int) -> torch.Tensor:
        """
        Update W by solving Eq. (14) in the paper (GPU-accelerated).
        
        Args:
            B: Current affinity matrix [n, n] on GPU
            k: Number of blocks
            
        Returns:
            W: Updated W matrix [n, n] on GPU
        """
        # Compute Laplacian
        L = self.compute_laplacian(B)
        
        # Compute eigenvalues and eigenvectors on GPU
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        
        # Select k smallest eigenvalues
        indices = torch.argsort(eigenvalues)[:k]
        U = eigenvectors[:, indices]
        
        # W = UU^T
        W = U @ U.T
        
        return W
    
    def update_Z(self, X: torch.Tensor, B: torch.Tensor, lambda_param: float) -> torch.Tensor:
        """
        Update Z by solving Eq. (15) in the paper (GPU-accelerated).
        
        Args:
            X: Node features [n, F] on GPU
            B: Current affinity matrix [n, n] on GPU
            lambda_param: Lambda parameter
            
        Returns:
            Z: Updated Z matrix [n, n] on GPU
        """
        n = X.shape[0]
        
        # For self-expressive learning: min ||X - XZ||^2 + lambda ||Z - B||^2
        # Solution: Z = (XX^T + λI)^(-1) (XX^T + λB) where XX^T is (n, n)
        
        I = torch.eye(n, device=self.device)
        XXT = X @ X.T  # (n, n)
        
        try:
            # Use torch.linalg.solve for GPU acceleration
            Z = torch.linalg.solve(XXT + lambda_param * I, XXT + lambda_param * B)
        except RuntimeError:
            # If singular, use pseudo-inverse
            Z = torch.linalg.pinv(XXT + lambda_param * I) @ (XXT + lambda_param * B)
        
        return Z
    
    def update_B(self, Z: torch.Tensor, W: torch.Tensor, lambda_param: float, gamma: float) -> torch.Tensor:
        """
        Update B by solving Eq. (16) in the paper (GPU-accelerated).
        
        Based on Proposition A.2 in the paper's appendix:
        The solution to min (1/2)||B - A||^2 s.t. diag(B)=0, B>=0, B=B^T
        where A = Z - (γ/λ) * (diag(W)1^T - W)
        
        Args:
            Z: Current Z matrix [n, n] on GPU
            W: Current W matrix [n, n] on GPU
            lambda_param: Lambda parameter
            gamma: Gamma parameter
            
        Returns:
            B: Updated B matrix [n, n] on GPU
        """
        n = Z.shape[0]
        
        # According to Eq. (16) and the optimization derivation in Appendix A,
        # the correct formula is: A = Z - (γ/λ) * (diag(W)1^T - W)
        # Note the NEGATIVE sign (minus, not plus)
        diag_w = torch.diag(W)  # Extract diagonal as vector (n,)
        diag_w_matrix = torch.outer(diag_w, torch.ones(n, device=self.device))  # (n, n) matrix
        A = Z - (gamma / lambda_param) * (diag_w_matrix - W)  # CORRECTED: minus sign
        
        # Remove diagonal: Â = A - diag(diag(A))
        A_hat = A - torch.diag(torch.diag(A))
        
        # Make symmetric and non-negative: B = [(Â + Â^T)/2]_+
        B = (A_hat + A_hat.T) / 2
        B = torch.clamp(B, min=0)
        
        # Set diagonal to 0 (enforce diag(B) = 0 constraint)
        B.fill_diagonal_(0)
        
        return B
    
    def learn_affinity_matrix(self, X: torch.Tensor, ego_nodes: torch.Tensor) -> torch.Tensor:
        """
        Learn affinity matrix for an ego-network using Algorithm 1 from the paper (GPU-accelerated).
        
        Args:
            X: Full node features [num_nodes, num_features] on GPU
            ego_nodes: Indices of nodes in the ego-network
            
        Returns:
            B: Learned affinity matrix [n_ego, n_ego] on GPU
        """
        # Extract features for ego-network (keep on GPU)
        X_ego = X[ego_nodes]  # [n_ego, F]
        n = X_ego.shape[0]
        
        # Initialize on GPU
        Z = torch.eye(n, device=self.device)
        B = torch.eye(n, device=self.device)
        
        # Alternating optimization (all on GPU)
        for iteration in range(self.max_iter):
            # Update W
            W = self.update_W(B, self.num_blocks)
            
            # Update Z
            Z = self.update_Z(X_ego, B, self.lambda_param)
            
            # Update B
            B_new = self.update_B(Z, W, self.lambda_param, self.gamma)
            
            # Check convergence
            if torch.linalg.norm(B_new - B, 'fro') < 1e-4:
                B = B_new
                break
            
            B = B_new
        
        return B  # Already on GPU
    
    def propagate_ego_network(self, X: torch.Tensor, B: torch.Tensor, ego_nodes: torch.Tensor) -> torch.Tensor:
        """
        Propagate features in ego-network using learned affinity matrix.
        
        Args:
            X: Full node features
            B: Affinity matrix
            ego_nodes: Indices of nodes in the ego-network
            
        Returns:
            H: Propagated features for the center node
        """
        # Extract features for ego-network
        X_ego = X[ego_nodes]  # [n_ego, F]
        
        # Normalize B (row normalization)
        row_sum = B.sum(dim=1, keepdim=True)
        row_sum = torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum))
        B_norm = B / row_sum
        
        # Propagate: H = B * X_ego
        H_ego = torch.matmul(B_norm, X_ego)  # [n_ego, F]
        
        # Return representation of center node (first node in ego_nodes)
        return H_ego[0]
    
    def forward(self, X: torch.Tensor, edge_index: torch.Tensor, 
                train_mode: bool = True) -> torch.Tensor:
        """
        Forward pass of GAUSS.
        
        CRITICAL: Both training and testing MUST use the same GAUSS propagation.
        The difference is only whether we update the MLP parameters.
        
        Args:
            X: Node features [num_nodes, num_features]
            edge_index: Edge index [2, num_edges]
            train_mode: Not used anymore - kept for API compatibility
                       GAUSS propagation is ALWAYS performed
            
        Returns:
            out: Node logits after GAUSS propagation and MLP
        """
        num_nodes = X.shape[0]
        
        # ALWAYS perform GAUSS propagation (both training and testing)
        # This ensures train/test consistency
        H_all = []
        
        for node_idx in range(num_nodes):
            # Construct ego-network
            ego_nodes = self.construct_ego_network(edge_index, num_nodes, node_idx)
            
            # Learn affinity matrix
            B = self.learn_affinity_matrix(X, ego_nodes)
            
            # Propagate in ego-network
            h = self.propagate_ego_network(X, B, ego_nodes)
            H_all.append(h)
        
        H = torch.stack(H_all, dim=0)
        
        # Apply MLP
        out = self.mlp(H)
        
        return out
    
    def get_embeddings(self, X: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get node embeddings without classification layer.
        
        Args:
            X: Node features
            edge_index: Edge index
            
        Returns:
            H: Node embeddings
        """
        num_nodes = X.shape[0]
        H_all = []
        
        for node_idx in range(num_nodes):
            ego_nodes = self.construct_ego_network(edge_index, num_nodes, node_idx)
            B = self.learn_affinity_matrix(X, ego_nodes)
            h = self.propagate_ego_network(X, B, ego_nodes)
            H_all.append(h)
        
        H = torch.stack(H_all, dim=0)
        return H
