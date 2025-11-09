"""
GAUSS: GrAph-customized Universal Self-Supervised Learning
Implementation based on the WWW 2024 paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import eigh
from typing import Optional, Tuple


class GAUSS(nn.Module):
    """
    GAUSS Model: GrAph-customized Universal Self-Supervised Learning
    
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
    
    def compute_laplacian(self, B: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian matrix from affinity matrix B.
        
        Args:
            B: Affinity matrix
            
        Returns:
            L: Laplacian matrix
        """
        # Ensure B is symmetric
        B = (B + B.T) / 2
        B = np.maximum(B, 0)  # Ensure non-negative
        
        # Compute degree matrix
        D = np.diag(B.sum(axis=1))
        
        # Laplacian matrix
        L = D - B
        
        return L
    
    def update_W(self, B: np.ndarray, k: int) -> np.ndarray:
        """
        Update W by solving Eq. (14) in the paper.
        
        Args:
            B: Current affinity matrix
            k: Number of blocks
            
        Returns:
            W: Updated W matrix
        """
        n = B.shape[0]
        
        # Compute Laplacian
        L = self.compute_laplacian(B)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigh(L)
        
        # Select k smallest eigenvalues
        indices = np.argsort(eigenvalues)[:k]
        U = eigenvectors[:, indices]
        
        # W = UU^T
        W = U @ U.T
        
        return W
    
    def update_Z(self, X: np.ndarray, B: np.ndarray, lambda_param: float) -> np.ndarray:
        """
        Update Z by solving Eq. (15) in the paper.
        
        Args:
            X: Node features [n, F]
            B: Current affinity matrix [n, n]
            lambda_param: Lambda parameter
            
        Returns:
            Z: Updated Z matrix [n, n]
        """
        n = X.shape[0]
        
        # Based on: min ||X - XZ||^2 + lambda ||Z - B||^2
        # Solution: Z = (X^T X + λI)^(-1) (X^T X + λB)
        # But X^T X is (F, F) and B is (n, n), so we use the correct formulation:
        # Z = (I + λI)^(-1) (I + λB) is incorrect
        # Correct: (X^T X)Z = X^T X + λ(Z - B) => (X^T X + λI)Z = X^T X + λB
        # But dimensions don't match. Let's use the direct solution for self-expressive learning:
        # min ||X - XZ||^2 + lambda ||Z - B||^2
        # Taking derivative: -2X^T(X - XZ) + 2λ(Z - B) = 0
        # X^T X - X^T XZ + λZ - λB = 0
        # (X^T X + λI)Z = X^T X + λB
        # This requires Z to be (F, n) not (n, n)
        # 
        # For self-representative learning, the correct form is:
        # Z = (X^T X + λI)^(-1) (X^T X + λB)
        # where both sides should have compatible dimensions
        # Actually, for Xi = Xi * Bi, we have:
        # min_Z ||Xi - Xi*Z||_F^2 s.t. ...
        # The solution is: Z = (Xi^T Xi + λI)^(-1) (Xi^T Xi + λB)
        # Xi is (n, F), Xi^T Xi is (F, F), but we need Z to be (n, n)
        #
        # Correct formulation for self-expressive: min ||X - XZ||^2 where X is (n, F) and Z is (n, n)
        # This means each row of X is represented by a linear combination of all rows
        # Taking derivative w.r.t. Z: -2X^T(X - XZ) + 2λ(Z - B) = 0
        # But X^T X is (F, F) and Z is (n, n) - dimension mismatch!
        #
        # The correct form should be row-wise: for each sample xi, solve xi = X * zi
        # Or use: Z = (XX^T + λI)^(-1) (XX^T + λB) where XX^T is (n, n)
        
        I = np.eye(n)
        XXT = X @ X.T  # (n, n)
        
        try:
            Z = np.linalg.solve(XXT + lambda_param * I, XXT + lambda_param * B)
        except np.linalg.LinAlgError:
            # If singular, use pseudo-inverse
            Z = np.linalg.pinv(XXT + lambda_param * I) @ (XXT + lambda_param * B)
        
        return Z
    
    def update_B(self, Z: np.ndarray, W: np.ndarray, lambda_param: float, gamma: float) -> np.ndarray:
        """
        Update B by solving Eq. (16) in the paper.
        
        Based on Proposition A.2 in the paper's appendix:
        The solution to min (1/2)||B - A||^2 s.t. diag(B)=0, B>=0, B=B^T
        where A = Z - (γ/λ) * (diag(W)1^T - W)
        
        Args:
            Z: Current Z matrix [n, n]
            W: Current W matrix [n, n]
            lambda_param: Lambda parameter
            gamma: Gamma parameter
            
        Returns:
            B: Updated B matrix [n, n]
        """
        n = Z.shape[0]
        
        # According to Eq. (16) and the optimization derivation in Appendix A,
        # the correct formula is: A = Z - (γ/λ) * (diag(W)1^T - W)
        # Note the NEGATIVE sign (minus, not plus)
        diag_w = np.diag(W)  # Extract diagonal as vector (n,)
        diag_w_matrix = np.outer(diag_w, np.ones(n))  # (n, n) matrix
        A = Z - (gamma / lambda_param) * (diag_w_matrix - W)  # CORRECTED: minus sign
        
        # Remove diagonal: Â = A - diag(diag(A))
        A_hat = A - np.diag(np.diag(A))
        
        # Make symmetric and non-negative: B = [(Â + Â^T)/2]_+
        B = (A_hat + A_hat.T) / 2
        B = np.maximum(B, 0)
        
        # Set diagonal to 0 (enforce diag(B) = 0 constraint)
        np.fill_diagonal(B, 0)
        
        return B
    
    def learn_affinity_matrix(self, X: torch.Tensor, ego_nodes: torch.Tensor) -> torch.Tensor:
        """
        Learn affinity matrix for an ego-network using Algorithm 1 from the paper.
        
        Args:
            X: Full node features [num_nodes, num_features]
            ego_nodes: Indices of nodes in the ego-network
            
        Returns:
            B: Learned affinity matrix
        """
        # Extract features for ego-network
        X_ego = X[ego_nodes].cpu().numpy()
        n = X_ego.shape[0]
        
        # Initialize
        Z = np.eye(n)
        B = np.eye(n)
        
        # Alternating optimization
        for iteration in range(self.max_iter):
            # Update W
            W = self.update_W(B, self.num_blocks)
            
            # Update Z
            Z = self.update_Z(X_ego, B, self.lambda_param)
            
            # Update B
            B_new = self.update_B(Z, W, self.lambda_param, self.gamma)
            
            # Check convergence
            if np.linalg.norm(B_new - B, 'fro') < 1e-4:
                B = B_new
                break
            
            B = B_new
        
        return torch.from_numpy(B).float().to(self.device)
    
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
