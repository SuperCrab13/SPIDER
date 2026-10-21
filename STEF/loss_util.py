import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
from typing import Optional


class HSICLoss(nn.Module):
    """
    Hilbert-Schmidt Independence Criterion
    """
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma
        
    def _rbf_kernel(self, X):
        """RBF kernel"""
        X_sqnorms = (X ** 2).sum(1)
        K = -2. * torch.mm(X, X.t()) + X_sqnorms.unsqueeze(1) + X_sqnorms.unsqueeze(0)
        K = torch.exp(-K / (2. * self.sigma ** 2))
        return K
        
    def forward(self, z1, z2):
        """
        """
        batch_size = z1.shape[0]
        
        # 中心化矩阵 H = I - 1/n * 1*1^T
        H = torch.eye(batch_size, device=z1.device) - 1.0 / batch_size
        
        # RBF核矩阵
        K = self._rbf_kernel(z1)
        L = self._rbf_kernel(z2)
        
        # HSIC = trace(K H L H) / (n-1)^2
        K_center = H @ K @ H
        L_center = H @ L @ H
        
        hsic = torch.trace(K_center @ L_center) / ((batch_size - 1) ** 2)
        
        return hsic


class RandomFeaturesMMD(nn.Module):
    def __init__(self, input_dim, n_features=32, sigma=1.0):
        super().__init__()
        self.n_features = n_features
        self.sigma = sigma
        self.W = nn.Parameter(torch.randn(input_dim, n_features) / sigma, requires_grad=False)
        self.b = nn.Parameter(torch.rand(n_features) * 2 * torch.pi, requires_grad=False)
    
    def phi(self, x):
        return torch.cos(x @ self.W + self.b) * np.sqrt(2.0 / self.n_features)
    
    def forward(self, x, y):
        fx = self.phi(x).mean(dim=0)
        fy = self.phi(y).mean(dim=0)
        return torch.norm(fx - fy, p=2) ** 2


class Discriminator(nn.Module):
    r"""
    Modality discriminator

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

    def __init__(
            self, in_features: int, out_features: int = 2,
            h_depth: int = 2, h_dim: Optional[int] = 256,
            dropout: float = 0.2
    ) -> None:
        super(Discriminator, self).__init__()
        od = collections.OrderedDict()
        self.h_depth = h_depth
        ptr_dim = in_features
        for layer in range(h_depth):
            od[f"linear_{layer}"] = torch.nn.Linear(ptr_dim, h_dim)
            od[f"act_{layer}"] = torch.nn.LeakyReLU(negative_slope=0.2)
            od[f"dropout_{layer}"] = torch.nn.Dropout(p=dropout)
            ptr_dim = h_dim
        od["pred"] = torch.nn.Linear(ptr_dim, out_features)
        self.od = torch.nn.ModuleDict(od)

    def forward(self, x):
        ptr = x
        for layer in range(self.h_depth):
            ptr = self.od[f"linear_{layer}"](ptr)
            ptr = self.od[f"act_{layer}"](ptr)
            ptr = self.od[f"dropout_{layer}"](ptr)
        pred = self.od['pred'](ptr)
        return pred
