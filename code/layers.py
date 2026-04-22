import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from utils import squash

# --- KAN Layer Integration (using FastKAN) ---
try:
    from fastkan import FastKAN as OriginalFastKAN

    print("Detected 'fastkan' library, will use FastKANWrapper.")

    class FastKANWrapper(nn.Module):
        def __init__(self, width: List[int], **kwargs):
            super().__init__()
            if not isinstance(width, list) or len(width) < 2:
                raise ValueError("FastKANWrapper 'width' must be a list containing at least two elements ([in_features, out_features])")
            layers_hidden = width
            self.fastkan_instance = OriginalFastKAN(
                layers_hidden=layers_hidden,
                num_grids=kwargs.get('num_grids', 8),
                grid_min=kwargs.get('grid_min', -2.0),
                grid_max=kwargs.get('grid_max', 2.0),
                use_base_update=kwargs.get('use_base_update', True),
                spline_weight_init_scale=kwargs.get('spline_weight_init_scale', 0.1)
            )

        def forward(self, x):
            return self.fastkan_instance(x)

    KANLayer = FastKANWrapper

except ImportError:
    print("Warning: 'fastkan' library not found. Using conceptual KANLayer (Linear + GELU) as placeholder.")
    print("To use real FastKAN features, please install 'fastkan' (pip install fast-kan).")

    class KANLayer(nn.Module):
        def __init__(self, width, **kwargs):
            super().__init__()
            if not isinstance(width, list) or len(width) < 2:
                raise ValueError("KANLayer placeholder 'width' must be a list containing at least two elements ([in_features, out_features])")
            in_features = width[0]
            out_features = width[-1]
            self.linear = nn.Linear(in_features, out_features)
            self.activation = nn.GELU()

        def forward(self, x):
            return self.activation(self.linear(x))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class PrimaryCapsuleLayer(nn.Module):
    def __init__(self, in_features: int, num_caps: int, cap_dim: int, use_kan_linear: bool = True,
                 kan_params: Dict = None):
        super().__init__()
        self.num_caps = num_caps
        self.cap_dim = cap_dim
        kan_params = kan_params if kan_params is not None else {}
        if use_kan_linear:
            self.capsule_linear = KANLayer(
                width=[in_features, num_caps * cap_dim], **kan_params
            )
        else:
            self.capsule_linear = nn.Linear(in_features, num_caps * cap_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_caps = self.capsule_linear(x)
        capsules = raw_caps.view(x.size(0), self.num_caps, self.cap_dim)
        squashed_caps = squash(capsules, dim=-1)
        return squashed_caps

class RoutingLayer(nn.Module):
    def __init__(self, num_primary_caps: int, primary_cap_dim: int,
                 num_digit_caps: int, digit_cap_dim: int,
                 routing_iterations: int = 3):
        super().__init__()
        self.num_primary_caps = num_primary_caps
        self.num_digit_caps = num_digit_caps
        self.routing_iterations = routing_iterations

        self.W = nn.Parameter(torch.randn(num_primary_caps, num_digit_caps, primary_cap_dim, digit_cap_dim))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        batch_size = u.size(0)

        u_hat = torch.einsum('bid,ijdo->bijo', u, self.W)
        b_ij = torch.zeros(batch_size, self.num_primary_caps, self.num_digit_caps, device=u.device)

        for iteration in range(self.routing_iterations):
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij.unsqueeze(-1) * u_hat).sum(dim=1)
            v_j = squash(s_j, dim=-1)

            if iteration < self.routing_iterations - 1:
                agreement = (u_hat * v_j.unsqueeze(1)).sum(dim=-1)
                b_ij = b_ij + agreement

        return v_j