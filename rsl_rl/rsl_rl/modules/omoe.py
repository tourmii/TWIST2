"""
Orthogonal Mixture of Experts (OMoE) module.

Implements the OMoE architecture from KungFuBot2 / VMS:
  "Learning Versatile Motion Skills for Humanoid Whole-Body Control"
  https://arxiv.org/abs/2509.16638

Key ideas:
  1. M expert networks each map (state, goal) -> feature vector u_i in R^d.
  2. Stack U = [u_1, ..., u_M] in R^{d x M}, then orthogonalize the
     COLUMNS via a differentiable Gram-Schmidt (GS) process. This produces
     features {v_1, ..., v_M} that are mutually orthogonal in R^d, so each
     expert lives in its own direction of the feature space.
  3. A router network produces weights alpha = Router(state, goal) of size M
     (top-k masking + softmax). The action head consumes sum_i alpha_i * v_i.
  4. Because GS is differentiable, the orthogonality constraint is propagated
     through training without an explicit Lagrangian / penalty term.

This module is framework-agnostic. It exposes a torch.nn.Module called
OMoE with a forward(x) -> action_features mapping. A higher-level
ActorCritic wrapper is provided in `omoe_actor_critic.py`.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_mlp(in_dim: int, hidden_dims: List[int], out_dim: int,
               activation: str = "elu") -> nn.Sequential:
    """Build a vanilla MLP. Activation is applied after every hidden layer
    but NOT after the output layer (the output is a raw feature vector).
    """
    act_cls = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }[activation.lower()]

    layers: List[nn.Module] = []
    last = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last, h))
        layers.append(act_cls())
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Differentiable Gram-Schmidt
# ---------------------------------------------------------------------------

def differentiable_gram_schmidt(U: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Apply Gram-Schmidt orthogonalization across the EXPERT dimension.

    Args:
        U: tensor of shape (B, d, M) where d is the per-expert feature dim
           and M is the number of experts. Each column U[..., i] is u_i.
        eps: numerical floor for division.

    Returns:
        V: tensor of shape (B, d, M) with mutually orthogonal columns
           (orthonormal after the L2 normalization step below).

    Notes:
        - The classical (non-modified) GS is unstable for ill-conditioned
          inputs but is fine here because M is small (typically 4-8) and
          the experts' outputs are kept on similar scales by the network's
          activations. We additionally L2-normalize each v_i for stability.
        - Operates in batch via a Python loop over experts (M is small).
    """
    B, d, M = U.shape
    V_cols: List[torch.Tensor] = []

    for i in range(M):
        u_i = U[..., i]                                  # (B, d)
        v_i = u_i
        for j in range(i):
            v_j = V_cols[j]                              # (B, d), already normalized
            # projection coefficient: <v_j, u_i> / <v_j, v_j>
            # Since v_j is unit-norm, denominator is 1, but we keep the
            # generic form for numerical robustness.
            num = (v_j * u_i).sum(dim=-1, keepdim=True)  # (B, 1)
            den = (v_j * v_j).sum(dim=-1, keepdim=True).clamp_min(eps)
            v_i = v_i - (num / den) * v_j
        # normalize so columns are orthonormal (stabilizes downstream sum)
        v_i = v_i / v_i.norm(dim=-1, keepdim=True).clamp_min(eps)
        V_cols.append(v_i)

    V = torch.stack(V_cols, dim=-1)                      # (B, d, M)
    return V


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class TopKRouter(nn.Module):
    """A small MLP that produces a softmax over experts, optionally masked
    so that only the top-k logits remain (others -> 0 weight).
    """

    def __init__(self, in_dim: int, num_experts: int,
                 hidden_dim: int = 256, top_k: Optional[int] = None,
                 noise_std: float = 0.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k if top_k is not None else num_experts
        assert 1 <= self.top_k <= num_experts
        self.noise_std = noise_std

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (alpha, logits)."""
        logits = self.net(x)                             # (B, M)

        if self.training and self.noise_std > 0:
            logits = logits + self.noise_std * torch.randn_like(logits)

        if self.top_k < self.num_experts:
            topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)
            mask = torch.full_like(logits, float("-inf"))
            mask.scatter_(-1, topk_idx, topk_vals)
            logits_masked = mask
        else:
            logits_masked = logits

        alpha = F.softmax(logits_masked, dim=-1)         # (B, M)
        return alpha, logits


# ---------------------------------------------------------------------------
# OMoE block
# ---------------------------------------------------------------------------

class OMoE(nn.Module):
    """Orthogonal Mixture of Experts.

    Args:
        in_dim:        dimension of the concatenated [state, goal] input.
        feature_dim:   per-expert output feature dimension d.
        num_experts:   M.
        expert_hidden: hidden layer sizes for each expert MLP.
        router_hidden: hidden size for the router MLP.
        top_k:         if set, only the top-k experts contribute (sparse mix).
        activation:    activation used inside experts and head.
        out_dim:       output (action) dimension produced by the shared head.
        head_hidden:   hidden sizes for the shared output head f(.) in eq. (8).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_experts: int = 6,
        feature_dim: int = 256,
        expert_hidden: List[int] = (512, 256),
        router_hidden: int = 256,
        head_hidden: List[int] = (256,),
        top_k: Optional[int] = None,
        activation: str = "elu",
        router_noise_std: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_experts = num_experts
        self.feature_dim = feature_dim

        self.experts = nn.ModuleList([
            _build_mlp(in_dim, list(expert_hidden), feature_dim, activation)
            for _ in range(num_experts)
        ])

        self.router = TopKRouter(
            in_dim=in_dim,
            num_experts=num_experts,
            hidden_dim=router_hidden,
            top_k=top_k,
            noise_std=router_noise_std,
        )

        self.head = _build_mlp(feature_dim, list(head_hidden), out_dim, activation)

        self._last_alpha: Optional[torch.Tensor] = None
        self._last_router_logits: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_dim) -> action: (B, out_dim)."""
        # 1) Each expert produces a feature vector u_i.
        u_list = [exp(x) for exp in self.experts]        # M x (B, d)
        U = torch.stack(u_list, dim=-1)                  # (B, d, M)

        # 2) Orthogonalize across experts (columns).
        V = differentiable_gram_schmidt(U)               # (B, d, M)

        # 3) Router produces mixing weights.
        alpha, logits = self.router(x)                   # (B, M)
        self._last_alpha = alpha
        self._last_router_logits = logits

        # 4) Weighted sum: sum_i alpha_i v_i  in R^d.
        # alpha: (B, M), V: (B, d, M) -> mixed: (B, d)
        mixed = torch.einsum("bm,bdm->bd", alpha, V)

        # 5) Shared output head f(.).
        return self.head(mixed)


    @staticmethod
    def load_balancing_loss(alpha: torch.Tensor) -> torch.Tensor:
        mean_usage = alpha.mean(dim=0)                 
        return (mean_usage - mean_usage.mean()).pow(2).mean()

    def aux_load_balancing_loss(self) -> torch.Tensor:
        assert self._last_alpha is not None, "Call forward() first."
        return self.load_balancing_loss(self._last_alpha)