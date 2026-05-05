from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from torch.distributions import Normal

from omoe import OMoE, _build_mlp


class ActorCriticOMoE(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        # Standard rsl_rl knobs (kept for compatibility)
        actor_hidden_dims: List[int] = (512, 256, 128),
        critic_hidden_dims: List[int] = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        # OMoE-specific knobs
        num_experts: int = 6,
        omoe_feature_dim: int = 256,
        omoe_expert_hidden: List[int] = (512, 256),
        omoe_router_hidden: int = 256,
        omoe_head_hidden: List[int] = (256,),
        omoe_top_k: Optional[int] = None,
        router_noise_std: float = 0.0,
        load_balance_coef: float = 0.0,
        **kwargs,
    ):
        if kwargs:
            print(f"[ActorCriticOMoE] Ignoring unused kwargs: {list(kwargs.keys())}")
        super().__init__()

        self.num_actions = num_actions
        self.load_balance_coef = float(load_balance_coef)

        # ------------- actor backbone: OMoE -------------
        self.actor = OMoE(
            in_dim=num_actor_obs,
            out_dim=num_actions,
            num_experts=num_experts,
            feature_dim=omoe_feature_dim,
            expert_hidden=list(omoe_expert_hidden),
            router_hidden=omoe_router_hidden,
            head_hidden=list(omoe_head_hidden),
            top_k=omoe_top_k,
            activation=activation,
            router_noise_std=router_noise_std,
        )

        # ------------- critic: vanilla MLP -------------
        self.critic = _build_mlp(
            in_dim=num_critic_obs,
            hidden_dims=list(critic_hidden_dims),
            out_dim=1,
            activation=activation,
        )

        # ------------- learned action std (state-independent) -------------
        # Same convention as rsl_rl ActorCritic.
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution: Optional[Normal] = None
        # disable arg validation for speed
        Normal.set_default_validate_args(False)

        print("[ActorCriticOMoE] actor:", self.actor)
        print("[ActorCriticOMoE] critic:", self.critic)

    # ------------------------------------------------------------------ #
    # rsl_rl ActorCritic API
    # ------------------------------------------------------------------ #

    @staticmethod
    def init_weights(sequential: nn.Module, scales: List[float]):  # noqa: D401
        """Kept for signature compatibility; we rely on PyTorch defaults."""
        pass

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations: torch.Tensor):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor(observations)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(critic_observations)

    # ------------------------------------------------------------------ #
    # OMoE-specific extras
    # ------------------------------------------------------------------ #

    def auxiliary_loss(self) -> torch.Tensor:
        """Optional load-balancing auxiliary loss. Zero if coef==0."""
        if self.load_balance_coef <= 0.0:
            return torch.zeros((), device=self.std.device)
        return self.load_balance_coef * self.actor.aux_load_balancing_loss()

    def expert_usage(self) -> Optional[torch.Tensor]:
        """For logging: mean alpha per expert from the most recent forward."""
        a = self.actor._last_alpha
        return None if a is None else a.mean(dim=0).detach()