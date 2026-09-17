"""Two-head joint actor: encoder (eyes -> currents) + decoder (dn -> velocity) in one SAC model.

Plan 07 (M4). The connectome is frozen and non-differentiable, so "joint" means both heads move in
the same gradient step and share one critic, not backprop through the brain. The encoder head sees
only the eye stack (`SpatialFeaturesExtractor`), the decoder head only the DN traces (standardised
with `KeyNormalizer`, the same buffers `set_dn_stats` writes), and the action concatenates the 816
current logits with the 4 velocity means so SB3's single squashed Gaussian samples both. This is
what removes the alternating scheme's frozen-partner staleness.
"""

import torch
from stable_baselines3.sac.policies import Actor
from torch import nn

from .sac import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    WARM_START_LOG_STD,
    AsymmetricSACPolicy,
    KeyNormalizer,
)
from .spatial_encoder import flat_dim
from .spatial_policy import SpatialFeaturesExtractor

VELOCITY_DIM = 4


def joint_action_dim():
    return flat_dim() + VELOCITY_DIM


def split_joint_action(action):
    """Split a joint action into (currents, velocity) along the last axis."""
    return action[..., : flat_dim()], action[..., flat_dim() :]


class JointActor(Actor):
    """SAC actor with an encoder head (eyes) and a decoder head (dn), concatenated."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_encoder = flat_dim()
        n_dn = int(self.observation_space["dn"].shape[0])
        self.velocity_norm = KeyNormalizer(self.observation_space, "dn")
        self.velocity_head = nn.Sequential(
            nn.Linear(n_dn, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, VELOCITY_DIM),
        )
        self.encoder_log_std = nn.Parameter(
            torch.full((self.n_encoder,), WARM_START_LOG_STD)
        )
        self.velocity_log_std = nn.Parameter(
            torch.full((VELOCITY_DIM,), WARM_START_LOG_STD)
        )
        # The base single-path head is replaced by the two heads above.
        self.mu = nn.Identity()
        self.log_std = nn.Identity()

    def get_action_dist_params(self, obs):
        encoder = self.features_extractor(obs)  # (N, 816) current logits
        velocity = self.velocity_head(self.velocity_norm(obs))  # (N, 4) velocity means
        mean = torch.cat([encoder, velocity], dim=1)
        log_std = torch.cat([self.encoder_log_std, self.velocity_log_std])
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX).expand_as(mean)
        return mean, log_std, {}


class JointSACPolicy(AsymmetricSACPolicy):
    """SAC policy whose actor is the two-head joint encoder+decoder."""

    def make_actor(self, features_extractor=None):
        extractor = SpatialFeaturesExtractor(self.observation_space)
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, extractor)
        return JointActor(**actor_kwargs).to(self.device)
