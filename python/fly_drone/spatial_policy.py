"""SB3 adapter for the v6 spatial encoder (standalone; additive).

Bridges ``spatial_encoder.SpatialEncoderNet`` to stable_baselines3 so the SAC
wiring in ``sac.py`` becomes a small change later: the actor's mean action *is*
the network's per-patch logits (no extra linear head, so the spatial structure
reaches the action directly), with the same ``log_std`` clamp as v5.

Nothing here is wired into ``sac.py``; the running v5 pipeline does not import
it, so the v5 path is untouched. Critic-side wiring (asymmetric, all keys) lands
with the M2 window.
"""

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import Actor, MultiInputPolicy
from torch import nn

from .sac import LOG_STD_MAX, LOG_STD_MIN
from .spatial_encoder import SpatialEncoderNet, flat_dim, flatten


class SpatialFeaturesExtractor(BaseFeaturesExtractor):
    """Eye stack -> the flattened per-patch logits (flat_dim features)."""

    def __init__(self, observation_space):
        super().__init__(observation_space, flat_dim())
        self.net = SpatialEncoderNet()

    def forward(self, observations):
        eyes = observations["eyes"].float() / 255.0
        return flatten(self.net(eyes))


class SpatialActor(Actor):
    """SAC actor whose mean action is the encoder's logits (no extra head)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The feature extractor already emits one logit per current; a linear
        # ``mu`` would only add a square 816x816 head. Identity keeps the spatial
        # structure and the parameter count down.
        self.mu = nn.Identity()

    def get_action_dist_params(self, obs):
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)
        log_std = torch.clamp(self.log_std(latent_pi), LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}


class SpatialSACPolicy(MultiInputPolicy):
    """SAC policy using the spatial encoder for the actor."""

    def make_actor(self, features_extractor=None):
        extractor = SpatialFeaturesExtractor(self.observation_space)
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, extractor)
        return SpatialActor(**actor_kwargs).to(self.device)
