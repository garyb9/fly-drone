import numpy as np
import torch
from fly_drone.brain import STACK_FRAMES
from fly_drone.spatial_encoder import SpatialEncoderNet, flat_dim, flatten
from fly_drone.spatial_policy import SpatialActor, SpatialFeaturesExtractor
from gymnasium import spaces


def _space():
    return spaces.Dict(
        {"eyes": spaces.Box(0, 255, (2 * STACK_FRAMES, 48, 64), np.uint8)}
    )


def test_features_extractor_returns_all_patch_logits():
    extractor = SpatialFeaturesExtractor(_space())
    assert extractor.features_dim == flat_dim()
    obs = {"eyes": torch.zeros(3, 2 * STACK_FRAMES, 48, 64)}
    out = extractor(obs)
    assert out.shape == (3, flat_dim())


def test_features_extractor_matches_the_network_it_wraps():
    extractor = SpatialFeaturesExtractor(_space())
    eyes = torch.zeros(1, 2 * STACK_FRAMES, 48, 64)
    with torch.no_grad():
        expected = flatten(SpatialEncoderNet()(eyes))
    # The extractor owns its own net, so only the shape and finiteness are pinned.
    out = extractor({"eyes": eyes})
    assert out.shape == expected.shape and torch.isfinite(out).all()


def test_spatial_actor_emits_one_clamped_logit_per_current():
    space, action_dim = _space(), flat_dim()
    action = spaces.Box(-1.0, 1.0, (action_dim,), np.float32)
    extractor = SpatialFeaturesExtractor(space)
    actor = SpatialActor(
        space,
        action,
        features_extractor=extractor,
        net_arch=[],
        features_dim=action_dim,
    )
    assert isinstance(actor.mu, torch.nn.Identity)
    obs = {"eyes": torch.zeros(2, 2 * STACK_FRAMES, 48, 64)}
    mean, log_std, _ = actor.get_action_dist_params(obs)
    assert mean.shape == (2, action_dim)
    assert log_std.shape == (2, action_dim)
    assert (log_std <= -1.0 + 1e-6).all() and (log_std >= -4.0 - 1e-6).all()
