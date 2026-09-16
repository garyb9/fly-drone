import numpy as np
import torch
from fly_drone.predictive import (
    EmaModule,
    MlpPredictor,
    PredictiveObjective,
    covariance_loss,
    ema_update,
    feature_stats,
    predictive_loss,
    variance_loss,
)


def test_mlp_predictor_shape():
    net = MlpPredictor(context_dim=5, action_dim=3, out_dim=7, hidden=16, depth=2)
    out = net(torch.zeros(4, 5), torch.zeros(4, 3))
    assert out.shape == (4, 7)


def test_ema_update_moves_target_toward_source():
    target = torch.nn.Linear(2, 2)
    source = torch.nn.Linear(2, 2)
    with torch.no_grad():
        target.weight.fill_(0.0)
        source.weight.fill_(2.0)
    ema_update(target, source, momentum=0.75)
    assert torch.allclose(target.weight, torch.full_like(target.weight, 0.5))


def test_ema_module_freezes_the_target_and_updates_it():
    source = torch.nn.Linear(3, 3)
    ema = EmaModule(source)
    assert all(not p.requires_grad for p in ema.target.parameters())
    before = ema.target.weight.detach().clone()
    with torch.no_grad():
        source.weight.add_(1.0)
    ema.update(source, momentum=0.5)
    assert not torch.allclose(before, ema.target.weight)


def test_variance_loss_penalises_collapse():
    rng = torch.Generator().manual_seed(0)
    spread = torch.randn(128, 16, generator=rng)
    collapsed = torch.zeros(128, 16)
    assert float(variance_loss(spread)) < 0.2
    assert float(variance_loss(collapsed)) > 0.5


def test_covariance_loss_penalises_correlated_dimensions():
    rng = torch.Generator().manual_seed(1)
    independent = torch.randn(256, 8, generator=rng)
    base = torch.randn(256, 1, generator=rng)
    correlated = base.repeat(1, 8)
    assert float(covariance_loss(independent)) < float(covariance_loss(correlated))
    assert float(covariance_loss(correlated)) > 0.1


def test_predictive_loss_reduces_to_mse_without_regularisers():
    pred = torch.randn(16, 4)
    pred[:, 0] = 0.0  # guarantee a collapsed dimension so the var term is positive
    target = torch.randn(16, 4)
    total, parts = predictive_loss(pred, target, var_weight=0.0, cov_weight=0.0)
    assert torch.allclose(total, torch.nn.functional.mse_loss(pred, target))
    full, full_parts = predictive_loss(pred, target, var_weight=1.0, cov_weight=0.04)
    assert full > total
    assert set(full_parts) == {"mse", "var", "cov"}
    assert float(full_parts["var"]) >= 0 and float(full_parts["cov"]) >= 0


def test_feature_stats_flags_collapse():
    collapsed = feature_stats(torch.ones(64, 12))
    assert collapsed["std_min"] < 1e-6
    assert collapsed["effective_rank"] < 1.5
    rng = np.random.default_rng(0)
    spread = feature_stats(
        torch.as_tensor(rng.normal(size=(64, 12)), dtype=torch.float32)
    )
    assert spread["std_mean"] > 0.5
    assert spread["effective_rank"] > 3.0


def test_predictive_objective_runs_and_updates_its_ema():
    objective = PredictiveObjective(context_dim=6, action_dim=4, out_dim=6, hidden=32)
    enc = torch.nn.Linear(6, 6)
    objective.attach_ema(enc)
    context = torch.randn(8, 6)
    action = torch.randn(8, 4)
    target = torch.randn(8, 6)
    total, parts = objective.loss(context, action, target)
    assert torch.isfinite(total)
    assert all(torch.isfinite(v) for v in parts.values())
    before = objective.ema.target.weight.detach().clone()
    with torch.no_grad():
        enc.weight.add_(0.5)
    objective.update_target(enc, momentum=0.5)
    assert not torch.allclose(before, objective.ema.target.weight)
