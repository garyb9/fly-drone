"""Predictive auxiliary objective (standalone; additive).

The dense, reward-free signal the v6 sensing spec calls for (§5 A2): predict the
next latent or brain trace from the current one plus the action (the injected
currents), JEPA/RSSM style, with an EMA target. The central failure mode is
representation collapse — a constant embedding makes the loss trivially small —
so this module also ships a VICReg-style variance/covariance term and a collapse
diagnostic (``feature_stats``).

Nothing here is wired into training yet; it is the reusable machinery for the
encoder's auxiliary head. The mapping to the architecture is:

    context = encoder latent (or current DN trace)
    action  = injected currents (8 for v5, 432 for the spatial v6 encoder)
    target  = next DN trace / next latent, usually from an EMA copy of the encoder
"""

import copy

import torch
from torch import nn

# V-JEPA2-style EMA momentum: the target follows the encoder slowly.
EMA_MOMENTUM = 0.996


class MlpPredictor(nn.Module):
    """MLP mapping (context, action) -> predicted latent."""

    def __init__(self, context_dim, action_dim, out_dim=None, hidden=256, depth=2):
        super().__init__()
        out_dim = out_dim or context_dim
        layers = []
        last = context_dim + action_dim
        for _ in range(depth):
            layers += [nn.Linear(last, hidden), nn.SiLU()]
            last = hidden
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, context, action):
        return self.net(torch.cat([context, action], dim=-1))


def variance_loss(x, eps=1e-4):
    """Hinge on each dimension's std: 0 once it reaches 1, positive when collapsed."""
    std = torch.sqrt(x.var(dim=0, correction=0) + eps)
    return torch.mean(torch.relu(1.0 - std))


def covariance_loss(x, eps=1e-8):
    """Off-diagonal covariance penalty: decouples dimensions (VICReg)."""
    n, d = x.shape
    centered = x - x.mean(dim=0)
    cov = (centered.T @ centered) / max(n - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    return (off.square().sum() + eps) / d


def predictive_loss(pred, target, var_weight=1.0, cov_weight=0.04, eps=1e-4):
    """MSE to the target plus anti-collapse terms on the prediction.

    Returns ``(total, parts)`` where ``parts`` has ``mse``, ``var`` and ``cov``.
    Set both weights to 0 for plain MSE.
    """
    mse = torch.nn.functional.mse_loss(pred, target)
    var = variance_loss(pred, eps=eps) if var_weight else pred.new_zeros(())
    cov = covariance_loss(pred) if cov_weight else pred.new_zeros(())
    total = mse + var_weight * var + cov_weight * cov
    return total, {"mse": mse, "var": var, "cov": cov}


@torch.no_grad()
def ema_update(target, source, momentum=EMA_MOMENTUM):
    """target <- momentum*target + (1-momentum)*source, for parameters and buffers."""
    for tp, sp in zip(target.parameters(), source.parameters(), strict=True):
        tp.mul_(momentum).add_(sp, alpha=1.0 - momentum)
    for tb, sb in zip(target.buffers(), source.buffers(), strict=True):
        tb.copy_(sb)


class EmaModule(nn.Module):
    """A frozen EMA copy of ``source``, updated by ``update(momentum)``."""

    def __init__(self, source):
        super().__init__()
        self.target = copy.deepcopy(source)
        for param in self.target.parameters():
            param.requires_grad_(False)

    def update(self, source, momentum=EMA_MOMENTUM):
        ema_update(self.target, source, momentum)


def feature_stats(x):
    """Collapse diagnostic: per-dim std and effective rank (exp of SVD entropy).

    A collapsed embedding has ``std_min`` near 0 and ``effective_rank`` near 1.
    """
    x = torch.as_tensor(x, dtype=torch.float32)
    if x.dim() == 1:
        x = x[None]
    std = x.std(dim=0, correction=0)
    centered = x - x.mean(dim=0)
    if centered.shape[0] < 2:
        return {"std_mean": 0.0, "std_min": 0.0, "effective_rank": 1.0}
    singular = torch.linalg.svdvals(centered)
    if float(singular.sum()) <= 0.0:
        return {"std_mean": 0.0, "std_min": 0.0, "effective_rank": 1.0}
    p = singular / singular.sum()
    entropy = -(p * torch.log(p + 1e-12)).sum()
    return {
        "std_mean": float(std.mean()),
        "std_min": float(std.min()),
        "effective_rank": float(torch.exp(entropy)),
    }


class PredictiveObjective(nn.Module):
    """Bundles the predictor with an optional EMA target encoder and the loss."""

    def __init__(
        self,
        context_dim,
        action_dim,
        out_dim=None,
        hidden=256,
        depth=2,
        var_weight=1.0,
        cov_weight=0.04,
    ):
        super().__init__()
        self.predictor = MlpPredictor(context_dim, action_dim, out_dim, hidden, depth)
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.ema = None

    def attach_ema(self, source):
        """Track an EMA copy of ``source`` (the encoder producing the targets)."""
        self.ema = EmaModule(source)
        return self.ema

    def update_target(self, source, momentum=EMA_MOMENTUM):
        if self.ema is None:
            raise RuntimeError("attach_ema must be called before update_target")
        self.ema.update(source, momentum)

    def forward(self, context, action):
        return self.predictor(context, action)

    def loss(self, context, action, target):
        return predictive_loss(
            self(context, action),
            target,
            var_weight=self.var_weight,
            cov_weight=self.cov_weight,
        )
