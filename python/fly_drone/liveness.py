"""Liveness bar: a measured, teacher-free "does the connectome drive the body?" gate.

P3 of ``docs/superpowers/specs/2026-09-19-body-agnostic-fidelity-cyborg-design.md`` and the plan
``docs/superpowers/plans/2026-09-19-fly-drone-10-liveness-relay.md``. This is an **additive
diagnostic**: it never changes ``roam_eval.ACCEPTANCE`` or the pre-registered A1-A7 criteria. It
answers a different question — is the drone moving *because it is being driven by what it sees*,
rather than idling, spinning, or flailing by chance?

Thresholds are pre-registered here before any relay result. A teacher may appear as a context row
in the report but never enters any criterion. "Alive" is deliberately weaker than skilled: the
liveness bar must pass before capability is expanded, and A1-A7 remain the skill bars.
"""

import numpy as np

LIVENESS = {
    "L1_max_slow_fraction": 0.25,
    "L2_min_coverage": 0.15,
    "L3_causal_metric": "slow_fraction",
    "L5_max_abs_yaw_bias": 0.25,
}
# Non-connectome controls that must not reproduce liveness by chance.
LIVENESS_CONTROLS = ("ghost", "random", "cue_script")


def liveness(results, policy, controls=LIVENESS_CONTROLS):
    """Liveness criteria from ``distill.screen`` results keyed ``"controller|ablation"``.

    Requires the intact, sensory-silenced and ghost conditions for ``policy`` plus the
    ``random``/``cue_script`` baselines. ``ghost`` is a condition of the policy; the baselines are
    controllers flown with ablation ``none``. The teacher is intentionally not a control.
    """
    from .roam_eval import FREE_CELLS, LOSS_OF_CONTROL, _per_seed, paired_bootstrap

    t = LIVENESS
    none = results[f"{policy}|none"]
    silenced = results[f"{policy}|sensory"]
    ghost = results.get(f"{policy}|ghost")

    slow = float(none["slow_fraction"])
    coverage = float(none["mean_visited_cells"]) / FREE_CELLS
    yaw = float(none["mean_abs_yaw_bias"])
    kinds = {}
    for run in none["runs"]:
        for key, value in run.get("collision_kinds", {}).items():
            kinds[key] = kinds.get(key, 0) + value
    loss = {k: kinds.get(k, 0) for k in LOSS_OF_CONTROL}

    metric = t["L3_causal_metric"]
    ci = paired_bootstrap(_per_seed(none, metric), _per_seed(silenced, metric))

    def mobile(summary):
        """A control is "alive" if it both moves and explores — the anti-luck test."""
        if summary is None:
            return None
        return bool(
            summary["slow_fraction"] <= t["L1_max_slow_fraction"]
            and summary["mean_visited_cells"] / FREE_CELLS >= t["L2_min_coverage"]
        )

    control_rows = {"ghost": mobile(ghost)}
    for name in ("random", "cue_script"):
        if name in controls:
            control_rows[name] = mobile(results.get(f"{name}|none"))
    present = {k: v for k, v in control_rows.items() if v is not None}

    l1 = slow <= t["L1_max_slow_fraction"]
    l2 = coverage >= t["L2_min_coverage"]
    # Senses keep it moving: silencing them must make it *more* stationary, CI excludes zero.
    l3 = bool(ci[1] < 0)
    l4 = not any(present.values())
    l5 = bool(yaw <= t["L5_max_abs_yaw_bias"] and not any(loss.values()))

    out = {
        "L1_mobility": {
            "slow_fraction": slow,
            "max": t["L1_max_slow_fraction"],
            "passed": bool(l1),
        },
        "L2_exploration": {
            "coverage": coverage,
            "min": t["L2_min_coverage"],
            "passed": bool(l2),
        },
        "L3_sense_causality": {
            "metric": metric,
            "intact": float(np.mean(_per_seed(none, metric))),
            "silenced": float(np.mean(_per_seed(silenced, metric))),
            "ci_intact_minus_silenced": ci,
            "passed": l3,
        },
        "L4_anti_luck": {"controls": control_rows, "passed": bool(l4)},
        "L5_non_degenerate": {
            "abs_yaw_bias": yaw,
            "max": t["L5_max_abs_yaw_bias"],
            "loss_of_control": loss,
            "passed": bool(l5),
        },
    }
    out["passed"] = bool(l1 and l2 and l3 and l4 and l5)
    out["thresholds"] = dict(t)
    return out
