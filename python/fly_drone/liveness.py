"""Liveness bar: a measured, teacher-free "does the connectome drive the body?" gate.

P3 of ``docs/superpowers/specs/2026-09-19-body-agnostic-fidelity-cyborg-design.md``, extended by
P4 of ``docs/superpowers/specs/2026-09-20-ongoing-state-and-faithful-readout.md``. This is an
**additive diagnostic**: it never changes ``roam_eval.ACCEPTANCE`` or the pre-registered A1-A7
criteria. It answers a different question — is the drone moving *because it is being driven by
what it sees*, rather than idling, spinning, or flailing by chance?

Thresholds are pre-registered here before any relay result. A teacher may appear as a context row
in the report but never enters any criterion. "Alive" is deliberately weaker than skilled: the
liveness bar must pass before capability is expanded, and A1-A7 remain the skill bars.

L6-L8 are anchored on published *Drosophila* free-flight statistics (``motion_stats.FLY_REFERENCE``,
sourced in ``docs/references.md``), not on the RL teacher. The bar only gets stricter: L1-L5 are
unchanged and L6-L8 are added. A criterion whose statistic is undefined is reported as
not-evaluable and does not pass.
"""

import numpy as np

from .motion_stats import in_window

LIVENESS = {
    "L1_max_slow_fraction": 0.25,
    "L2_min_coverage": 0.15,
    "L3_causal_metric": "slow_fraction",
    "L5_max_abs_yaw_bias": 0.25,
}
# Non-connectome controls that must not reproduce liveness by chance.
LIVENESS_CONTROLS = ("ghost", "random", "cue_script")
# L6-L8 statistic groups. Move/pause both feed intermittency; ``mean_pause_bout_s`` is reported
# as context but not scored (free-flying flies hover rather than stop, so a missing pause is not
# a failure). Speed shape and inter-saccade interval are context, not criteria.
L6_KEYS = ("bouts_per_min", "mean_move_bout_s")
L7_KEYS = ("saccade_rate_hz", "mean_saccade_amplitude_rad")
L8_KEYS = ("msd_exponent",)
CONTEXT_KEYS = (
    "median_speed",
    "p90_speed",
    "speed_cv",
    "mean_pause_bout_s",
    "median_isi_s",
    "isi_cv",
    "bout_length_cv",
)


def _windowed(summary, keys):
    """Score each statistic against its cited window.

    A statistic that is undefined (``None``) makes the criterion not pass; a criterion whose
    statistics are *all* undefined is reported as not-evaluable (``None``). Undefined never
    silently passes.
    """
    detail, outcomes = {}, []
    for key in keys:
        ok, window = in_window(key, summary.get(key))
        detail[key] = {
            "value": summary.get(key),
            "window": list(window),
            "passed": ok,
        }
        outcomes.append(ok)
    if all(outcome is None for outcome in outcomes):
        return None, detail
    return bool(all(outcome is True for outcome in outcomes)), detail


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
    l6, l6_detail = _windowed(none, L6_KEYS)
    l7, l7_detail = _windowed(none, L7_KEYS)
    l8, l8_detail = _windowed(none, L8_KEYS)

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
        "L6_intermittency": {
            "statistics": l6_detail,
            "passed": l6,
        },
        "L7_saccadic_turning": {
            "statistics": l7_detail,
            "passed": l7,
        },
        "L8_exploration_structure": {
            "statistics": l8_detail,
            "passed": l8,
        },
        "context": {key: none.get(key) for key in CONTEXT_KEYS},
    }
    criteria = {
        f"L{index}": out[key]["passed"]
        for index, key in (
            (1, "L1_mobility"),
            (2, "L2_exploration"),
            (3, "L3_sense_causality"),
            (4, "L4_anti_luck"),
            (5, "L5_non_degenerate"),
            (6, "L6_intermittency"),
            (7, "L7_saccadic_turning"),
            (8, "L8_exploration_structure"),
        )
    }
    out["criteria"] = criteria
    out["not_evaluable"] = [key for key, value in criteria.items() if value is None]
    out["passed"] = bool(all(value is True for value in criteria.values()))
    out["thresholds"] = dict(t)
    return out
