# Declared excitability prior — gate (partial: mobility unlocked, sense-causality lost)

**Status:** recorded 2026-09-20. A declared threshold-scale prior moves the model off its
threshold-bound regime and the drone **moves** for the first time (slow fraction 0.627 → 0.148,
mobility bar passes), but the motion is no longer sense-causal and exploration still misses the bar.
Not adopted as the default; the liveness bar is not relaxed. **Non-pre-registered**: 15 seeds.

- Bundle: `data/malecns-excitable` (generated, git-ignored; marker `dynamics = declared-excitable-v1`,
  `threshold_scale = 0.85`), built by
  `scripts/make_dynamics_bundle.py --prior excitable --scale 0.85`
- Adapter: `docs/results/adapter/adapter-excitable.json` (`declared-v1:0b46fe6304ce2c3a`)
- Report: `docs/results/liveness/liveness-check-excitable-smoke.json`
- Command: `fly-drone liveness-check --bundle data/malecns-excitable \
  --adapter docs/results/adapter/adapter-excitable.json --episodes 15 --seconds 120 --level 3 \
  --workers 6 --seed-base 1000`
- Runtime: 1 868 s. Teacher never in the loop.

## The declared prior

`dynamics.excitable_arrays` keeps the Shiu leak and canonical globals (`noise_sigma = 0.02`,
`refrac_ms = 2.0`) and scales the per-neuron threshold by a fixed factor. The factor is **declared,
not fitted**: pick the most excitable scale that keeps the declared adapter battery calibratable and
unsaturated. On the battery, `threshold_scale` 0.80 saturates the power readout (rest 0.86 ≈ max,
so `g_fwd` cannot be calibrated and calibration refuses loudly); 0.85 is the edge
(rest `power` 0.56, max 0.83) and is the value used.

Motivation: the readout audit (`FINDING-2026-09-20.md`) showed the network sits near threshold
(24/2022 descending traces above 0.05 at rest; `power` ≈ 0.32), so weak sensory drive does not
propagate. This is the same regime the fly.ai prior art flags (`external-prior-art.md:71`).

## Liveness result (seeds 1000–1014)

| Criterion | canonical v4 | excitable 0.85 | bar |
| --- | --- | --- | --- |
| L1 mobility (`slow_fraction`) | 0.627 ❌ | **0.148 ✅** | ≤ 0.25 |
| L2 exploration (coverage) | 0.032 ❌ | 0.067 ❌ | ≥ 0.15 |
| L3 sense-causality | [−0.488, −0.265] ✅ | [−0.010, 0.109] ❌ | CI upper < 0 |
| L4 anti-luck | ✅ | ✅ | controls fail L1∧L2 |
| L5 non-degenerate | ✅ | ✅ | yaw ≤ 0.25 |
| **overall** | ❌ | ❌ | |

Intact condition (15 seeds):

| metric | canonical v4 | excitable 0.85 |
| --- | ---: | ---: |
| beacons/min | 0.033 | 0.000 |
| collisions/min | 1.200 | 0.633 |
| near dodge | 0.670 | 0.772 |
| visited cells | 8.27 | 17.20 |
| slow fraction | 0.627 | 0.148 |
| abs yaw bias | 0.011 | 0.015 |

## Interpretation

Excitability is the first lever that produced a large behavioural change: the drone flies and covers
twice the ground. But the liveness bar correctly refuses it:

- **L3 fails.** Silencing the visual field no longer slows the drone (intact 0.148 vs silenced: CI
  [−0.010, 0.109]), so the motion is **endogenous**, not driven by what the eyes see. The declared
  adapter's forward drive is `power − rest_power`; with the network pushed off threshold, that term
  is carried by spontaneous activity rather than by vision.
- **L2 fails.** 6.7 % coverage is movement, not exploration; beacons/min is still 0.00.

This is exactly the failure mode the liveness bar exists to catch: "alive by itself" rather than
"alive because it is driven by its senses". The next step is to combine excitability with the
declared optic-flow relay (which at 0.85 already spreads to 164 descending cells vs 72 at canonical)
and ask whether vision can now *steer* the excited network — or to make the adapter's forward drive
depend on visual laterality rather than raw power above rest.

## Verdict

Keep the canonical v4 bridge. The excitability prior is a promising enabler, not an adopted model;
no 50-seed gate yet. The liveness bar is unchanged.

## Reproduce

```bash
env -u PYTHONPATH .venv/bin/python scripts/make_dynamics_bundle.py \
    --prior excitable --scale 0.85 --output data/malecns-excitable
env -u PYTHONPATH .venv/bin/fly-drone adapter-calibrate --bundle data/malecns-excitable \
    --output docs/results/adapter/adapter-excitable.json
env -u PYTHONPATH .venv/bin/fly-drone liveness-check --bundle data/malecns-excitable \
    --adapter docs/results/adapter/adapter-excitable.json \
    --output docs/results/liveness/liveness-check-excitable-smoke.json \
    --episodes 15 --seconds 120 --level 3 --workers 6 --seed-base 1000
```
