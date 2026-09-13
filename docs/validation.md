# Validation

The status of every acceptance criterion from the
[milestone plan](superpowers/plans/2026-09-13-fly-drone-01-foundations.md), with measured numbers.
Raw reports are in [`results/`](results/). Measurements were taken on an x86-64 desktop
(28 threads, WSL2, Mesa llvmpipe rendering) on 2026-09-13 unless noted.

## Acceptance summary

| Criterion                                                  | Target                                        | Measured                                                               | Status               |
| ---------------------------------------------------------- | --------------------------------------------- | ---------------------------------------------------------------------- | -------------------- |
| Physics: equal thrust hovers, motor signs, lag, saturation | pass                                          | `tests/test_physics.py` (6 tests)                                      | ✅                   |
| PID hover baseline, 30 s                                   | altitude RMS < 0.15 m                         | 0.000 m, all motors 14,475.8 RPM (= analytic ω_h)                      | ✅                   |
| Rust golden trace                                          | stable                                        | `golden_trace.rs` (2 tests)                                            | ✅                   |
| Python binding, reset isolation, seed replay               | pass                                          | `test_runtime.py`, `test_env.py`                                       | ✅                   |
| Export parity Rust ↔ PyTorch                               | ≤ 1e−4                                        | 3.8e−6                                                                 | ✅                   |
| Sensory causality (synthetic + rendered, with silencing)   | separation, silencing removes it              | L/R Δ 0.817, silencing Δ 0.587, rendered L/R Δ 0.800, silenced Δ 0.000 | ✅                   |
| Cue sign follows target side                               | correct sign, \|Δ\| > 0.5 for \|β\| ∈ 0.1…0.6 | `test_light_cue_sign_follows_target_side`                              | ✅                   |
| Policy hover, 30 s, 5 seeds                                | settled RMS < 0.15 m                          | 0.0005 m (v1 policy)                                                   | ✅                   |
| Visual steering, 50 held-out seeds                         | ≥ 80%                                         | v1: **40%** (dead zone, see below); v3 retraining in progress          | ⏳                   |
| Ablations degrade steering                                 | trained > zero, sensory, shuffle              | v1: 40% vs 0% / 0% / 12%                                               | ✅ (v1), re-check v3 |
| Looming-obstacle response                                  | measurable avoidance vs ablations             | not yet trained                                                        | ⏳                   |
| Viewer: pause, reset, reconnect, interventions, telemetry  | pass                                          | `yarn browser:check`: 9/9                                              | ✅                   |
| Real-time loop                                             | ≥ 1× with full graph                          | 1.24× offline (was 0.72×); live server re-check pending                | ✅ offline           |

## Steering: the dead-zone finding

The first held-out evaluation ([`results/evaluation-conservative-baseline.json`](results/evaluation-conservative-baseline.json),
encoder v1, 0.45 rad eye splay, warm start + 1,024 PPO steps):

| Condition | Success | Collisions | Mean final \|β\| (rad) |
| --------- | ------- | ---------- | ---------------------- |
| trained   | 40%     | 0%         | 0.259                  |
| zero      | 0%      | 0%         | 2.578                  |
| sensory   | 0%      | 0%         | 1.357                  |
| shuffle   | 12%     | 0%         | 1.122                  |

Wilson 95% interval for 20/50: [0.28, 0.54]. All 30 failures stopped turning at
|β| ≈ 0.25–0.36 rad, regardless of the starting bearing. The mean initial |β| was 0.47 rad.
Rendering the target at fixed bearings showed that with a ±19.8° binocular overlap, both light cues
saturate inside ±0.3 rad, and the remaining left−right difference is small and **sign-inverted**
([`sensory-model.md`](sensory-model.md) → "Why the splay is 0.75 rad"). No decoder can read the
target's side from the cues in that zone. Widening the splay to 0.75 rad gives a correctly signed
cue difference of ≈ ±2 from |β| = 0.1 rad. The ablation gaps show that the v1 decoder's partial
steering did come from connectome activity: zeroed features spin the drone, and silencing the
visual inputs removes all success.

A note on the v1 evaluation bug, fixed before these numbers were taken: the initial bearing had not
been wrapped to [−π, π] while the final one was. Both are now wrapped.

## Performance

| Quantity                                 | Value                                                                         | Source                                                                       |
| ---------------------------------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Full-graph load                          | 100 ms (native)                                                               | [`results/native-benchmark.json`](results/native-benchmark.json)             |
| Brain RSS                                | 90 MB native; 461 MB with Python metadata                                     | native benchmark; assay                                                      |
| Neural tick, full graph                  | p50 2.10 ms, p95 2.88 ms (assay, sustained stimulus); 3.9 / 5.1 ms under load | [`results/sensory-assay.json`](results/sensory-assay.json), native benchmark |
| Frame (8 ticks + 40 physics + eyes)      | 32.3 ms: brain 22.6, eyes 6.2, plant 3.4                                      | offline profile, 60 frames                                                   |
| Loop real-time factor                    | **1.24×** (was 0.72× before render fix)                                       | offline profile                                                              |
| Eye rendering, 2 × 64×48                 | 6.2 ms (was 28.2 ms: 640×480 4×MSAA buffer + reflection)                      | render benchmark                                                             |
| PID-only physics                         | 17.7× real time (no brain, no cameras)                                        | [`results/baseline.json`](results/baseline.json)                             |
| 50-seed evaluation, 4 conditions + hover | 412 s wall, 12 workers (v1 render settings)                                   | evaluation report                                                            |
| Live server, first pass                  | 0.71× real time, 446 missed frame deadlines (v1 render settings)              | viewer screenshot; re-check pending                                          |

The graph was never reduced to meet a deadline. The brain now dominates frame time (≈ 70%). Most of
each tick is 166,700 Box–Muller Gaussian draws. A faster noise generator would change the
reproducible noise stream and the golden traces, so it is deferred.

## Open items

- Retrain the visual decoder on encoder v3 and evaluate 50 seeds with ablations.
- Train and evaluate the looming task.
- Live `yarn dev:policy` check: turning toward Left/Right targets, deadline count at the new render
  cost.
