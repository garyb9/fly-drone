# Validation

The status of every acceptance criterion from the
[milestone plan](superpowers/plans/2026-09-13-fly-drone-01-foundations.md), with measured numbers.
Raw reports are in [`results/`](results/). Measurements were taken on an x86-64 desktop
(28 threads, WSL2, Mesa llvmpipe rendering) on 2026-09-13 unless noted.

## Acceptance summary

| Criterion                                                  | Target                                              | Measured                                                               | Status     |
| ---------------------------------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------- | ---------- |
| Physics: equal thrust hovers, motor signs, lag, saturation | pass                                                | `tests/test_physics.py` (7 tests, incl. moved-obstacle contact)        | ✅         |
| PID hover baseline, 30 s                                   | altitude RMS < 0.15 m                               | 0.000 m, all motors 14,475.8 RPM (= analytic ω_h)                      | ✅         |
| Rust golden trace                                          | stable                                              | `golden_trace.rs` (2 tests)                                            | ✅         |
| Python binding, reset isolation, seed replay               | pass                                                | `test_runtime.py`, `test_env.py`                                       | ✅         |
| Export parity Rust ↔ PyTorch                               | ≤ 1e−4                                              | ≈ 4e−6                                                                 | ✅         |
| Sensory causality (synthetic + rendered, with silencing)   | separation, silencing removes it                    | L/R Δ 0.817, silencing Δ 0.587, rendered L/R Δ 0.800, silenced Δ 0.000 | ✅         |
| Cue sign follows target side                               | correct sign, \|Δ\| > 0.5 for \|β\| ∈ 0.1…0.6       | `test_light_cue_sign_follows_target_side`                              | ✅         |
| Visual steering, 50 held-out seeds                         | ≥ 80%, balanced left/right ≥ 80%                    | **100%** (50/50), balanced 100%, final \|β\| 0.024 rad                 | ✅         |
| Ablations degrade steering                                 | trained > zero, sensory, shuffle (raw and balanced) | balanced 1.00 vs 0.00 / 0.00 / 0.12                                    | ✅         |
| Policy hover, 30 s, 5 seeds                                | settled RMS < 0.15 m                                | 0.028 m                                                                | ✅         |
| Looming-obstacle response                                  | avoidance ≥ 80% balanced, above ablations           | training in progress                                                   | ⏳         |
| Viewer: pause, reset, reconnect, interventions, telemetry  | pass                                                | `yarn browser:check`: 9/9                                              | ✅         |
| Real-time loop                                             | ≥ 1× with full graph                                | 1.24× offline (was 0.72×); live server re-check pending                | ✅ offline |

## Visual steering (encoder v3) — accepted

Policy `runs/v3-s04/actor.json`: calibration with 256 rendered trials, supervised warm start
(teacher scale 0.4), then 30,000 PPO steps across 4 environments. The report is
[`results/evaluation-visual-v3.json`](results/evaluation-visual-v3.json). It covers 50 held-out
seeds × 4 conditions plus 5 × 30 s hover runs, in 246 s wall time with 12 workers.

| Condition              | Success  | Left targets | Right targets | Balanced | Mean final \|β\| (rad) |
| ---------------------- | -------- | ------------ | ------------- | -------- | ---------------------- |
| trained                | **100%** | 100%         | 100%          | **1.00** | 0.024                  |
| zero features          | 0%       | 0%           | 0%            | 0.00     | 1.464                  |
| silenced visual inputs | 48%      | 100%         | 0%            | 0.00     | 0.493                  |
| shuffled features      | 16%      | 12.5%        | 19.2%         | 0.12     | 1.245                  |

Wilson 95% interval for 50/50: [0.93, 1.00].

**Why "balanced" matters.** With vision silenced, the decoder still produces a fixed turn from
tonic motor activity, so it "succeeds" whenever the random target happens to be on that side:
48% raw, all on the left. Raw success rate alone would overstate what a blind policy does, so
acceptance requires the minimum of the left and right success rates (≥ 80%) to beat every
ablation.

**Model selection.** 10-seed screens over teacher scales 0.4 / 0.7 / 1.0, every 5k checkpoint.
Scale 0.4 reached 90% from the warm start alone and 100% from 10k PPO steps on. Scales 0.7 and
1.0 stayed at 60% throughout. The stronger teacher gains fit worse (warm-start MSE 0.0056 and
0.0113 vs 0.0018), which is consistent with the lag/overshoot argument in
[`training.md`](training.md) §4.

## Steering v1: the dead-zone finding

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
| 50-seed evaluation, 4 conditions + hover | 412 s wall (v1 render) → 246 s (v3 render), 12 workers                        | evaluation reports                                                           |
| Live server, first pass                  | 0.71× real time, 446 missed frame deadlines (v1 render settings)              | viewer screenshot; re-check pending                                          |

The graph was never reduced to meet a deadline. The brain now dominates frame time (≈ 70%). Most of
each tick is 166,700 Box–Muller Gaussian draws. A faster noise generator would change the
reproducible noise stream and the golden traces, so it is deferred.

## Open items

- Train and evaluate the looming task (obstacle launched at the drone; see [`training.md`](training.md)).
- Live `yarn dev:policy runs/v3-s04/actor.json` check: turning towards Left/Right targets, and the
  deadline count at the new render cost.
