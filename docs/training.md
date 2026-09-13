# Training and evaluation: learning a decoder on a frozen connectome

Code: `python/fly_drone/env.py` (`ConnectomeEnv`), `training.py` (`train`, `export_actor`,
`evaluate`), `calibration.py` (`collect`, `warm_start`), `normalizer.py`.

**What is learned:** only a small MLP decoder, which maps neural activity to motion intent.
**What is frozen:** the connectome (wiring, weights, signs), neuron parameters, tonic bias, the
sensory encoder and the stabiliser. PPO never touches `brain-core` state.

## 1. The decision process

| Element           | Definition                                                                                                                    |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Decision rate     | 25 Hz (one camera frame, 8 neural ticks, 40 physics steps)                                                                    |
| Observation `o_t` | `f_t ∈ [0,1]^2022`, the activity traces of descending + VNC motor neurons at the end of the frame                             |
| Action `a_t`      | `[−1,1]^4`, scaled to `u = a ⊙ [0.4, 0.4, 0.2, 0.8]` = `[v_x, v_y, v_z (m/s), ψ̇ (rad/s)]`                                     |
| Horizon           | 750 frames (30 s) truncation; early termination on crash/out-of-bounds/tilt                                                   |
| Hidden from actor | camera pixels, cues, pose, target position, bearing                                                                           |
| Episode reset     | seed → brain noise stream, target side `±1`, lateral offset `U(0.6, 1.5)` m at `x = 2` m, obstacle mirrored on the other side |

The observation is a deterministic function of sensory history filtered through the graph, so
the problem is partially observable. The 200 ms activity trace (neuron-model §4) provides
short-term memory.

## 2. Reward

With `z` the altitude, `β_t = wrap(atan2(Δy, Δx) − ψ)` the target bearing, `Δ` the horizontal
offset to the target, and `o` the obstacle:

```
r_t = 1 − 4 (z − 1)² − 0.05 ‖a_t‖²                               hover + effort
    + 10 (|β_{t−1}| − |β_t|) + 0.5 cos β_t − 0.2 ‖Δ_xy‖           visual / looming tasks
    − 2 exp(−‖p − o‖² / 0.2)                                     looming task only
    − 20 · 1[terminated]
```

### Why a turning-progress term (potential-based shaping)

The previous reward `2 cos β_t` pays for _being_ aligned. A policy therefore earns most of it
without learning to turn, whenever the target starts near the front. The progress term is
potential-based shaping (Ng, Harada & Russell, 1999) with potential `Φ(s) = −10 |β|`:

```
F(s, s′) = γ Φ(s′) − Φ(s)      ⇒   with γ = 1:   Σ_t F = Φ(s_T) − Φ(s_0) = 10 (|β_0| − |β_T|)
```

Summed over an episode it is exactly 10 × the total bearing reduction, and it cannot be farmed by
oscillating: turning away costs what turning back earns. For `γ = 1` shaping of this form leaves
the optimal policy unchanged. With SB3's `γ = 0.99`, the implementation (no `γ` factor on
`Φ(s′)`) differs from exact potential shaping by `(1−γ) Φ(s′) = −0.1|β|` per step. That is a small
extra pull towards alignment, not a new optimum.

## 3. Policy, value and PPO

Actor and critic are separate MLPs `2022 → 32 → 32 → {4, 1}` with tanh, and they share only the
fixed normaliser:

```
x̂ = (f − μ) ⊘ σ,          μ, σ: calibration statistics (σ clamped ≥ 0.003), else 0 and 1
π_θ(a | o) = N(m_θ(x̂), diag(e^{2 log σ_π}))      state-independent log-std
```

PPO (Schulman et al., 2017) maximises the clipped surrogate with generalised advantage
estimation (Schulman et al., 2016):

```
ρ_t(θ) = π_θ(a_t|o_t) / π_θold(a_t|o_t)
L^CLIP = E_t[ min(ρ_t Â_t, clip(ρ_t, 1−ε, 1+ε) Â_t) ]
δ_t = r_t + γ V(o_{t+1}) − V(o_t),       Â_t = Σ_l (γλ)^l δ_{t+l}
loss = −L^CLIP + c_v (V − R̂)² − c_e H[π]
```

| Hyper-parameter   | Value                                                                                |
| ----------------- | ------------------------------------------------------------------------------------ |
| `γ`, `λ`, `ε`     | 0.99, 0.95, 0.2 (SB3 defaults)                                                       |
| rollout           | `n_steps × n_envs ≈ 1024` transitions                                                |
| minibatch, epochs | 64, 5                                                                                |
| learning rate     | 3e−4 from scratch; 1e−5 after a warm start (so fine-tuning does not undo it)         |
| parallel envs     | `--envs` (default 4); each is a spawned process with its own full brain and renderer |

The deterministic actor exported to Rust is the Gaussian mean, `clamp(m_θ(x̂), −1, 1)`.

## 4. Supervised warm start (behaviour cloning of a turn reflex)

RL from scratch on 2,022 noisy features is sample-hungry, so a decoder is first fitted by
regression. `fly-drone calibrate` renders the target at a known angle `α ~ U(−0.75, 0.75)` rad
(position `[2, 2 tan α, 1]`), holds the frame for 35 camera periods, and records the features.
The label is a proportional yaw command:

```
y = [0, 0, 0, s · clip(1.5 α / 0.8, −1, 1)]       s = teacher scale (--teacher-scale)
```

After action scaling (`× 0.8` rad/s) the teacher is the yaw-rate law `ψ̇ = 1.5 s · α` (for
`|α| < 0.53` rad). Here `α` is the bearing error `β` in the calibration frame, so the teacher
behaves as a first-order closed loop:

```
β̇ = −k β,   k = 1.5 s   ⇒   β(t) = β_0 e^{−k t},   t_½ = ln 2 / k
```

| `s` | `k` (1/s) | time to halve bearing | residual after 10 s |
| --- | --------- | --------------------- | ------------------- |
| 0.4 | 0.6       | 1.16 s                | `e^{−6} ≈ 0.25%`    |
| 0.7 | 1.05      | 0.66 s                | ≈ 0                 |
| 1.0 | 1.5       | 0.46 s                | ≈ 0                 |

So a _perfect_ imitation at `s = 0.4` would pass the 10 s "halve the bearing" test easily.
Failures come from imperfect decoding: 200 ms trace lag, noise, and generalising from static
frames to a moving body, not from a gain that is too weak. Larger `s` buys margin against lag
but risks overshoot once the delay `≈ τ_a` is significant (phase lag `k τ_a` rad).

Fitting: MSE on the actor head only, Adam at `1e−3`, 1,500 steps, minibatch 128, `μ, σ` set from
the calibration features. Then `log σ_π ← −2.5` (exploration std ≈ 0.08), so PPO starts close to
the cloned policy. Simulator bearing is used **only to make labels offline**, never as an input.

## 5. Export and parity

`export_actor` writes `actor.json` (version, `encoder_version`, dataset SHA-256, feature ids,
`μ`, `σ`, layers, action limits), loads it into the Rust runtime, and compares 32 random inputs:
`max |PyTorch − Rust| ≤ 1e−4`, or export fails. Typical error is `≈ 4e−6`, which is f32 round-off.
`fly-drone export <checkpoint.zip> --output actor.json` does the same for any saved checkpoint.

## 6. Evaluation protocol and acceptance

`fly-drone evaluate --policy actor.json --episodes 50 --workers 12` runs these trials in spawned
worker processes:

| Condition | Change                                      | Question answered                                     |
| --------- | ------------------------------------------- | ----------------------------------------------------- |
| `none`    | trained policy, intact brain                | does it steer?                                        |
| `zero`    | features replaced by zeros                  | is the output driven by neural activity at all?       |
| `sensory` | all Mi1/Tm3/LC4/LPLC2 input cells silenced  | does it need vision _through the connectome_?         |
| `shuffle` | features permuted (fixed per seed)          | does it need the anatomical identity of each feature? |
| `hover`   | trained policy, 30 s episodes (seeds 2000+) | does it keep altitude while steering?                 |

Held-out seeds are 1000–1049 (training uses seed 42 + env rank; calibration uses 200+). Each run
lasts 10 s. With `β_0, β_T` wrapped to `[−π, π]`:

```
success = no termination  ∧  |β_T| < 0.5 |β_0|
steering_passed = success_rate(none) ≥ 0.8
ablation_passed = success_rate(none) > success_rate(m)   for m ∈ {zero, sensory, shuffle}
hover_passed   = all 30 s hover runs complete ∧ max settled altitude RMS < 0.15 m
                 (settled RMS = √mean((z − 1)²) over t ≥ 2 s)
```

### How precise is a 50-trial success rate?

Success counts are binomial. For `p̂ = k/n`, the Wilson 95% interval is

```
(p̂ + z²/2n ± z √(p̂(1−p̂)/n + z²/4n²)) / (1 + z²/n),   z = 1.96
```

At `n = 50`, `p̂ = 0.80` gives `[0.67, 0.89]`. The ±11% width means a policy measuring 0.80 might
truly be 0.70. When comparing against an ablation, prefer a clear gap (for example 0.8 vs ≤ 0.5)
over a one-trial difference. The seeds are fixed, so conditions are paired trial-by-trial, and
per-seed outcomes are in the report for McNemar-style comparison.

## 6b. Looming task: an obstacle flies at the drone

`ConnectomeEnv(task="looming")` tests the second visual pathway (dark-area expansion → LC4/LPLC2)
instead of target steering.

**Episode.** The target is parked behind the drone at `[−3.8, 0, 1]`, so it is out of view. The
obstacle (radius `R_o = 0.25` m, near-black) waits at

```
o₀ = [2.5, s·U(0.05, 0.3), 1 + U(−0.1, 0.1)],   s = ±1 (obstacle side, balanced)
```

After a delay `t_d ~ U(0.4, 2.0)` s it moves in a straight line at `v ~ U(0.8, 1.2)` m/s towards the
drone's **start** position `p₀`:

```
o(t) = o₀ + v (t − t_d) · (p₀ − o₀)/‖p₀ − o₀‖,     t ≥ t_d   (updated once per 40 ms frame)
```

If the drone stays still, contact happens when `‖p − o‖ ≤ R_o + r_drone ≈ 0.31` m. That is after
`(‖p₀ − o₀‖ − 0.31)/v ≈ 2.2/v ≈ 1.8–2.7` s of flight. Per-frame steps are `v·0.04 ≤ 4.8` cm,
much smaller than 0.31 m, so the obstacle cannot tunnel through the drone. The obstacle is a MuJoCo
**mocap body**. A static world geom moved through `model.geom_pos` keeps its compile-time
collision bounds and never registers contact (`test_moved_obstacle_registers_contact`).

**Time budget.** The drone can move at most `0.4` m/s laterally and `0.2` m/s vertically.
Clearing `R_o + r_drone ≈ 0.31` m sideways takes `≈ 0.8` s at full command, plus the PID's
position-hold lag. The loom cue grows as `1/d³` (sensory-model §2). With loom gain 150 (encoder v4), LC4/LPLC2 input
cells start firing at ≈ 1.5–2 m, about 1.2–1.7 s before contact at 1 m/s. With the earlier gain of
12 they fired only at ≈ 0.5 m, which left no time and gave 0% avoidance.

**Reward** (looming only; the visual bearing terms are off):

```
r_t = 1 − 4 (z − 1)² − 0.05 ‖a‖² − 0.5 ‖p_xy − p₀,xy‖² − 2 exp(−‖p − o‖² / 0.2) − 20 · 1[terminated]
```

The drift term makes an unnecessary dodge cost a little. The near-miss term and the contact
penalty make a real threat worth dodging. Episodes truncate at 150 frames (6 s).

**Evaluation** (`fly-drone evaluate --task looming`):

```
success          = no contact and no crash within the 6 s episode
balanced success = min(success | obstacle left, success | obstacle right)
avoidance_passed = success ≥ 0.8 ∧ balanced ≥ 0.8
ablation_passed  = trained > zero, sensory, shuffle   (raw and balanced)
```

Each run also records the minimum obstacle distance and the **pre-launch displacement**, the
largest horizontal drift before the obstacle starts moving. A policy that simply always flies
away scores well on survival and would also survive with vision silenced. The ablation
comparison catches that, and a large pre-launch displacement explains it.

## 7. Workflow and run layout

```bash
yarn assay                                               # causal gate (also run by train)
yarn calibrate --trials 256 --output runs/calibration-256.npz
yarn train --task visual --steps 50000 --envs 4 \
     --calibration runs/calibration-256.npz --teacher-scale 0.7 --output runs/visual-v2
yarn evaluate --policy runs/visual-v2/actor.json --episodes 50 --workers 12 \
     --output runs/visual-v2/evaluation.json
yarn train --task looming --steps 30000 --resume runs/visual-v2/ppo.zip --output runs/looming
```

```
runs/<name>/
  sensory-assay.json     gate result at training start
  warm-start.json        calibration fit (not a flight test)
  warm-actor.json, warm-ppo.zip
  checkpoints/rl_model_<steps>_steps.zip
  ppo.zip, actor.json    final PPO model and Rust actor
  training.json          steps, task, seed, envs, teacher scale, export parity
  evaluation.json        per-condition success, per-seed runs, hover, acceptance
```

`runs/` is git-ignored. Accepted reports are copied to [`results/`](results/) and summarised in
[`validation.md`](validation.md).
