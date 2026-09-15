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

## 8. Free roam: DAgger warm start, then PPO

One decoder (`2022 → 64 → 64 → 4`, tanh) learns all free-roam skills. Stage 1 imitates the
sight-gated composite teacher (`teacher.py`, [`free-roam.md`](free-roam.md) §4) in closed loop.
Stage 2 is reinforcement learning (PPO) on the free-roam reward. The connectome is frozen in
both. The math is in [`overview/README.md`](overview/README.md) §7.

### 8.1 Gate before training

The teacher itself must pass A3 first. Otherwise the decoder would clone a behaviour that cannot
prove sight.

```bash
fly-drone roam-screen teacher random --ablations none ghost --seeds 20 --seed-base 6000 \
    --seconds 60 --level 3 --workers 6 --output runs/roam/screen.json
```

The report's `near_dodge_rate`, `near_dodge_by_side` and `balanced_dodge_rate` use the
pre-registered scoring (only throws that hit or came within 2 m, as `roam_eval.dodge_rates`).
`threat_dodge_rate` counts every throw and is kept for comparison only. Pass: teacher
`near_dodge_rate` ≥ 0.8 and `balanced_dodge_rate` ≥ 0.8, teacher|ghost `near_dodge_rate` ≤ 0.3.

### 8.2 DAgger

| Iteration | Who flies                                               | Command                                                                                                      |
| --------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| 0         | teacher + action noise (σ 0.2 on forward, lateral, yaw) | `roam-collect --output runs/roam/d0.npz --flights 128`                                                       |
| 1         | student with prob. 0.5                                  | `roam-collect --student runs/roam/fit0/warm-actor.json --beta 0.5 --output runs/roam/d1.npz --seed-base 400` |
| 2         | student with prob. 0.75                                 | `... --student runs/roam/fit1/warm-actor.json --beta 0.25 --seed-base 600`                                   |
| 3         | student only                                            | `... --student runs/roam/fit2/warm-actor.json --beta 0 --seed-base 800`                                      |

The teacher labels every frame (every second frame is stored). After each collection, fit on
**all** data so far and screen the student:

```bash
fly-drone roam-fit runs/roam/d0.npz runs/roam/d1.npz --output runs/roam/fit1 --steps 4000
fly-drone roam-screen runs/roam/fit1/warm-actor.json teacher --ablations none ghost \
    --seeds 10 --workers 6 --output runs/roam/fit1/screen.json
```

`roam-fit` weights each drive equally (inverse frequency), holds out 10% of flights, reports
per-drive MSE and R², sets `log σ_π = −2.5`, and exports a parity-checked `warm-actor.json` with
arena limits plus `warm-ppo.zip` for PPO.

### 8.3 PPO fine-tune

```bash
fly-drone train --task free_roam --resume runs/roam/fit3/warm-ppo.zip --steps 200000 \
    --envs 4 --learning-rate 1e-5 --log-std -1.5 --output runs/roam/ppo
```

Free-roam environments run level 3 with respawn, so a crash costs −20 without ending the
episode (horizon 1,500 frames = 60 s), and `actor.json` is exported with the arena limits
`[0.7, 0.5, 0.3, 0.8]`. Keep the learning rate small so PPO refines the clone instead of erasing
it. `--log-std` reopens exploration, which a warm start sets to −2.5. The step count is a
starting point; watch screened A1–A3 on checkpoints rather than reward alone.

### 8.4 Evaluate

```bash
fly-drone evaluate --task free_roam --policy runs/roam/ppo/actor.json --workers 6 \
    --output runs/roam/ppo/evaluation.json
```

This runs seven brain conditions plus teacher, cue-script and random baselines, and checks the
pre-registered A1–A7 ([`free-roam.md`](free-roam.md) §6). Accepted reports go to `results/`.

## 9. Encoder v5: SAC rounds

Design: [`superpowers/specs/2026-09-14-learned-encoder-sac-design.md`](superpowers/specs/2026-09-14-learned-encoder-sac-design.md);
math and stage list: [`overview/README.md`](overview/README.md) §7.3–7.4. All commands strip ROS
(`env -u PYTHONPATH`), run at ≤ 6 workers, and ask the user before starting. Paths live under
`runs/v5/`.

**Training-only randomisation.** SAC (unlike DAgger/PPO) randomises the wall band's grey level
each episode, `BAND_GREY = (0.0, 0.15)` (`sac.py`), so the encoder cannot key looming off one
fixed band contrast.

**Memory.** The default 100 k replay buffer (`uint8` luma + DN traces, both obs and next_obs since
SB3's `DictReplayBuffer` does not support `optimize_memory_usage`) holds ≈ 5.3 GB; each of the ≤ 6
environment workers adds ≈ 0.9 GB (brain + renderer). Watch `free -g` a few minutes into every run.

**Stop rule.** Checkpoint every 50 k frames. After each round, `sac-validate` computes the
near-dodge rate on 10 validation seeds (9000–9009, disjoint from the 50 evaluation seeds) plus E1
on the same seeds. If a round's `near_dodge_rate` is not higher than the best earlier round's,
stop and report rather than starting the next round.

**Resuming after a crash.** `sac-round ... --init checkpoints/<learner>_<n>_steps.zip` restarts
training from that checkpoint's weights, but `train_round` always starts a fresh replay buffer
and calls `model.learn(..., reset_num_timesteps=True)`, so the run begins at frame 0 again and
trains the full `--frames` you pass — it does not pick up where the crashed run left off. Pass
the _remaining_ frames (`--frames` minus the checkpoint's step count), not the original budget,
or the round will overshoot. Use `sac-export` (below) to validate a checkpoint before deciding.

**Critic-only warm-up.** The first `--actor-warmup` frames of every `sac-round` (default
`ACTOR_WARMUP_FRAMES = 50000`, summed over workers) train only the critic; the actor and the
entropy coefficient stay frozen, so an untrained critic cannot wreck the warm-started encoder or
decoder (spec §4). The value is recorded in `round.json`; `--actor-warmup 0` disables it. Because
frames count from 0 again on a resume, a crash resume repeats the warm-up — pass
`--actor-warmup 0` when the checkpoint's critic is already past it.

**`sac-export`: deployable artifacts from any `.zip`.** Turns a round's own output or a
mid-round `CheckpointCallback` checkpoint into the same artifacts `train_round` writes, reusing
`LearnedEncoder.from_actor(...).save`/`export_decoder` and the version pinning — nothing about
the export path is reimplemented.

```bash
# Validate the 50k checkpoint early, instead of waiting for the whole round:
env -u PYTHONPATH .venv/bin/fly-drone sac-export runs/v5/round1/encoder/checkpoints/encoder_50000_steps.zip --learner encoder --output runs/v5/round1/encoder-50k.pt
env -u PYTHONPATH .venv/bin/fly-drone sac-validate --decoder runs/v5/round0/decoder.json --encoder runs/v5/round1/encoder-50k.pt --output runs/v5/round1/validation-50k.json

# Re-pin the frozen decoder onto a new encoder round's output to measure it before a decoder
# round: same pairing (check_encoder=False) the encoder-learning env already uses.
env -u PYTHONPATH .venv/bin/fly-drone sac-export --repin-decoder runs/v5/round0/decoder.json --encoder runs/v5/round1/encoder/encoder.pt --output runs/v5/round1/decoder-repinned.json
env -u PYTHONPATH .venv/bin/fly-drone sac-validate --decoder runs/v5/round1/decoder-repinned.json --encoder runs/v5/round1/encoder/encoder.pt --output runs/v5/round1/validation-repinned.json
```

`--learner decoder --encoder <pt>` exports and parity-checks (≤ 1e-4) a decoder checkpoint the
same way. `--repin-decoder` copies the source JSON's weights and shapes unchanged, only
rewriting `encoder_version` to the new encoder and recording `repinned_from`; it refuses any
source JSON whose `output` or `layers` it cannot copy verbatim.

### 9.1 Stage 1: DAgger decoder on L2 with v4

Iteration 0 (teacher only):

```bash
env -u PYTHONPATH .venv/bin/fly-drone roam-collect --output runs/v5/dagger/it0.npz --flights 128 --seconds 60 --levels 2 --workers 6 --seed-base 200
env -u PYTHONPATH .venv/bin/fly-drone roam-fit runs/v5/dagger/it0.npz --output runs/v5/dagger/it0
env -u PYTHONPATH .venv/bin/fly-drone roam-screen runs/v5/dagger/it0/warm-actor.json teacher random --level 2 --seeds 10 --seed-base 9000 --workers 6 --output runs/v5/dagger/it0-screen.json
```

Iterations 1–3, student in the loop, for `k, beta` in `(1, 0.5), (2, 0.25), (3, 0.0)`:

```bash
env -u PYTHONPATH .venv/bin/fly-drone roam-collect --output runs/v5/dagger/it$k.npz --student runs/v5/dagger/it$((k-1))/warm-actor.json --beta $beta --flights 128 --seconds 60 --levels 2 --workers 6 --seed-base $((200 + 1000*k))
env -u PYTHONPATH .venv/bin/fly-drone roam-fit runs/v5/dagger/it*.npz --output runs/v5/dagger/it$k
env -u PYTHONPATH .venv/bin/fly-drone roam-screen runs/v5/dagger/it$k/warm-actor.json --level 2 --seeds 10 --seed-base 9000 --workers 6 --output runs/v5/dagger/it$k-screen.json
```

Pick the iteration with the highest beacons/min on the validation seeds (ties: fewer
collisions/min) and record it in `runs/v5/dagger/choice.json`.

### 9.2 Stage 2a: v4 clone, round-0 decoder, v4 E1 baseline

```bash
env -u PYTHONPATH .venv/bin/fly-drone encoder-collect --output runs/v5/clone/data.npz --flights 32 --seconds 60 --workers 6 --seed-base 600
env -u PYTHONPATH .venv/bin/fly-drone encoder-clone runs/v5/clone/data.npz --output runs/v5/clone --steps 20000
env -u PYTHONPATH .venv/bin/fly-drone roam-collect --output runs/v5/round0-data/it0.npz --flights 128 --seconds 60 --levels 2 --workers 6 --seed-base 200 --encoder runs/v5/clone/encoder.pt
env -u PYTHONPATH .venv/bin/fly-drone sac-init-decoder runs/v5/round0-data/it0.npz --encoder runs/v5/clone/encoder.pt --output runs/v5/round0
env -u PYTHONPATH .venv/bin/fly-drone roam-screen runs/v5/round0/decoder.json --encoder runs/v5/clone/encoder.pt --level 2 --seeds 10 --seed-base 9000 --workers 6 --output runs/v5/round0/l2-screen.json
env -u PYTHONPATH .venv/bin/fly-drone sac-validate --decoder runs/v5/round0/decoder.json --encoder runs/v5/clone/encoder.pt --output runs/v5/round0/validation.json
env -u PYTHONPATH .venv/bin/fly-drone encoder-checks --policy runs/v5/dagger/it<chosen>/warm-actor.json --output runs/v5/e1-v4-baseline.json
```

Round-0 data must be collected with `roam-collect --encoder <clone>`: the stage-1 DAgger files were
flown with v4, and the clone produces different DN/motor traces for the same flights, so
`sac-init-decoder` rejects files whose `encoder_version` differs from the encoder it is given (this
changed after the Task 12 gate failure of 2026-09-15). `sac-init-decoder` warm-starts the round-0
decoder on those traces. It fits the pre-tanh mean against `atanh(clip(label, ±0.97))`, so
saturated labels keep a gradient; the per-drive MSE in the report stays in tanh (action) space.
Expect `export_max_error ≤ 1e-4` in `runs/v5/round0/warm-start.json`. The L2 screen is an
engineering sanity check (not acceptance): if beacons/min is below 0.8× the chosen DAgger actor's,
the clone or the tanh head lost the skill and SAC would start from a broken system. `encoder-checks`
with no `--encoder` measures v4 on the 50 evaluation seeds — the "v4 baseline reported alongside" E1.

### 9.3 Rounds 1–3: alternating encoder and decoder SAC

For `k = 1, 2, 3`, with `PREV_DEC`/`PREV_DEC_ZIP` = `runs/v5/round0/decoder.{json,zip}` at k = 1
(otherwise `runs/v5/round$((k-1))/decoder/decoder.{json,zip}`) and `ENC_INIT` =
`runs/v5/clone/encoder.pt` at k = 1 (otherwise `runs/v5/round$((k-1))/encoder/encoder.zip`):

Give each round its own `--seed` (e.g. `42 + 10*k`): resuming a round from a `.zip` init
restores the frozen partner's weights but must still draw fresh episode layouts, and a
repeated seed across rounds would replay the same episodes.

```bash
env -u PYTHONPATH .venv/bin/fly-drone sac-round encoder --output runs/v5/round$k/encoder --frames 350000 --decoder $PREV_DEC --init $ENC_INIT --seed $((42 + 10*k)) --workers 6 > runs/v5/round$k-encoder.log 2>&1
env -u PYTHONPATH .venv/bin/fly-drone sac-round decoder --output runs/v5/round$k/decoder --frames 150000 --encoder runs/v5/round$k/encoder/encoder.pt --init $PREV_DEC_ZIP --seed $((43 + 10*k)) --workers 6 > runs/v5/round$k-decoder.log 2>&1
env -u PYTHONPATH .venv/bin/fly-drone sac-validate --decoder runs/v5/round$k/decoder/decoder.json --encoder runs/v5/round$k/encoder/encoder.pt --output runs/v5/round$k/validation.json
```

Report a table of rounds 0..k (near-dodge, balanced, ghost near-dodge, beacons/min,
collisions/min, E1 AUC, mean metabolic cost from `rollout/metabolic_cost`) and apply the stop rule
above. Take the round with the best validation near-dodge rate (ties: more beacons/min) as final:

```bash
mkdir -p runs/v5/final && cp runs/v5/round<best>/encoder/encoder.pt runs/v5/round<best>/decoder/decoder.json runs/v5/final/
```

### 9.4 E3: brain-bypass control

```bash
env -u PYTHONPATH .venv/bin/fly-drone sac-round bypass --output runs/v5/bypass --frames 450000 --encoder runs/v5/final/encoder.pt --workers 6 > runs/v5/bypass.log 2>&1
env -u PYTHONPATH .venv/bin/fly-drone roam-screen runs/v5/final/decoder.json runs/v5/bypass/bypass.zip --encoder runs/v5/final/encoder.pt --ablations none ghost --seeds 50 --seconds 120 --level 3 --seed-base 1000 --workers 6 --output runs/v5/e3-screen.json
env -u PYTHONPATH .venv/bin/python -c "
import json; from pathlib import Path; from fly_drone.roam_eval import bypass_comparison
r = json.loads(Path('runs/v5/e3-screen.json').read_text())['results']
full = next(v for k, v in r.items() if k.startswith('policy:') and k.endswith('|none'))
byp = next(v for k, v in r.items() if k.startswith('bypass:') and k.endswith('|none'))
out = bypass_comparison(full, byp); Path('runs/v5/e3.json').write_text(json.dumps(out, indent=2)); print(out)"
```

`sac-round bypass` trains a decoder that reads the 8 currents directly, brain bypassed, at the
budget of three decoder rounds (150 k × 3). If `bypass_better` is true in `runs/v5/e3.json`, stop
and report to the user before drawing any stage-4 conclusion.

### 9.5 Stage 4: evaluation, E1–E4, results

```bash
env -u PYTHONPATH .venv/bin/fly-drone evaluate --task free_roam --policy runs/v5/final/decoder.json --encoder runs/v5/final/encoder.pt --episodes 50 --workers 6 --output runs/v5/evaluation.json > runs/v5/evaluation.log 2>&1
env -u PYTHONPATH .venv/bin/fly-drone encoder-checks --policy runs/v5/final/decoder.json --encoder runs/v5/final/encoder.pt --output runs/v5/e1-e2.json
env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_arena.py::test_legacy_room_mjcf_unchanged tests/test_env.py -k "legacy or replay_is_bit_identical"
```

Copy `evaluation.json`, `e1-e2.json`, `e1-v4-baseline.json`, `e3.json` and every round's
`validation.json` to `docs/results/encoder-v5/`, and record A1–A7, E1 (v5 vs v4 AUC), E2 margins,
E3 comparison and E4 in [`validation.md`](validation.md), thresholds taken straight from
`ACCEPTANCE` and `ENCODER_CHECKS`, never rounded in the pass direction. The actor is added to
`docs/results/accepted-policies.json` only if A1–A6, E1, E2 and E4 all pass and the user agrees.
