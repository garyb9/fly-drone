# Neuron model, graph format, readouts, determinism

Everything here lives in `crates/brain-core` (pure Rust, no Python/MuJoCo/browser dependency) and
is exposed to Python by `crates/brain-python` (PyO3). The simulator core (`core/{lif,sim,format,rng,roles}.rs`)
is copied from fly-playground; see [`source-provenance.json`](source-provenance.json) for hashes.

Symbols used below:

| Symbol            | Meaning                             | Value                         |
| ----------------- | ----------------------------------- | ----------------------------- |
| `N`               | neurons                             | 166,700                       |
| `E`               | retained directed edges             | 10,520,431                    |
| `Δt`              | neural tick                         | 5 ms (200 Hz)                 |
| `τ_m`             | membrane time constant              | 20 ms                         |
| `λ`               | per-tick leak `exp(−Δt/τ_m)`        | `exp(−0.25) ≈ 0.7788`         |
| `v_th`, `v_reset` | threshold, reset (normalised units) | 1.0, 0.0                      |
| `t_ref`           | refractory period                   | 2 ms → `round(2/5) = 0` ticks |
| `σ`               | per-tick input noise std            | 0.02                          |
| `τ_a`             | activity trace time constant        | 40 ticks = 200 ms             |

## 1. Leaky integrate-and-fire, discrete

Continuous reference (EPFL _Neuronal Dynamics_, ch. 1.3):

```
τ_m dv/dt = −(v − v_rest) + R I(t),     spike and v ← v_reset when v ≥ v_th
```

Take `v_rest = 0`, `R = 1`, and integrate exactly over one tick with the input held constant. The
homogeneous part decays by `λ = e^(−Δt/τ_m)`, so the code folds the input into a per-tick jump:

```
u_i(t)   = I_syn,i(t) + I_inj,i(t) + b_i + σ ξ_i(t),     ξ ~ N(0, 1)
v_i(t+1) = λ v_i(t) + u_i(t)
if v_i(t+1) ≥ v_th:  s_i(t) = 1,  v_i(t+1) = v_reset,  refrac_i = t_ref_ticks
else:                s_i(t) = 0
```

`lif::integrate_one` is the pure per-neuron function. A refractory neuron returns
`(v_reset, no spike, refrac−1)`. With the defaults `t_ref_ticks = 0`, so the only refractoriness
is the reset itself.

### Synaptic propagation and the one-tick delay

When neuron `i` spikes at tick `t`, every outgoing edge `k` adds to the _next_ tick's buffer:

```
I_syn,j(t+1) += w_sim,k · sign_i          for each target j of i
```

`input_cur` and `input_next` are swapped after the tick and the new `input_next` is zeroed. So:

- the update does not depend on neuron order within a tick (double-buffering), and
- every synapse has exactly one tick (5 ms) of delay. A path of `h` hops cannot deliver any
  effect in fewer than `h` ticks. The 40 settling ticks after reset allow up to 40 hops.

### Weights and sign

`graph.bin` stores quantised `i16` weights `q_k` and one scale `w_norm`:

```
w_sim,k = q_k · w_norm
```

`w_norm = 0.003` for the full MaleCNS bundle (`manifest.json`). Edges with fewer than 3 synaptic
contacts are dropped by the data pipeline.

`sign_i` comes from each neuron's dominant predicted transmitter (Dale's principle is assumed):
acetylcholine → `+1` (excitatory); GABA and glutamate → `−1` (inhibitory, the _Drosophila_
GluClα hypothesis). Unresolved and modulatory sources (dopamine, serotonin, octopamine, …) keep
their anatomical edges but **inject zero direct current**, so in this model they are
anatomically present and electrically silent. This is a modelling hypothesis recorded in
`manifest.json`, not measured physiology.

Measured from the canonical `graph.bin` (2026-09-20): **9,189 neurons (5.5 % of the network) have
out-edges whose weights are all exactly zero**, spanning 332,316 edges (3.2 %) — histamine 6,179,
unresolved 2,489, dopamine 391, octopamine 82, serotonin 48. They integrate input and spike; they
transmit nothing. Two consequences are easy to miss. The histaminergic photoreceptor→lamina relay
carries no signal at all, so the P3 optic-flow relay injects into T4/T5 _past_ a dead layer rather
than through it. And because `make_sign_bundle.py` retags only the inhibitory bit in `neurons.bin`
(`graph.bin` is byte-identical across canonical and `malecns-sign-s2`), retagging a zero-weight
neuron's sign is a no-op — the s2 convention could not have acted through histamine, so its null
probe result is not evidence about the sign hypothesis. Restoring modulatory transmission requires
rebuilding `graph.bin` upstream; see
[`superpowers/specs/2026-09-20-ongoing-state-and-faithful-readout.md`](superpowers/specs/2026-09-20-ongoing-state-and-faithful-readout.md) §4.

### Silencing

A silenced neuron is clamped every tick: `v = v_reset`, `s = 0`, no propagation. Its inputs are
discarded. The sensory ablation (`BrainRuntime.silence_sensors`) silences the union of every
visual and looming input cell. Because they never spike, no visual information can enter the
graph.

## 2. Tonic flight-power bias: a worked fixed point

`BrainRuntime` applies a constant bias `b = 0.85` to the DLMn/DVMn indirect flight-power
motoneurons (`power_l`, `power_r`). Ignoring noise and synaptic input, and starting from reset:

```
v(1) = 0.85                         < 1   (no spike)
v(2) = 0.7788 · 0.85 + 0.85 = 1.512 ≥ 1   (spike, reset)
```

So the cell fires every second tick: rate `= 1/(2Δt) = 100 Hz`, and the spike train has mean
`s̄ = 0.5`. The activity trace (§4) converges to `a = 0.5`. That is exactly the dark-condition
readout in [`results/sensory-assay.json`](results/sensory-assay.json):
`power_l = thrust = 0.5031` (the remaining 0.003 is noise and synaptic input). In general a
constant drive `b` produces a free-running cell whenever the fixed point `b/(1−λ)` exceeds
`v_th`, i.e. `b > 1 − λ ≈ 0.221`.

**Two fidelity gaps this exposes.**

_Only one population is ever tonically driven._ `BrainRuntime` biases the DLMn/DVMn set and nothing
else (`brain.py:202-203`); the other 166,698 neurons sit at `b = 0` under `v_th = 1`, so each needs
sustained drive above 0.221 to fire at all. At rest `mean_feature_activity = 0.0048` and 24 of the
2,022 descending/VNC traces exceed 0.05. A brain with no ongoing activity can only be reactive.
Notably the fly's _steering_ motoneurons (`steer_l`/`steer_r`: `b1/b2/b3 MN`, `i1/i2 MN`,
`iii1/iii3 MN`, `hg1-4 MN`) get no tonic drive, although in a flying fly they fire about once per
wingbeat continuously and steering modulates that ongoing train rather than recruiting it from
silence.

**Addressed by C3a (`declared-tonic-v1`).** The tonic drive is now a declared, additive manifest
marker — `{"roles": {"power_l": 0.85, "power_r": 0.85, "steer_l": 0.85, "steer_r": 0.85}}` — read
by `BrainRuntime`, applied on reset, and silenced by the `silence_tonic` ablation. The steering
pair uses the same arithmetic as the power pair (`b = 0.85` → one spike every second tick,
`a → 0.5`, mid-rate so input can push it both ways), so it is declared, not fitted. Build it with
`scripts/make_tonic_bundle.py`; the canonical bundle (no marker) is byte-unchanged and keeps the
historical power-only bias. The per-neuron _excitability_ lever (C3b) is **dropped**: no
comparable published resting baseline exists, and a uniform threshold shift would contradict the
sparse, structured resting activity the connectome is reported to support — see
[`results/liveness/FINDING-2026-09-20-c3b-dropped.md`](results/liveness/FINDING-2026-09-20-c3b-dropped.md).

_There is no refractory period._ `refrac_ticks = round(refrac_ms / dt_ms) = round(2.0 / 5.0) = 0`
(`crates/brain-core/src/core/lif.rs:40-44`), so the only refractoriness is the reset itself. A cell
is silent or fires every tick, with no graded rate code between — the regime is bimodal. Shortening
`dt` would fix it and would change every downstream constant, so this is recorded rather than
acted on.

## 3. Sensory injection

Four input roles are registered at load (see [`sensory-model.md`](sensory-model.md)):
`light_l`, `light_r` (annotated Mi1/Tm3 sets from `sensory-mappings.json`), and
`looming_l`, `looming_r` (every LC4 and LPLC2 cell on that side from `cells.json`). Every tick,
`inject(role, c)` adds the current cue value `c ∈ [0, 2]` to `I_inj` of every cell in the role.
A camera frame sets the cues once and they are held for the next 8 ticks (40 ms).

A bright cue of `c` alone drives each input cell to the fixed point `c/(1−λ) = 4.52c`. Any
`c > 0.221` makes the cell fire periodically; `c ≥ 1` makes it fire every tick (200 Hz).

## 4. Activity trace, readouts and policy features

Each neuron carries an exponential moving average of its spikes:

```
a_i(t+1) = a_i(t) + (s_i(t) − a_i(t)) / τ_a          (τ_a = 40 ticks)
```

This is a first-order low-pass with a 200 ms time constant, so `a_i ∈ [0, 1]` estimates the
per-tick firing probability. Firing rate in Hz is `≈ 200 · a_i`. After a step change the trace
reaches `1 − e^(−1) ≈ 63%` of its new value in 40 ticks (5 camera frames).

- **Readout role** (`readout(role)`): mean of `a_i` over the role's active cells. It drives the
  fly mirror and the viewer meters.
- **Policy features** (`BrainRuntime.features()`): the vector of `a_i` for the 2,022 cells whose
  group is `descending_neuron` or `vnc_motor`. These are the _only_ inputs of the actor. The
  Gym observation space is `Box(0, 1, (2022,))`.

## 5. Connectome storage (CSR)

The file formats are identical to fly-playground: little-endian, `neurons.bin` (16-byte header,
24-byte records: body id, position, group, flags) and `graph.bin` (32-byte header, then
`offsets: u32[N+1]`, `targets: u32[E]`, `weights: i16[E]`). Row `i` holds the targets
`targets[offsets[i]..offsets[i+1]]`. The parsers in `core/format.rs` reject bad magic, short
buffers, overflowing sizes and inconsistent offsets.

Memory, full graph:

| Array                                     | Formula   | Bytes   |
| ----------------------------------------- | --------- | ------- |
| offsets                                   | `4 (N+1)` | 0.67 MB |
| targets                                   | `4 E`     | 42.1 MB |
| weights on disk                           | `2 E`     | 21.0 MB |
| weights in RAM (`f32`, pre-multiplied)    | `4 E`     | 42.1 MB |
| state `v, a, input_cur, input_next, bias` | `5 · 4 N` | 3.3 MB  |

`graph.bin` is therefore ≈ `32 + 4(N+1) + 6E` bytes ≈ 63.8 MB (61 MiB). The native benchmark
measured about 90 MB RSS for the brain alone ([`results/native-benchmark.json`](results/native-benchmark.json)).
Python adds `cells.json` metadata, giving ≈ 490 MB per full runtime.

### Cost per tick

```
work(t) = Θ(N)                  noise sample + integrate every active neuron
        + Θ(Σ_{i fired} fanout_i)   scatter spikes along CSR rows
        + Θ(N)                  activity-trace update and buffer clear
```

The Θ(N) terms are dominated by the 166,700 Gaussian draws. Measured cost is 3.9 ms p50 and
5.1 ms p95 (native, single thread, x86-64). The p95 exceeds the 5 ms real-time budget.
Missed deadlines are reported, and the graph is never reduced to compensate.

## 6. Determinism

- Fixed `Δt` and integer tick counters (`brain.tick`, `plant.step_counter`).
- All noise comes from one `SplitMix64` stream seeded by `reset(seed)`. Gaussians use Box–Muller,
  `ξ = √(−2 ln u₁) · cos(2π u₂)` (one draw per neuron per tick, so the stream position depends
  only on tick count × active neurons). Reset also clears the
  voltages, traces, buffers, interventions, and the vision encoder's previous frame.
- Given the same data files, seed and sequence of `inject/stimulate/silence` calls, `step`
  produces bit-identical activity. `crates/brain-core/tests/golden_trace.rs` checks this against
  a committed trace on the synthetic fixture in `pipeline/out/fixture/`.
  `tests/test_env.py::test_clocks_reset_and_seed_replay` checks it end to end through Python.
- Rendered camera pixels come from MuJoCo/EGL and are not promised to be identical across GPUs
  or drivers. Seeds make runs repeatable on one machine.

## 7. Actor inference (Rust)

`policy.rs` runs the exported decoder with no Python:

```
x₀ = (f − μ) ⊘ σ                      f: 2,022 features; μ, σ from the normaliser
x_{l+1} = tanh(W_l x_l + b_l)         hidden layers (32, 32)
y = clamp(W_L x_L + b_L, −1, 1)       linear output layer
command = y ⊙ [0.4, 0.4, 0.2, 0.8]    [vx, vy, vz (m/s), yaw_rate (rad/s)]
```

Before accepting a policy, loading checks the version, encoder version, dataset SHA-256,
feature-id list, finite parameters, positive scales and layer shapes. Export compares Rust and
PyTorch outputs on 32 random feature vectors and fails if they differ by more than `1e−4`.
