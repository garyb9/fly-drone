# Sensory model: cameras → cues → visual neurons

This is an **engineered adapter**, not a model of the fly retina or optic lobe. It converts two
small RGB images into four bounded currents and injects them into anatomically annotated cell
sets. Everything downstream of injection is the frozen connectome.

Code: `python/fly_drone/plant.py` (camera placement), `crates/brain-core/src/vision.rs` (the
encoder, `Eyes::encode`), and `python/fly_drone/brain.py` (the role definitions).

## 1. Camera geometry

Two MuJoCo cameras are attached to the drone body at `(0.035, 0, 0.008)` m (just ahead of centre,
slightly up):

| Property           | Value                                                                 |
| ------------------ | --------------------------------------------------------------------- |
| Resolution         | 64 × 48 px (`P = 3072` px per eye)                                    |
| Vertical FOV       | `fovy = 75°`                                                          |
| Horizontal FOV     | `fovx = 2·atan(tan(fovy/2) · 64/48) = 2·atan(0.7673 · 1.333) ≈ 91.3°` |
| Yaw splay          | left `+0.75 rad` (+43.0°), right `−0.75 rad` (body `+Y` is left)      |
| Angular resolution | `64 px / 91.3° ≈ 0.70 px/deg`                                         |

In MuJoCo a camera looks along its local `−Z`. `xyaxes = (sin ψ, −cos ψ, 0; 0, 0, 1)` puts local
`+Y` world-up and the view direction at body yaw `ψ`. Horizontal coverage per eye is `ψ ± 45.6°`:

```
left eye   [−2.6°, +88.6°]
right eye  [−88.6°, +2.6°]
binocular overlap  ±2.6°  (5.3° wide), total field ±88.6° (177°)
```

A target straight ahead is seen by both eyes equally. Once it is more than ≈ 2.6° to one side it
leaves the other eye, so the sign of `c_L − c_R` gives the side of the target down to small
bearings. That sign is the signal a steering decoder needs.

### Why the splay is 0.75 rad, not 0.45 rad (measured)

With the earlier 0.45 rad splay the overlap was ±19.8°. The target (≈ 12.6° wide at 2 m) then
sat inside _both_ fields for any bearing within ≈ ±0.3 rad. Both light cues saturated near 2, and
the residual difference was small **and had the wrong sign** (the far eye's view is brighter near
its image edge). A trained decoder stopped turning at ≈ 0.25–0.36 rad in every failed evaluation
trial (see [`results/evaluation-conservative-baseline.json`](results/evaluation-conservative-baseline.json)).
Measured sustained cues, target at 2 m (scratch experiment, `brain.sense` twice per frame so the
transient term is zero):

| bearing (rad) | splay 0.45: `c_L − c_R` | splay 0.75: `c_L − c_R` |
| ------------- | ----------------------- | ----------------------- |
| −0.60         | −1.76                   | −1.72                   |
| −0.30         | **+0.09**               | −2.00                   |
| −0.20         | **+0.22**               | −2.00                   |
| −0.10         | +0.04                   | −1.93                   |
| 0.00          | 0.00                    | 0.00                    |
| +0.10         | −0.03                   | +1.96                   |
| +0.20         | **−0.23**               | +2.00                   |
| +0.30         | **−0.17**               | +2.00                   |
| +0.60         | +1.78                   | +1.79                   |

Bold marks sign errors (a target on the right reading brighter on the left). The wider splay
removes the dead zone. `tests/test_assay.py::test_light_cue_sign_follows_target_side` guards it.
Changing camera geometry changes the encoder identity (`ENCODER_VERSION`
`bright-contrast-400-splay075-noaa-loom150-v4`), so older calibrations and actors are rejected at load.

### Rendering settings (part of the encoder identity)

MuJoCo renders each eye with an offscreen buffer sized exactly 64 × 48, **no multisampling**
(`offsamples = 0`) and **floor reflection disabled**. With the default 640 × 480 buffer at 4× MSAA
and reflections on, two eyes cost ≈ 28 ms per frame on Mesa llvmpipe, which made the loop run
below real time. The chosen settings cost ≈ 5–10 ms. Aliased edges change the target's pixel
count by a few pixels, so these settings are part of `ENCODER_VERSION`.

Rendering uses Mesa's CPU rasteriser by default (`MUJOCO_GL=egl`, llvmpipe), which is portable
and matches CI. On WSL2, `GALLIUM_DRIVER=d3d12` renders on the GPU but was _slower_ for 64 × 48
images (readback dominates) and produces different pixels, so it is not used for training.

### Worked example: how big is the target?

The target is a sphere of radius `R = 0.22` m. At distance `d = 2` m its angular diameter is
`θ = 2·asin(R/d) = 12.6°`, which is ≈ 8.8 px across and covers ≈ `π·4.4² ≈ 61` px ≈ 2% of one eye.
The obstacle (`R = 0.25` m, near-black) at 2 m is ≈ 14.3°.

## 2. Encoder (per eye, per camera frame)

For each pixel with 8-bit sRGB values `(r, g, b)`, the encoder computes Rec. 709 luma on the
_encoded_ values (no gamma linearisation):

```
Y = (0.2126 r + 0.7152 g + 0.0722 b) / 255 ∈ [0, 1]
```

and accumulates two image statistics:

```
B = (1/P) Σ_px 400 · max(0, Y − 0.55)        bright excess (sustained)
D = (1/P) Σ_px 1[Y < 0.18]                    dark area fraction
```

Using the previous frame's `B′` and `D′` from the same encoder (which reset clears):

```
light cue  c_L = clamp(1.5 B + 6 · max(0, B − B′), 0, 2)
loom cue   c_O = clamp(150 · max(0, D − D′),        0, 2)
```

The four currents are `[light_l, light_r, loom_l, loom_r]`.

Interpretation:

- `1.5 B` is a **sustained ON** response to bright image regions. The 0.55 threshold rejects
  mid-grey walls and the floor; the gain 400 makes a ≈2% bright target produce a cue of order 1.
  Example: pixels at `Y ≈ 0.8` covering fraction `p` give `B ≈ 400 · 0.25 · p = 100p`. So `p = 2%`
  → `B ≈ 2` → `c_L` saturates at 2. These constants calibrate the demo scene. They are not
  photoreceptor physiology.
- `6 · max(0, ΔB)` is an **ON-transient**: the cue rises when brightness in that eye increases,
  for example when turning towards the target.
- `150 · max(0, ΔD)` is a crude **looming** proxy: the dark area growing between frames. The gain
  was 12 until encoder v4; see "Choosing the loom gain" below.

### Why dark-area growth approximates looming

For an approaching object of radius `R` at distance `d(t)`, closing at speed `v = −ḋ`, the
small-angle angular size is `θ ≈ 2R/d`. Its image area grows as `θ²`:

```
D ∝ θ² ∝ 1/d²   ⇒   Ḋ ∝ 2 v / d³  = 2 D / τ,   τ = d / v   (time to contact)
```

So `ΔD` per 40 ms frame rises steeply as the obstacle nears (`∝ 1/d³`), much like the
`θ̇`/`τ`-tuned responses attributed to LC4/LPLC2. Confounds: yaw rotation that sweeps a dark
region into view, lighting changes, and a drone that is the one approaching.

### Choosing the loom gain (encoder v4)

An LIF input cell fires once its held cue exceeds ≈ 0.22 (neuron-model §3), so the gain `g` sets
the distance at which LC4/LPLC2 start spiking: they fire when `g · ΔD ≳ 0.22`. Measured per-frame
dark-area growth for a 0.25 m obstacle approaching at 1 m/s (1 px = 1/3072 ≈ 0.00033):

| distance | ΔD per frame (centred / 0.15 m off) | gain for cue 0.25 |
| -------- | ----------------------------------- | ----------------- |
| 2.0 m    | 0.0016 / 0.0007                     | 154 / 384         |
| 1.5 m    | 0.0020 / 0.0026                     | 128 / 96          |
| 1.0 m    | 0.0055 / 0.0081                     | 45 / 31           |
| 0.75 m   | 0.0140 / 0.0163                     | 18 / 15           |

With `g = 12`, cells fired only at ≈ 0.5 m. That is ≈ 0.2 s before contact, too late for a
0.4 m/s lateral command to clear 0.31 m, and the looming PPO run scored 0% avoidance. Feasibility
check: a scripted dodger that triggers when the cue would cross 0.22 at gain `g` survives
**75% at g = 100 and 100% at g = 150, 200, 250** (20 seeds). Noise floor: a steady hover produces
ΔD = 0 exactly (deterministic rendering), and slow yaw produces ≈ 2 px at the 90th percentile
(cue 0.10 at g = 150, below firing). Edge sweeps during turns reach ΔD ≈ 0.28 and saturate the cue
at any gain. **g = 150** is the lowest gain that makes dodging fully feasible. This is sensor
calibration for 64 × 48 px at 25 Hz, not a brain change. Lighting robustness is a known limit
(see [`validation.md`](validation.md)).

### Measured looming response

Obstacle approaching the hovering drone at 1 m/s from 3 m (**encoder v3, loom gain 12**; at gain 150 every cue below saturation is 12.5× larger). The
`escape` readout is the connectome's own giant-fiber-adjacent readout, not a decoder output:

| distance (m) | centred: `loom_l` / `loom_r` | 0.3 m left: `loom_l` / `loom_r` | escape readout (left case) |
| ------------ | ---------------------------- | ------------------------------- | -------------------------- |
| 2.64         | 0.008 / 0.008                | 0.008 / 0.000                   | 0.00                       |
| 1.68         | 0.016 / 0.016                | 0.012 / 0.000                   | 0.00                       |
| 1.04         | 0.066 / 0.066                | 0.074 / 0.000                   | 0.00                       |
| 0.72         | 0.164 / 0.168                | 0.133 / 0.000                   | 0.00                       |
| 0.40–0.50    | 0.000 / 0.000 (fills view)   | 0.691 / 0.000                   | **0.55**                   |

The cues are lateralised: an off-centre obstacle drives only its own side. They grow steeply as
the obstacle nears. They drop to zero once the obstacle fills the image, because `D` saturates
and `ΔD → 0`. The escape readout rises from 0 to ≈ 0.3–0.55 through the anatomical
LC4/LPLC2 → descending pathway, before any learned decoding.

## 3. Injection into the connectome

| Cue                  | Target cells                                                  | Source                                          |
| -------------------- | ------------------------------------------------------------- | ----------------------------------------------- |
| `light_l`, `light_r` | Mi1 / Tm3 cells annotated by root side (1,903 / 1,924 cells)  | `data/malecns/sensory-mappings.json` → `inputs` |
| `loom_l`, `loom_r`   | every `LC4` and `LPLC2` cell with that side (165 / 146 cells) | `cells.json` type/side                          |

Timing, per camera frame (25 Hz):

```
cues ← encode(render(eye_l), render(eye_r))
repeat 8 times (40 ms):
    for each role: I_inj,i += cue for every cell i in the role
    brain.step(1)
    plant.advance(command)      # 5 physics substeps each
```

A cue is therefore a 40 ms current pulse, re-sampled every frame. See
[`neuron-model.md`](neuron-model.md) §3 for how a held current of `c` becomes a firing rate.

## 4. Causal assay (the gate before RL)

`fly-drone assay` (`python/fly_drone/assay.py`) checks that visual information actually
propagates through the frozen graph to the policy features:

| Check                         | Condition                                             | Threshold |
| ----------------------------- | ----------------------------------------------------- | --------- |
| Synthetic left vs right       | max \|features_left − features_right\| over 200 ticks | `> 1e−4`  |
| Silencing effect              | max \|left − left_silenced\|                          | `> 1e−4`  |
| Silenced equals dark          | max \|left_silenced − dark\|                          | `< 1e−6`  |
| Rendered target left vs right | same, with MuJoCo-rendered frames                     | `> 0.01`  |
| Rendered silenced             | left_silenced vs right_silenced                       | `< 1e−6`  |

`train` refuses to start if any check fails. Passing shows only that software signals
propagate. It does not show behaviour or biological validity. Current values are in
[`results/sensory-assay.json`](results/sensory-assay.json).

## 5. Known limitations

- There is no optical flow, no colour opponency, no photoreceptor adaptation, and no
  per-ommatidium sampling. Two image-wide scalars per eye discard all spatial layout within an
  eye.
- `D` responds to _any_ dark-area increase, including rotation and self-motion.
- Thresholds (0.55, 0.18) and gains (400, 1.5, 6, 12) are tuned to this scene's lighting and
  camera range.
- The cues saturate at 2, so a very bright or very close stimulus loses gradation.
- Cells are driven by uniform injected current. Real Mi1/Tm3 receive retinotopic,
  photoreceptor-derived input.

## 6. Encoder v5 (learned, free roam only)

Design: [`superpowers/specs/2026-09-14-learned-encoder-sac-design.md`](superpowers/specs/2026-09-14-learned-encoder-sac-design.md).
Code: `python/fly_drone/encoder.py` (network, save/load, clone), `python/fly_drone/brain.py`
(`FrameStack`, channel roles, encoder identity). Free roam trains and flies this encoder; every
legacy task (visual, looming, approach, track, steer_dodge, escape) stays on encoder v4 above.

**Input.** Each eye's Rec. 709 luma (as in §2, `Y ∈ [0, 1]` packed as `uint8`), the last 3
frames per eye, stacked into a `6 × 48 × 64` tensor. An empty stack repeats the first frame it
receives across all 3 slots; a respawn or a probe teleport calls `clear_vision_history()`, which
empties the stack so the encoder does not read the jump as motion.

**Network.** A shared `EyeNet` (conv 16 @ 5×5 stride 2 → conv 32 @ 3×3 stride 2 → conv 32 @ 3×3
stride 2 → dense 64) runs once per eye. The right eye's 3-frame stack is mirrored left-right and
both eyes carry a side flag (+1 left, −1 right) as a fourth input channel, so one set of weights
serves both eyes. The two 64-unit eye features (128 total) feed a linear head to 8 values; `tanh(·)

- 1`maps them to currents in`[0, 2]`, the same range v4 uses.

**Channels (8, one per anatomical population, uniform current within each):**

| Channel     | Cells (L / R) |
| ----------- | ------------- |
| `mi1_l/r`   | 886 / 887     |
| `tm3_l/r`   | 1,017 / 1,037 |
| `lc4_l/r`   | 71 / 55       |
| `lplc2_l/r` | 94 / 91       |

The encoder cannot address individual neurons, only these 8 uniform populations. Silencing by
pathway name still works: `light_l`/`light_r` aggregate `mi1_*`+`tm3_*` and `looming_l`/`looming_r`
aggregate `lc4_*`+`lplc2_*` per side (`BrainRuntime.pathway_ids`), so the v4 ablation vocabulary
carries over unchanged.

**Sensing order.** Currents applied at step `t` come from the stack ending with the frame
rendered at the _end_ of step `t − 1` (`env.py`: `push_frame` after `step`, `encode_stack` at the
start of the next `_sense`). On `reset`, the drone pushes the first frame and then settles for 40
brain ticks on zero currents — identical for a deployed encoder and one still learning, since
neither has a second frame yet. v4 keeps its own order: it renders and encodes the current frame
synchronously, with no one-step lag.

**Identity.** A learned encoder's version is `"learned-v5:"` followed by the first 16 hex
characters of the sha256 of its weights (`extractor` + `mu` state dicts, sorted keys). An actor
(`load_policy`) loads only into a runtime whose `encoder_version` matches exactly — a decoder
trained against one encoder checkpoint cannot silently run against another. Legacy actors keep
loading into `BrainRuntime()` with no encoder, `ENCODER_VERSION =
"bright-contrast-400-splay075-noaa-loom150-v4"`, unaffected by any of this.

**Metabolic cost** (encoder learner only, not part of the deployed system):

```
r_enc = r − 0.01 · mean(lc4_l, lc4_r, lplc2_l, lplc2_r) − 0.002 · mean(mi1_l, mi1_r, tm3_l, tm3_r)
```

Split rather than one shared λ: a single λ = 0.01 on all 8 channels could cost up to 0.02 per
step (40% of the 0.05 alive bonus) and would push the encoder to dim the light channels foraging
depends on just as hard as it dims loom. Loom should stay quiet unless something is closing, so it
carries the larger penalty (`λ_loom = 0.01`); light is allowed to run higher (`λ_light = 0.002`).
Both terms are logged separately (`rollout/metabolic_cost` and its components) so the trade-off is
visible during training, not just inferred from behaviour.

**Checks E1–E2** (`roam_eval.ENCODER_CHECKS`, spec §5), computed on the 50 held-out evaluation
seeds, intact brain. A threat is _threat-positive_ if it is within 3 m, closing, and visible;
_threat-negative_ if no threat is within 6 m (frames in between are excluded). A beacon is
_beacon-positive_/_negative_ by visibility. `light` = mean(`mi1_*`, `tm3_*`) currents, `loom` =
max(`lc4_*`, `lplc2_*`) currents:

| ID  | Check                                                                                                                                                 | Bar                                                       |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| E1  | ROC AUC of `loom` on threat-positive vs threat-negative frames                                                                                        | ≥ 0.8 (`E1_min_loom_auc`), v4 baseline reported alongside |
| E2  | AUC(`light`→beacon) − AUC(`light`→threat) ≥ 0.05 (`E2_min_light_margin`) **and** AUC(`loom`→threat) − AUC(`loom`→beacon) ≥ 0.1 (`E2_min_loom_margin`) | both hold                                                 |

E1 alone would pass if the loom channels just tracked "anything nearby"; E2 forces the light and
loom channels apart, so a high E1 AUC means the connectome is actually receiving a
threat-selective loom signal rather than the encoder relabelling one general alarm current.
