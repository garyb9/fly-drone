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
| Yaw splay          | left `+0.45 rad` (+25.8°), right `−0.45 rad` (body `+Y` is left)      |
| Angular resolution | `64 px / 91.3° ≈ 0.70 px/deg`                                         |

In MuJoCo a camera looks along its local `−Z`. `xyaxes = (sin ψ, −cos ψ, 0; 0, 0, 1)` puts local
`+Y` world-up and the view direction at body yaw `ψ`. Horizontal coverage per eye is `ψ ± 45.6°`:

```
left eye   [−19.8°, +71.4°]
right eye  [−71.4°, +19.8°]
binocular overlap  ±19.8°  (39.7° wide), total field ±71.4° (142.7°)
```

A target straight ahead is seen by both eyes about equally. As its bearing moves past ±19.8°, it
leaves one eye. That left/right imbalance is the signal a steering decoder can exploit.

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
loom cue   c_O = clamp(12 · max(0, D − D′),         0, 2)
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
- `12 · max(0, ΔD)` is a crude **looming** proxy: the dark area growing between frames.

### Why dark-area growth approximates looming

For an approaching object of radius `R` at distance `d(t)`, closing at speed `v = −ḋ`, the
small-angle angular size is `θ ≈ 2R/d`. Its image area grows as `θ²`:

```
D ∝ θ² ∝ 1/d²   ⇒   Ḋ ∝ 2 v / d³  = 2 D / τ,   τ = d / v   (time to contact)
```

So `ΔD` per 40 ms frame rises steeply as the obstacle nears (`∝ 1/d³`), much like the
`θ̇`/`τ`-tuned responses attributed to LC4/LPLC2. Confounds: yaw rotation that sweeps a dark
region into view, lighting changes, and a drone that is the one approaching.

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
