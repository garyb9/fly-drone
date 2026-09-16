# Retinotopic sensing (encoder v6) — design

**Status:** proposed, awaiting user decision (2026-09-16). The first gate (M1/M1b feasibility) has
been run and **passed, with a corrected injection target**.

**Related:** the action side is [`2026-09-16-wing-level-action-design.md`](2026-09-16-wing-level-action-design.md).
Together they widen what the connectome senses and deepen what it commands; they share the training
and evaluation changes described here (§5–§6) and in that spec (§6–§7).

Documentation + read-only probes only — no training has been run, no `ACCEPTANCE` threshold touched,
and no source under `python/fly_drone/` changed.

## 1. Why

`overview/README.md` states the goal: the connectome **sees through the drone's cameras** and every
skill is attributable to neurons. Today it receives only **8 uniform scalar currents** (Mi1, Tm3,
LC4, LPLC2 per side), computed by an engineered adapter (`sensory-model.md` §2/§6). The measured
consequence:

| Evidence | Value |
| --- | --- |
| E1 loom selectivity, v4 | **0.687** (pre-registered bar 0.8) |
| E1, round-0 v5 | 0.686 |
| Ratio of connectome to encoder in the visual path | optic lobe is 53.6% of the brain (89,403 cells) but is driven by 8 numbers |

No decoder can recover threat direction the encoder discarded. The goal needs a **spatially
structured** signal entering the connectome, so the optic lobe's own circuitry — not a hand-built
global scalar — computes motion and looming. This spec is that change.

## 2. Feasibility gates (run 2026-09-16; scripts under `scripts/`)

### 2.1 M1 — geometry probe (`scripts/retinotopy_probe.py`)

For each candidate population, PCA of the per-side 3D centroids, grid occupancy (24×24), planarity.

| Population | n / side | Planarity | PC1 / PC2 | Occupancy | Structure |
| --- | --- | --- | --- | --- | --- |
| L1 / L2 / L3 | ~884 | 0.987–0.990 | **0.96 / 0.02** | **0.06–0.10** | **1D arc** — no 2D map |
| Mi1 / Tm3 | ~886 | 0.93 | 0.56 / 0.37 | 0.36 | genuine 2D sheet |
| Tm4 | ~835 | 0.93–0.95 | 0.54 / 0.40 | 0.36 | genuine 2D sheet |
| Tm2 / Tm1 | ~883 | 0.93 | 0.55 / 0.37 | 0.36 | genuine 2D sheet |
| T2 | ~815 | 0.99 | 0.63–0.80 / 0.18–0.36 | 0.24–0.40 | 2D, PC1-dominant |
| T4a/b, T5a/b | ~840 | 0.96–0.97 | 0.77 / 0.20 | 0.28–0.56 | 2D curved strip |
| LC4 / LPLC2 | 55–185 | 0.99 | — | 0.09–0.15 | too sparse to tile |

**Finding:** the lamina (L1–L3) collapses to a 1D arc in centroid space, so my initial
recommendation to inject there is **falsified**. The earliest dense 2D sheets are Mi1/Tm3, Tm4 and
T2.

### 2.2 M1b — propagation assay (`scripts/spatial_assay.py`, `propagation_trace.py`, `input_candidate_scan.py`, `spatial_lateralization.py`)

Uniform current 1.6 into each candidate population (left side, 120–240 ticks), measuring ipsilateral
LC4/LPLC2 and the escape readout:

| Injected population | LC4_l | LPLC2_l | escape |
| --- | --- | --- | --- |
| LC4 (direct) | 0.952 | 0.000 | 0.751 |
| LPLC2 (direct) | 0.000 | 0.952 | 0.821 |
| **T2** | **0.493** | 0.000 | **0.549** |
| **Tm4** | **0.292** | 0.001 | **0.535** |
| Tm2 | 0.026 | 0.000 | 0.003 |
| Mi1, Tm3, L1–L5, Mi4, Mi9, Tm1, T1, T3, T4a/b, T5a/b, Lawf1/2 | ~0.000 | ~0.000 | ~0.000 |

A direct trace of a uniform Mi1/Tm3 injection shows the signal reaching T4a at only 0.04 and
LC4/LPLC2 at ≤0.001 — **the Mi1/Tm3 → T4/T5 → loom route does not carry usable drive in this
frozen graph.** The loom/escape circuit is fed in practice by **T2** and **Tm4** (and by the direct
LC4/LPLC2 route v4 already uses).

Lateralisation and spatial selectivity (left/right whole-side, then half-sheet):

| Stimulus | Result |
| --- | --- |
| Tm4 left / Tm4 right | LC4_l 0.292 / LC4_r 0.382; cross-talk 0.000; **lateralisation +1.00** |
| T2 left / T2 right | LC4_l 0.493 / LC4_r 0.792; cross-talk 0.000; **lateralisation +1.00** |
| Tm4 left, low-PC1 half vs high-PC1 half | esc 0.418 vs 0.142 → **sub-regions differ (spatially selective)** |
| T2 left, low vs high half | esc 0.543 vs 0.532 → **wide-field, not selective** |

**Gate result: PASS.** A spatially structured current has a spatially structured, perfectly
lateralised effect on the connectome's loom/escape output. The injection target is corrected to:

- **Tm4 — primary retinotopic channel** (dense 2D sheet, lateralised, sub-region selective).
- **T2 — secondary wide-field channel** (strongest driver, but not direction-selective).
- **LC4 / LPLC2 — retained direct channel** (v4's guaranteed route; LPLC2 is *only* reachable
  directly).
- **Rejected, with evidence:** Mi1/Tm3, L1–L5, T4/T5 (no propagation to the loom circuit).

## 3. Principles kept

- Connectome frozen: no cell added, no weight, sign or parameter changed.
- The encoder sees **only its own camera frames**. No pose, target, threat position or task id.
- Bounded uniform current per patch, within `[0, 2]`; the encoder still cannot address individual
  neurons — it drives a low-resolution spatial map, not a control code.
- One encoder and one decoder for free roam, no mode switching.
- Legacy is untouched: v4 and every accepted actor keep loading; `learned-v6:` is a new identity.
- `roam_eval.ACCEPTANCE` and E1/E2 bars unchanged. Any new criterion is additive.

## 4. Architecture

```
2 eyes 64x48 luma, 4 frames (3 past + 1 current, frame differences)
        |
   [EyeNet CNN]  ->  low-res current map per eye:  NX x NY  x  channel-groups
        |
   [precomputed retinotopic mapper]  position -> map cell (per population, per side)
        |
   Tm4 (primary) + T2 (wide-field) + LC4/LPLC2 (direct)   currents in [0,2]
        |
   FROZEN CONNECTOME  ->  motion/loom/escape computed internally
```

- **Map:** default `NX x NY = 12 x 8` per eye per group, uniform within a patch. Each neuron is
  assigned to the nearest patch by 2D PCA of that population's positions (cached offline in the
  encoder identity).
- **Channels:** `tm4_l/r` (12×8), `t2_l/r` (12×8), `lc4_l/r` (coarse, e.g. 4×3 given sparsity),
  `lplc2_l/r` (coarse). Bandwidth rises from 8 scalars to roughly `2 × (96 + 96 + 12 + 6) ≈ 420`
  currents per frame; the true ceiling is set by the graph, not by us.
- **Input temporal channels:** explicit frame difference (L8 s16; v4's working cue is literally
  `ΔD`), on top of the 3-frame stack.
- **Identity:** `learned-v6:<weights-hash>`; `BrainRuntime` builds the spatial input roles from the
  cached mapper; a decoder pinned to a v6 encoder cannot run against another.
- **Injection cost:** ~400 `inject` calls per 40 ms frame over precomputed roles — negligible next
  to the whole-brain LIF step. The bottleneck stays CPU simulation, not the encoder.
- **Code (additive so far):** `python/fly_drone/retinotopy.py` (position -> patch roles) and
  `python/fly_drone/spatial_encoder.py` (the `learned-v6:` conv net with frame-difference inputs,
  emitting the patch maps; 432 currents/frame). Both are standalone — no `brain.py` wiring yet, so
  the v5 path is untouched. Wiring lands with M2.

### 4.1 Clone / warm start

The v4 clone remains the warm start, but it now lands on the spatial channels: distribute the v4
loom cue across the `lc4`/`lplc2` patches and the v4 light cue across a coarse light map. The
existing difference-aware and mirror augmentation (`encoder.py`) carry over.

### 4.2 Domain randomisation

Brightness/contrast/noise at `push_frame`, plus layout/band randomisation, so the map keys on
structure rather than absolute luma. Required for the goal's "unseen arenas".

## 5. Training

Carried over from the analysis, re-based on the spatial interface. The full argument is in the run
notes; the essentials:

| Change | Source | Why |
| --- | --- | --- |
| Spatial encoder with motion input + domain randomisation | L8 s16, L7 s2 | Represent loom/motion directly |
| Predictive auxiliary objective (next DN-trace, EMA target) | L8 s27/s58 | Dense, reward-free gradient; makes currents dynamically meaningful |
| Threat curriculum: threats-without-pillars training level | L2 s44, L9 s52 | Threats are ~3–7/episode; lateral/vertical get no gradient |
| Dense, potential-based threat shaping | L2 s44, `training.md` §2 | Sparse −20 collisions propagate too slowly |
| Recovery-focused DAgger | L3 s41, Aviral Kumar | Recovery data is sight-realizable and attacks covariate shift |
| Gated joint encoder+decoder training | L8/L9 | Removes the alternating non-stationarity — **only** with E1/E2 and E3 as hard anti-wire gates |

Deep-dive triggers a decoder change only if the vy/vz instrumentation (M1–M3, landed in Phase 1)
shows mode-averaging; see the action spec §5.

## 6. Evaluation

- **Preserved:** A1–A7, E1–E4, threat-aim scoring, `roam_eval.ACCEPTANCE` (no threshold moved).
- **New causal gate — spatial silencing:** silence one hemifield / one map region and require the
  mirrored behavioural change. This is strictly stronger evidence than whole-pathway silence and
  directly serves the goal's causality clause.
- **New generalization gate:** train on a distribution of layouts, evaluate on a shifted one
  (hw3 adversarial obstacle; L9 SIMPLER ranking). The current held-out seeds share the arena family.
- **E3 (brain bypass)** becomes more decisive: with a spatial encoder, a bypass decoder matching the
  full system would show the optic lobe is still decorative.
- **E1/E2 re-based on spatial channels**, reported against the v4 0.687 baseline; the 0.8 bar stays.

## 7. Risks and falsifiers

| Risk | Evidence so far | Mitigation |
| --- | --- | --- |
| Spatial current too weak to drive the graph | Real: Mi1/Tm3 → 0; Tm4 → LC4 0.29 at full drive | Use Tm4/T2; retain the direct LC4/LPLC2 route; per-patch gain is a tunable in the encoder head |
| The map is not truly retinotopic | Tm4 has a clean 2D sheet; the visual-axis correspondence is unverified | Functional mapping is enough for the agent; verify with the spatial-silence gate, not anatomy |
| LPLC2 unreachable except directly | Confirmed (Tm4/T2 → LPLC2 ≈ 0) | Inject LPLC2 directly; keep it in the channel set |
| Widening increases SAC search | — | Clone init + predictive auxiliary objective + curriculum |
| `learned-v6` invalidates v5 round-0 data | Certain | Land after the Phase 2 re-run; v5 stays reproducible |

## 8. Sequencing (local GPU)

The binding cost is CPU whole-brain simulation, not the RTX 4070. Spatial injection is near-free in
compute; the expensive items are the extra training rounds.

| Step | What | Gate | Cost |
| --- | --- | --- | --- |
| M0 | Phase 2 re-run of v5 (entropy fix + guard + fixed-probe E1) | — | existing ~11 h |
| **M1/M1b** | **Done.** Geometry probe + propagation/lateralisation assay | **Passed** | minutes |
| M2 | Spatial encoder v6 (Tm4/T2/LC4/LPLC2), motion input, domain randomisation, v4-clone init | Spatial E1/E2 beat v4; ghost clean | collection + 1 round |
| M3 | Training signal: curriculum, shaping, recovery DAgger, n-step; read M1–M3 | Causal dodge (ghost ≤ 0.3, E2 pass) | 1–2 rounds |
| M4 | Gated joint training | Joint > alternating **without** failing E1/E2/E3 | 1 long run |
| M5 | Unseen-layout distribution + final E1–E4 | Goal-level generalization | eval |

## 9. Artifacts from this step

- `scripts/retinotopy_probe.py` — M1 geometry probe.
- `scripts/spatial_assay.py`, `scripts/propagation_trace.py`,
  `scripts/input_candidate_scan.py`, `scripts/spatial_lateralization.py` — M1b propagation and
  lateralisation assays.
- `runs/probe/retinotopy/{report.json,*.png}`, `runs/probe/spatial_assay.json` (git-ignored).
- `python/fly_drone/retinotopy.py` + `tests/test_retinotopy.py` — the mapper and its tests.
- `python/fly_drone/spatial_encoder.py` + `tests/test_spatial_encoder.py` — the v6 network and its
  tests (frame-difference inputs, per-channel map shapes/bounds, `learned-v6:` version, flatten).
- `python/fly_drone/predictive.py` + `tests/test_predictive.py` — the auxiliary predictive objective
  (EMA target, VICReg anti-collapse, collapse diagnostic).

## 10. Open decisions

1. **Map resolution** (`12×8` default) — larger buys resolution, costs encoder capacity; recommend
   12×8 first.
2. **Channels** — Tm4 + T2 + direct LC4/LPLC2 (recommended) vs Tm4 only vs all four.
3. **Timing** — land after the M0 Phase 2 re-run (recommended) vs fold into it.
4. **Joint training** — proceed at M4 with E1/E2/E3 hard gates (recommended) vs stay alternating.
