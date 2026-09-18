# Anatomical eye-map sensing (v6b) — plan

**Status: closed, not implemented (2026-09-18).** Task 1's diagnostic killed the approach before any
code: the chart's second principal angle is the dihedral angle between the cell-sheet plane and the
ommatidial-direction plane (67–81°), which no rotation about the shared PC1 can change, so no rigid
direction-aligned chart exists; and no true cell↔ommatidium join is available locally
(`cells.json` carries no ommatidial/column coordinate). See
[`CONCLUSION-2026-09-18.md`](../../results/encoder-v6/CONCLUSION-2026-09-18.md).

**Goal:** replace the v6 encoder's *rectangular* retina and its *position-PCA* patch chart with the
measured eye geometry, so the injected currents follow ommatidial viewing directions rather than a
degenerate second principal axis. Keep the connectome frozen and every anti-wire gate in place.

**Why.** `scripts/eye_alignment.py` (workstream A, commit `dbf27bb`) measured the v6 chart against the
vendored Buchner-1971 ommatidial directions: PC1 aligns at **0.0°** but PC2 differs by **67–81°**, and
the population sheet normal sits **69–81°** from the mean gaze. `retinotopy.build_map` uses PC1→patch-x
and **PC2→patch-y**, so the 12×8 grid's second axis carries no visual meaning; 768 of the 816 currents
(4 spatial channels × 96 patches × 2 eyes) are charted through it. That is a plausible reason the
encoder, which fits the low-dimensional v4 cues anyway (light r 0.998 / loom r 0.934), gives SAC no
usable gradient to improve through the connectome — both M4 runs collapsed instead
(`docs/results/encoder-v6/JOINT-2026-09-17.md`).

**Hard constraint (the join).** `data/malecns/cells.json` carries only `position`, `type`, `side`,
`nt`, `group`. There is **no ommatidial/column coordinate**, and the source project marks the
cell↔ommatidium join as modelled. So a true per-cell viewing direction is unavailable; the chart below
is an explicit, geometrically-motivated **modelled chart**, not an anatomical join. Honest labels:
every artefact records `chart="eye-directions-modelled"`.

## Design

Two separable changes; A is required, B is the substantive fix.

**A. Anatomical retina front-end.**
- Build per-ommatidium luma from the rendered eye images with `EyeMap.project_to_pixels` /
  `sample` (measured directions, our 64×48 splayed cameras), instead of the current rectangular
  `eye_inputs`. Visible ommatidia per eye ≈ 700–1000.
- Keep the temporal contract: a 3-frame stack plus frame differences plus a side flag
  (`spatial_encoder.eye_inputs` shape/behaviour is the reference).
- The trained backbone still consumes a fixed-size image, so (i) rasterise the ommatidial samples into
  a fixed sensor image (e.g. resample the visible ommatidia to a 32×24 grid in
  (azimuth, elevation) with a nearest/acceptance kernel), or (ii) build a permutation/adjacency that
  lets a 1-D/2-D conv run on the ommatidial index order. Recommend (i): reshape-only, no new net.

**B. Direction-aligned patch chart.**
- For each (population, side): take the visible ommatidial direction basis (already computed in
  `eye_alignment.direction_basis`). Fit the cell-position sheet onto that basis: keep PC1 (aligned),
  and choose the PC2 axis by the rotation about PC1 that best matches the measured direction PC2
  (equivalently, patch by the cell positions projected onto the direction basis after the best-fit
  rotation). Normalise to `nx × ny`.
- Rebuild `RetinotopicMap`/`define_roles` with unchanged patch counts (`DEFAULT_GRID 12×8`,
  `DIRECT_GRID 4×3`) so roles, metabolic cost, group slices and `flat_dim() == 816` are unchanged.
- Re-run `eye_alignment.py`; the new chart's PC2 angle and sheet-normal angle become the recorded
  diagnostic. Set a **measured target** (propose PC2 ≤ 25°, normal ≤ 30°); if the fit cannot reach it,
  stop and report rather than relax it.

**Injection mechanism unchanged:** one Rust input role per occupied patch; values clipped to [0, 2].

**Encoder:** keep the learned v6 encoder in phase 1 (reuse `learned-v6`, `SpatialEncoder`,
`JointSAC`, all gates). Phase 2 (optional) replaces the learned maps with fixed optic-lobe-style
filters (luminance for Mi1/Tm3, temporal difference for Tm4/T2, loom for LC4/LPLC2), which removes the
SAC-collapse failure mode entirely at the cost of a hand-built front end.

## Gates (unchanged bars)

| Gate | Bar |
| --- | --- |
| Chart | re-measured PC2 / normal reach the measured target (proposed 25° / 30°), else stop |
| Clone | light r and loom r no worse than the current clone (0.998 / 0.934) |
| Foraging | 30-seed level-2 beacons/min ≥ 0.8 × v4 reference (gate 1.547) |
| E1/E2/E3 | unchanged: union E1, E2 margins, bypass not better |
| Liveness | light ≥ 0.10, loom ≥ 0.08 |

`roam_eval.ACCEPTANCE` is not touched. v4/v5 and the accepted actors stay loadable.

## Tasks

1. **Chart module + diagnostic.** `retinotopy.build_map` gains an `eye_chart` path using
   `EyeMap`; `eye_alignment.py` reports before/after. Tests: chart is deterministic; `flat_dim()`,
   group/channel slices and role names unchanged; a synthetic aligned sheet reaches the target.
2. **Anatomical retina.** Sensor image builder from `EyeMap.sample`; tests: shape/range, left/right
   split, frame-difference contract; a spot at a known bearing lands in the expected patch.
3. **Re-clone.** Re-fit the spatial clone on the existing `runs/v5/clone/data.npz` (no new flights);
   verify light/loom r and E1 on the fixed probe.
4. **Round 0.** Freeze-decoder round-0 fit; 30-seed gate.
5. **Train (needs sign-off).** One encoder round or one joint round (whichever the user picks) with
   the same guard/liveness/E1/E2/E3; report and keep-or-revert.
6. **Report.** `docs/results/encoder-v6/EYE-CHART-<date>.md` with the chart angles, clone fidelity,
   gate, E1/E2/E3 and the verdict.

## Open decisions (need the user)

1. **O1 — modelled chart.** Accept a modelled direction-aligned chart (no true join) as the honest
   best available, or stop v6 sensing claims at the current position chart?
2. **O2 — encoder.** Phase-1 learned encoder (reuses everything; may still collapse) vs phase-2 fixed
   optic-lobe filters (no collapse; more hand-built code). Recommend phase 1, phase 2 only if it
   collapses.
3. **O3 — retina raster.** Resample ommatidia to a fixed 32×24 sensor image (recommended), or keep the
   native ommatidial index order with a custom adjacency.
4. **O4 — identity.** Advance to a new prefix (`learned-v6b:`) so the current `learned-v6:` clone and
   decoders keep loading, or keep `learned-v6:` and accept that the charts differ (rejected — the
   chart changes the map, so it must change the hash).

## Risks

| Risk | Mitigation |
| --- | --- |
| Modelled chart is not anatomical | record it; rely on the functional gates, not the chart, as evidence |
| Rasterisation loses the measured geometry | spot-check known bearings land in the expected patch (Task 2) |
| Chart does not improve E1 | Task 1 + clone before any long run; stop if the clone or chart target misses |
| New identity breaks tooling | dedicated prefix, keep `learned-v6:` loading; Rust prefix list gains `learned-v6b:` |
| Cost | no new flights; reuse the existing clone data; one gated round at most |
