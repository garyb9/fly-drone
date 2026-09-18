# Encoder v6 sensing — conclusion (2026-09-18)

**Verdict: closed, negative.** The v6 goal — a learned retinotopic encoder that raises E1 above the
0.8 bar by giving the optic lobe 816 directional currents instead of v4's 8 scalars — is not reachable
with the available connectome data and the SAC objective at this scale. v6's infrastructure is kept;
the sensing claim is withdrawn. `roam_eval.ACCEPTANCE` and the E1/E2 bars are untouched throughout.

## What v6 shipped (reusable)

- `learned-v6:` encoder identity, `BrainRuntime` v6 path, Rust prefix, `SpatialEncoder`,
  `SacRoamEnv` spatial mode, `spatial-maps`/E1–E2 tooling, viewer `sensory` telemetry.
- `JointActor`/`JointSACPolicy`/`JointSAC` joint encoder+decoder with **per-head entropy**, `JointRoamEnv`,
  joint export/load, and v6 bypass support (E3) — all tested.
- `eye_geometry.EyeMap` (measured Buchner-1971 directions) and the eye-alignment diagnostic.
- The viewer can now fly a learned-encoder pair (`serve --encoder`), and does for the v6 round-0 pair.

## What v6 achieved

- **Clone** `learned-v6:01280e414169ff9c`: light r 0.998, loom r 0.934 on held-out frames.
- **Round-0 decoder** under the clone: 30-seed level-2 gate **PASS 2.00** (gate 1.547).
- **Fixed-probe E1 0.720 (fail, bar 0.8)**; E2 pass. Round 0 is the best valid v6 pair, not accepted.

## What failed, and why

| Attempt | Outcome |
| --- | --- |
| Task 11 alternating encoder (5 attempts) | collapsed: saturate 2.0 / centre 0.995 / near-constant; one passed 100k then degraded by 350k (`RESULTS-2026-09-16.md` §6–§7) |
| M4 joint, single scaled α (300k) | guard fail (beacons 0.00, ghost dodge 1.00, E2 fail); E1 union 0.492; auto-α drifted 9.8e-5 → 0.048 (`JOINT-2026-09-17.md` §1–§6) |
| M4 joint, per-head α (stopped ~160k) | encoder saturated by 100k (light/motion pinned 2.0); velocity α 0.01 → 3.6e-7 (`JOINT-2026-09-17.md` §7) |

The 816-dim action has a degenerate optimum (constant/saturated currents) that the task reward, the
clone anchor, the predictive auxiliary objective, the liveness gate, and both entropy regimes failed
to prevent. The alternating scheme's frozen partner and the joint scheme's single critic both route
too little usable gradient to the encoder.

## The structural finding (eye-alignment diagnostic)

`scripts/eye_alignment.py` compared each v6 population's chart (`retinotopy.build_map`: PC1→patch-x,
PC2→patch-y) with the measured ommatidial directions:

- PC1 aligns at **0.0°**; PC2 differs by **67–81°**; the population sheet normal sits **69–81°** from
  the mean gaze.
- The second principal angle is the **dihedral angle between the two planes** and is invariant to any
  rotation about the shared PC1, so **no rigid chart can fix it**. A direction-aligned rigid chart was
  attempted (plan 08, Task 1): index-based Kabsch gave inconsistent results (7–77°), as expected
  without a correspondence.
- This is what a **curved** retinotopic surface looks like under 2-D PCA: the PCA plane is not the
  tangent/visual plane. The correct chart needs the true **cell↔ommatidium join**, which
  `data/malecns/cells.json` does not carry (only `position`, `type`, `side`, `nt`, `group`), and which
  the source project marks as modelled.

So 768 of the 816 currents are charted through a second axis with no visual meaning. That is a
plausible, though not proven, contributor to the collapse — and either way it means v6 could not
honestly be called retinotopic from the data on hand.

## Consequence

- v6 sensing is closed. Do not spend further training compute on a learned v6 encoder.
- `docs/results/current-policies.json` stays on the v5 pair; the v6 round-0 pair remains loadable
  (viewer/`--encoder`) as the best valid v6 artifact but is not accepted.
- Plan 07 (M4) and plan 08 (v6b anatomical chart) are closed. Task 12 (V6.2 training signal) is moot.
- If v6 is ever revived, the prerequisite is the real cell↔ommatidium join (e.g. FlyWire optic-lobe
  column annotations), not another training scheme.

## Next

Wing-level action and the vertical axis (deferred while M4 ran). Scope separately.
