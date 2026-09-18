# External prior-art harvest — design

**Status:** proposed, authorized 2026-09-18. Workstream B runs now; workstream E (`sign-v2`) is
authorized as an additive new frozen identity. Documentation and design first; no source under
`python/fly_drone/` changes until the docs land.

**Related:** [`external-prior-art.md`](../../external-prior-art.md) is the survey this spec executes.
The sensing work it verifies is [`2026-09-16-retinotopic-sensing-v6-design.md`](2026-09-16-retinotopic-sensing-v6-design.md)
(§7 risk "the visual-axis correspondence is unverified").

**Scope.** Harvest the working pieces of the surveyed projects into this project without touching the
goal contract: the connectome stays frozen, only decoders/encoders learn, `roam_eval.ACCEPTANCE` is
never relaxed, and every accepted actor keeps loading bit-identically.

## 1. Principles kept

- **Frozen brain.** No workstream changes wiring, weights, signs or neuron parameters of the canonical
  `data/malecns` bundle. Alternate bundles (workstreams B and E) are separate, additive identities.
- **Acceptance untouched.** `roam_eval.CONDITIONS` and `ACCEPTANCE` are not modified. Rewired and
  sign probes emit separate diagnostic reports.
- **Legacy reproducible.** `test_legacy_room_mjcf_unchanged` and all accepted actors are unaffected;
  the v6 work in progress keeps its path.
- **Additive identities.** New bundles get a distinct `bundle_hash`; a policy can only run against the
  bundle it was trained on.
- **No long run without sign-off.** Only graph rebuilds and classification probes are new compute;
  none are training runs.

## 2. Step 0 — identity hardening (prerequisite for B and E)

The current identity is `dataset_hash = sha256(graph.bin)` (`brain.py:81`). A sign change lives in
`neurons.bin`, so today it would **not** change the identity: an actor could silently load against a
different sign convention. Fix this additively.

| Requirement | Design |
| --- | --- |
| New identity covers all model-defining bytes | `bundle_hash = sha256(graph.bin ‖ neurons.bin ‖ J)` where `J` is canonical JSON of the manifest fields that define the model (`version`, `n_neurons`, `core_count`, `n_edges`, `w_norm`, `model`, `selection`, `wiring`/`sign_convention` when present), sorted keys, no whitespace |
| Legacy stays loadable | `dataset_hash` (graph-only) is kept and still recorded on the canonical bundle; existing actors validate against it unchanged |
| Alternate bundles are gated | When `manifest.json` marks the bundle (`"wiring": "rewired"` or `"sign_convention": "s2"/"s3"`), `BrainRuntime` requires the policy's `bundle_hash` to match, in addition to `dataset_hash` |
| No silent cross-loading | `Policy::from_json` / `distill` / `sac` / `calibration.warm_start` check `bundle_hash` when the runtime is an alternate bundle |

Tests: flipping one `is_inhibitory` flag changes `bundle_hash` but not `dataset_hash`; identical bytes
give identical hashes; an alternate bundle refuses a policy without the matching `bundle_hash`; the
canonical bundle's `dataset_hash` is byte-for-byte unchanged.

## 3. Workstream A — retinotopic eye-geometry verification

**Source.** `dylankainth/flybrain` `flybrain_eye_map.py`, `eye_map/receptor_directions_buchner71.csv`
(Buchner 1971, digitized by the Straw lab, BSD; vendored with its `NOTICE.md`).

**Landing.** `python/fly_drone/eye_geometry.py` + `python/fly_drone/vendor/eye_map/` + `NOTICE.md` +
`tests/test_eye_geometry.py`.

- Port only the **geometry**: `EyeMap` load (3D unit vectors, ±`left/right`), `visible_mask`,
  `project_to_pixels`, `sample` with a ~5° acceptance window.
- Recompute the frustum for **our** geometry: two cameras splayed ±0.75 rad, 64×48, 177° combined
  (`sensory-model.md` §2), instead of their single 66°×50° Tello camera.
- Verify the v6 map: `scripts/eye_alignment.py` reports the principal angles between each
  population's positional PCA basis (`retinotopy.py:_pca2d`) and the visible ommatidial direction
  PCA, plus the angle from the sheet normal to the mean gaze. This converts the v6 §7 risk from
  "unverified" to "measured". **Result (2026-09-18):** PC1 aligns (0.0°) but PC2 differs by
  67–80° and the sheet normal sits 70–75° from the mean gaze. Because the eye map is in the fly
  head frame, cell positions in the scene frame, and the source marks the cell↔ommatidium join as
  modelled, this is recorded as a **diagnostic, not an assertion**; the v6 functional gates
  (spatial silencing, E1/E2) remain the evidence that the map carries direction.
- **Read-only w.r.t. behaviour:** no change to injection, roles or the encoder. The test is a
  provenance/geometry check, not a gate that can fail the v6 path.

**Not ported.** Their R1-6 `assign_photoreceptors` cell mapping — the repo itself flags the
cell↔ommatidium join as modelled (needs FlyWire visual-column data absent from their files), and this
project injects at Mi1/Tm3/LC4/LPLC2 with a learned encoder.

## 4. Workstream B — rewired-graph causal control (runs now)

**Source.** FlyGM §4.1 (degree-preserving rewiring, Erdős–Rényi, MLP baselines) and
fly-self-driving (`train_street.py`; real 20·19·19/20 vs rewired 16/20).

**Why.** Our ablations shuffle **features**; none rewires **wiring**. A rewired connectome with the
same degree sequence is the strongest available control for "the connectome is doing the work".

**Landing.** `scripts/make_rewired_bundle.py`, `scripts/rewired_report.py`, a `.gitignore` entry for
`data/malecns-rewired/`.

- Read the CSR `graph.bin` (format: `crates/brain-core/src/core/format.rs`), produce a rewired graph,
  and write a sibling bundle with the same `neurons.bin`, `cells.json`, `groups.json` and an augmented
  `manifest.json`. No Rust change: `BrainRuntime(data=...)` already loads an arbitrary directory.
- Two control modes:
  - **R1 `target-shuffle`** — permute each source's target list; preserves out-degree, changes
    in-degree. Cheap, mirrors fly-self-driving's shuffled control.
  - **R2 `edge-swap`** — double-edge swaps preserving in- and out-degree; matches FlyGM's
    degree-preserving rewiring. Stricter and slower.
- Deterministic given `--seed`; the report records seed, mode, edge count and resulting `bundle_hash`.
- `scripts/rewired_report.py` runs a **decoder-free lateralisation probe** — inject left
  Tm4/T2 or the direct LC4/LPLC2 route, measure ipsilateral vs contralateral loom/escape — on
  intact vs rewired for the same seed and writes `runs/diagnostics/rewired-<bundle_hash>.json`.
  A free-roam policy comparison would need a decoder retrained per bundle, which is a training
  run and stays out of scope.
- **Result (2026-09-18, `swap`, seed 0, 16,450,801 swaps).** In-degree and out-degree are
  preserved exactly, `dataset_hash` is unchanged and only `bundle_hash` differs
  (`dbc0f1256bb6`); runtime ~23 s. Left Tm4 injection: laterality **+1.000 intact, +0.000
  rewired**; left T2 the same. The direct LC4/LPLC2 probe measures the injected cells and is
  reported as n/a. First causal evidence that the connectome's specific wiring, not the graph
  size, carries the loom lateralisation.

**Contract guard.** `roam_eval.CONDITIONS`/`ACCEPTANCE` untouched; the report is not an acceptance
run; accepted actors never load `data/malecns-rewired/`; `test_legacy_room_mjcf_unchanged` and the
canonical `dataset_hash` must be unchanged after the scripts run.

**Falsifier.** If intact and rewired are indistinguishable on coverage/foraging/dodging, the
connectome-specific claim is in trouble and must be reported, not hidden.

## 5. Workstream C — fly.ai oracle cross-check

**Source.** `flybrain` PyPI (frozen 166,700-neuron MaleCNS LIF, `brain.cells`, `brain.step(inject=)`),
and fly.ai's published laterality results.

**Landing.** `scripts/check_fly_ai_oracle.py` (optional dependency `flybrain`, skipped when absent;
not in default `pytest`).

- Inject a constant current into left LC4+LPLC2 and assert our Rust trace shows left DNp01 activity
  greater than right (fly.ai: left +17–25 sp/s, right unchanged); repeat for LC10a → DNa02 on the
  left only. Compare **direction/laterality**, not absolute rates (different `dt`, tonic and gains).
- Reconcile the edge-count discrepancy (§5 of the survey): count ordered pairs under our ≥3 filter
  (10,520,431) and under fly.ai's build (25,582,938) from the same source file and document the
  definitional difference in `external-prior-art.md` and `data-pipeline.md`.
- **Result (2026-09-18):** the edge probe runs and reports the 2.43× gap as open (needs the raw
  flat-connectome feather, not vendored). The laterality probe is written and gated but **skipped**:
  `flybrain` is not installed. Install `pip install flybrain` to run it; it is never in `pytest`.

**Falsifier.** A laterality or sign inversion relative to fly.ai invalidates the current input roles
or sign hypothesis and blocks workstream E until resolved.

## 6. Workstream D — stimulus-battery calibration

**Source.** FlyDrones `src/flydrones/calibrate.py` (`STIMULI`, `record_responses`, `fit_readout`).

**Landing.** `scripts/stimulus_battery.py`, emitting `runs/diagnostics/battery-<encoder>.json`.
A standalone script rather than an edit to `calibration.py`, to avoid touching a module on the active
v6 path; integrate later only if useful.

- Seven stimuli adapted to the cues this project can drive (light and loom per side; v4 has no
  vertical-flow cue, so sinking/rising are dropped). Baseline-centred ridge from the eight compact
  readouts to turn/escape targets.
- **Result (2026-09-18):** escape **R² = 0.981**, turn **R² = 0.327**. The compact readouts carry
  the loom/escape direction strongly and the light-direction turn only weakly — expected, since the
  decoder reads all 2,022 DN/VNC traces, not these eight readouts.
- Diagnostic only: no decoder, no threshold, no acceptance.

## 7. Workstream E — transmitter-sign `sign-v2` (authorized, additive)

**Conventions.** S1 current (`ACh +`; `GABA/Glu −`; unresolved/modulatory silent,
`neuron-model.md:68`); S2 fly.ai (`GABA`, `Glu`, **histamine** inhibitory); S3 FlyGM (`ACh`, `Glu`,
`ASP`, **`His`** excitatory; `GABA`/`Gly` inhibitory).

**Landing.** `scripts/make_sign_bundle.py`, `data/malecns-sign-s2/`, `data/malecns-sign-s3/` (both
git-ignored), and a decision report under `runs/diagnostics/`.

- The flag that Rust reads is `is_inhibitory` in `neurons.bin` (`format.rs:71`); edges are unchanged.
  Each variant is a regenerated `neurons.bin` with a `sign_convention` marker in `manifest.json` and a
  new `bundle_hash` (Step 0). No Rust change.
- Run the laterality and loom probes (workstream C) plus a small behaviour probe across S1/S2/S3.
- **Contract:** the canonical bundle, `ENCODER_VERSION` and every accepted actor stay untouched.
  `sign-v2` becomes a **new frozen identity** only after the probes justify it and the user signs off;
  S1 remains the default until then.

**Falsifier.** If the variants do not differ on the laterality/loom probes, the sign convention is not
the limiting factor and `sign-v2` is not adopted.

**Status (2026-09-18): blocked pending a dependency decision.** The raw
`body-neurotransmitters-male-cns-v1.0.feather` (43 MB) downloads and its SHA-256 matches the manifest
(`95c92892…`). It is dictionary-coded with labels `acetylcholine`, `gaba`, `glutamate`, `histamine`,
`octopamine`, `serotonin`, `unclear`. But the project venv has **no Arrow/Feather reader** (no
`pyarrow`, `pandas`, `polars`, `duckdb`, `fastparquet`), the upstream sign mapping is not in this repo
to reproduce, and no `sign-v2` code has been written. E needs an optional Arrow dependency (or a
separate environment) before it can proceed; the canonical bundle is unchanged.

## 8. Sequencing

| Step | What | Depends on | Compute |
| --- | --- | --- | --- |
| 0 | This survey + spec (docs only) | — | none |
| 1 | Identity hardening | docs | fast tests |
| 2 | **B** rewired bundle + report (now) | Step 1 | graph rebuild, minutes–hour |
| 3 | A eye-geometry verification | — | fast tests |
| 4 | D stimulus battery | — | minutes |
| 5 | C fly.ai oracle + edge-count reconciliation | A/roles | one brain build |
| 6 | E sign-v2 probes + decision | Step 1, C | graph rebuild + probes |

Every step: `env -u PYTHONPATH .venv/bin/python -m pytest -q`, `.venv/bin/ruff format python tests &&
.venv/bin/ruff check python tests`, then commit and push.

## 9. Artifacts

- `docs/external-prior-art.md` (survey), this spec, `docs/references.md` links.
- `python/fly_drone/eye_geometry.py`, `python/fly_drone/vendor/eye_map/` + `NOTICE.md`,
  `tests/test_eye_geometry.py`, `scripts/eye_alignment.py`.
- `scripts/make_rewired_bundle.py`, `scripts/rewired_report.py`,
  `scripts/check_fly_ai_oracle.py`, `scripts/stimulus_battery.py`, `scripts/eye_alignment.py`,
  `scripts/make_sign_bundle.py` (pending).
- `runs/diagnostics/` (git-ignored): rewired report, battery, sign decision.
- Test additions: identity hashing, state-carry regression, eye-geometry alignment.

## 10. Open decisions

1. **Rewiring mode** — R1 `target-shuffle` first (recommended for runtime) vs R2 `edge-swap`
   (matches FlyGM §4.1) vs both.
2. **Oracle scope** — laterality only (recommended) vs also porting `flybrain`'s readout as a second
   training path (rejected: different thesis).
3. **`sign-v2` adoption** — default S1 until the probes report; any change to the canonical sign
   convention is a separate contract change requiring explicit sign-off. **Blocked:** needs an
   optional Arrow reader and the upstream transmitter→sign mapping (see workstream E status).
4. **Edge-count reconciliation** — must be explained before any edge count is cited externally.
