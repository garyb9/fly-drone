# C4 — use the brain's escape response, and make loom selective

**Status:** A0 feasibility passed and A1/A2 landed 2026-09-20 (codec `v3` =
`declared-v3:467faae753cef735`, free falsifier passed — see
[`../../results/adapter/FINDING-2026-09-20-c4a-a0.md`](../../results/adapter/FINDING-2026-09-20-c4a-a0.md)
and [`FINDING-2026-09-20-c4a-a1-a2.md`](../../results/adapter/FINDING-2026-09-20-c4a-a1-a2.md)).
**A3 run 2026-09-22: v3 fails — C4a alone is rejected** by its own gate (A3 dodge 0.64 → 0.34, and
liveness L6 regressed) even though it raises near-threat `|vy|` to 0.8 and passes the ghost
causality test — [`FINDING-2026-09-22-c4a-a3.md`](../../results/adapter/FINDING-2026-09-22-c4a-a3.md).
Do not spend the 50-seed gate on C4a. **C4b (selective loom) is now the prerequisite**, and the
climb cut is a candidate to revisit. Codec v3 remains selectable; whether it stays the shipped
default is an open user decision (see `docs/HANDOFF.md`).

**B0 run 2026-09-22: no adoptable C4b candidate.** Neither the v4 front-end (E1 0.738) nor the
frozen v6 spatial clone (E1 0.720, motion 0.657, union 0.710) clears the E1 ≥ 0.8 bar, and the
relay route is structurally infeasible (T4/T5 do not reach LC4/LPLC2; a translation model cannot
represent looming) — [`FINDING-2026-09-22-c4b-b0.md`](../../results/adapter/FINDING-2026-09-22-c4b-b0.md).
**C4b is blocked on E1 and is a user decision** (train v6 to clear E1, design a new selective
front-end, or defer avoidance).
Successor to the P4 spec
([`2026-09-20-ongoing-state-and-faithful-readout.md`](2026-09-20-ongoing-state-and-faithful-readout.md)).
Contract changes **C4a** (bridge uses the connectome's escape/loom response on the lateral axis) and
**C4b** (a selective loom front-end), with **C4c** optional (read the descending giant-fibre cells).
Additive, versioned, silenceable; `roam_eval.ACCEPTANCE` and the canonical bundle never change.

Evidence: [`docs/results/adapter/FINDING-2026-09-20-v2-acceptance.md`](../../results/adapter/FINDING-2026-09-20-v2-acceptance.md)
and [`FINDING-2026-09-20-loom-escape-audit.md`](../../results/adapter/FINDING-2026-09-20-loom-escape-audit.md).

## 1. The principle this spec adds

P4 established _alive_ (the brain moves the body). This spec is about _using what the brain already
computes_. The audit shows the connectome's own escape response is present and correct — it reaches
~0.9 on a looming object in both the declared bridge and the accepted teacher — while the declared
bridge converts it into a **climb** and a fractional sidestep (`0.5 · yaw`), so the drone flies into
threats and walls. **A sensory pathway that fires is not the same as a behaviour that uses it**: the
second half of the loop is the bridge's job, and it must put the signal on the axis the behaviour
needs. As always: the brain decides, the body executes, the constants are declared (no teacher, no
mode switch), and any addition is silenceable and versioned.

## 2. Facts (measured, free)

1. **Sensing and the brain's loom response work.** `loom` saturates (~2.0) and the connectome's
   `escape` readout reaches 0.87–0.96 over near threats; the teacher's reaches 0.92–0.96.
2. **The bridge under-commands lateral escape.** Near-threat peak `|vy|` is 0.24–0.39 for v2 vs
   **1.0** for the teacher, while `vz` (climb) peaks at 1.0. The codec has `vy = 0.5 · yaw`.
3. **There is no wall avoidance.** v2 takes 9–16 collisions per 120 s, all but 1–2 on walls; the
   teacher takes 0–2. A head-on wall gives a symmetric (steer ≈ 0) loom response → climb, not turn.
4. **The loom cue is not threat-selective.** It saturates on turn sweeps (≈28 % of frames ≥ 0.22
   between threats). [`sensory-model.md`](../../sensory-model.md) records the cause: the v4/v5
   8-scalar loom cue caps the E1 selectivity AUC at 0.686 (< 0.8), and Mi1/Tm3/T4/T5 do not reach the
   loom circuit. The documented fix is the v6 spatial (retinotopic) front-end
   ([`2026-09-16-retinotopic-sensing-v6-design.md`](2026-09-16-retinotopic-sensing-v6-design.md)).
5. **Prior art reads the descending escape pathway.** FlyDrones routes optic flow → T4/T5/LPLC2/LC4
   and reads **DNg02/DNp03/DNp01** to the sticks; fly.ai injects LPLC2/LC4/LC10a and reports the
   left-LC4+LPLC2 → left-giant-fibre-DNp01 laterality ([`external-prior-art.md`](../../external-prior-art.md)).
   Our bridge reads `escape` off giant-fibre **motor cells** and climbs.

## 3. Facts an implementer needs

- The declared codec lives in `python/fly_drone/adapter.py`. It reads only the environment's
  (possibly ablated) 2,022-vector of descending/VNC traces via `brain.readout_ids`
  (`escape`, `power_l/r`, `steer_l/r`, `wing_l/r`, `thrust`), and is calibrated on `STIMULI`
  (`_params` / `_params_v2`). Its identity is `_version(params, dataset_hash, visual, codec)`;
  `--codec v1|v2` selects committed artifacts and the default is v2
  ([`FINDING-2026-09-20-c3-and-codec-v2.md`](../../results/liveness/FINDING-2026-09-20-c3-and-codec-v2.md)).
- The bridge **may not** read `loom_l/r`, pose, target or pixels: those are inputs to the brain, not
  part of the feature vector. The escape _side_ must come from a neural readout (motor asymmetry, or
  a declared sided descending readout — C4c).
- Cell types and sides: `brain.cells[i]["type"]`, `["side"]` from `data/malecns/cells.json`; the
  loom sensory cells are the LC4 (165) and LPLC2 (146) per side (`sensory-model.md` §3). The free
  probe patterns already exist (`scripts/readout_audit.py`, `scripts/propagation_trace.py`,
  `scripts/spatial_lateralization.py`).
- Gates to reuse: `roam_eval.adapter_check` (A1–A7), `liveness-check` (L1–L8, L7 deferred), and the
  E1/E2 selectivity gates in `sensory-model.md` §6.

## 4. Workstream A — C4a: the bridge uses the escape response laterally

Goal: when the connectome's loom/escape response fires, command a **decisive lateral escape** with
the direction taken from the brain's own sided response, instead of a climb plus `0.5 · yaw`.

### A0 — feasibility probe (blocking, free)

Find a neural readout that carries the **escape side**. Candidates, in order:

1. a declared sided readout over the descending loom-escape cells (candidate types from the giant
   fibre / DNp01 family; confirm against `cells.json` and the LC4/LPLC2→DN route);
2. the existing wing steering motoneurons (`steer_l − steer_r`), which already flip on loom.

Measure, with the existing stimulus battery and a lateralised loom (e.g. the `loom_l`/`loom_r`
rows of `readout_audit.json`), that the sided readout separates left from right above the noise
floor. **If no sided neural signal exists, C4a is not implementable from neurons as specified** and
falls back to C4c (add the descending readout) before any codec change.

### A1 — the declared change (codec `v3`)

One fixed formula, one identity, silenceable via the existing ablation of the escape pathway:

- keep the current forward drive and the (declared) steering;
- replace the escape response's **climb-dominant** mapping with a lateral escape along the neural
  escape side: `vy_escape = ESCAPE_SIDE_GAIN · escape_latch · side_hat`, with `escape_latch` the
  existing `_latch(escape)` and `side_hat ∈ {−1, 0, +1}` from A0;
- keep a **declared, smaller** climb term (`CLIMB_FRACTION`) so a threat can still be climbed over;
- declare both constants and add them to the calibration battery (they are calibrated on declared
  stimuli, not fitted to any teacher).

The codec identity is `declared-v3:<hash>`; `adapter.json`/`adapter-v2.json` are untouched, and the
default codec becomes v3 only after it clears the gate.

### A2 — free falsifier (before any seeds)

Offline `scripts/command_audit.py` (extended): a lateralised loom must command `|vy|` at least the
teacher's near-threat level (~0.8) with the correct sign (away from the loom side), and a head-on
loom must command a lateral escape rather than only climb. Fail here → no seeds.

### A3 — gate (run 2026-09-22: **FAILED**)

- A3 near-threat: dodge ≥ 0.8, balanced ≥ 0.8, **ghost ≤ 0.3** (the causal test — a blind drone must
  not inherit the dodge).
- A4 loom: loom silencing must increase collisions (ratio ≥ 2.0).
- A2 collisions/min ≤ 0.5 (needs C4b for the wall share).
- No regression: liveness L1–L8 (L7 deferred) and A6 `slow_fraction` ≤ 0.1.

Result: dodge 0.341 (balanced 0.154) vs the 0.8 bar; ghost 0.205 **passes**; A4 1.45; A2 4.30/min;
A6 `slow_fraction` improved but coverage still 0.257; **liveness L6 regressed** (mean move bout
9.14 → 12.80 s). Verdict and analysis: `FINDING-2026-09-22-c4a-a3.md`.

### A4 — tests

Unit tests for the new readout's side sign and for the codec's command on a synthetic loom-left vs
loom-right vector; a test that the canonical artifacts stay byte-identical; a test that the additions
are silenceable (ablation → command reverts to the v2 behaviour).

## 5. Workstream B — C4b: a selective loom front-end (wall avoidance)

Goal: separate a closing threat/wall from a turn sweep, so avoidance is causal (A4) and walls are
avoided (A2).

### B0 — selectivity feasibility (free)

Run the documented E1/E2 gates on the current v4 front-end and on each candidate:

1. the **v6 spatial encoder** ([`2026-09-16-retinotopic-sensing-v6-design.md`](2026-09-16-retinotopic-sensing-v6-design.md)) — the documented fix;
2. the existing **P3 relay** (`relay=True`, optic flow → T4/T5) **extended to LC4/LPLC2**.

A candidate must clear E1 ≥ 0.8 (loom AUC on threat-positive frames) and E2 (the light/loom
margin), else it is not adopted. Report which cells carry the selective signal.

**Result (run 2026-09-22): neither candidate is adopted.** v4 E1 **0.738**; frozen v6 spatial clone
E1 **0.720** (motion 0.657, union 0.710); both fail the 0.8 bar. The relay route is structurally
infeasible: its T4/T5 targets do not propagate to LC4/LPLC2 (M1b ≤0.001) and a global-translation
model cannot represent looming. See `FINDING-2026-09-22-c4b-b0.md`. C4b is blocked on E1.

### B1 — the declared change

An additive, versioned front-end (bundle + identity), silenceable through the existing
`relay`/`sensory` ablations. This changes the _input_ mapping only; wiring, weights and neuron
parameters stay frozen.

### B2 — gate

E1/E2 as above, then A2/A4: wall collisions must fall and loom silencing must raise collisions.

## 6. Workstream C — C4c (optional): read the descending giant-fibre cells

If A0 finds no usable sided motor signal, add a declared sided readout over the descending
loom-escape cells (the prior art's DNg02/DNp03/DNp01 route) and re-run A0. This is a **readout**
addition (no change to the brain), versioned with the adapter identity.

## 7. The full gate (pre-registered)

Each addition is scored on the same battery, in order, before the next:

1. free falsifier (A2/B0) — no seeds spent until it passes;
2. 15-seed smoke on `adapter-check` (A1–A7) + `liveness-check`, direction only;
3. 50-seed `adapter-check` on the winning cell only, `ACCEPTANCE` unchanged, plus liveness L1–L8
   (L7 deferred on this body).

**Falsifiers (pre-registered):**

- C4a does not raise near-threat `|vy|` or A3 → the escape signal is not usable laterally; stop.
- C4b does not clear E1/E2 → loom stays non-selective; wall avoidance is not causal; stop.
- Either addition fails the ghost control (blind dodge > 0.3) → the behaviour is not visual; stop.
- Any addition regresses A6 `slow_fraction` or liveness → rejected.

## 8. Execution order

1. A0 feasibility (free) → decides A1 vs C4c.
2. A1 codec v3 + A2 offline falsifier (free).
3. B0 selectivity feasibility (free) → **run 2026-09-22: no candidate clears E1; C4b blocked**.
4. A3 gate → **run 2026-09-22: C4a rejected**; 15-seed smoke of a C4b cell is deferred until B0
   has an adoptable front-end.
5. 50-seed gate on the best cell; then decide the default codec. **Not reached.**

Per step: run tests, `ruff`, commit and push.

## 9. Invariants

- `roam_eval.ACCEPTANCE` A1–A7 and `liveness` L1–L8 are unchanged; both are re-run, not relaxed.
- Canonical `adapter.json`, `adapter-v2.json`, `adapter-check.json` and the bundle hash are
  byte-identical.
- The bridge reads only neural activity (no pose, target, task id or pixels); one formula; no mode
  switching; no teacher in the behaviour path.
- Every addition is declared, versioned, and silenceable, and is rejected by its own falsifier.

## 10. Artifacts

- This spec; the A0/B0 feasibility reports under `docs/results/adapter/`.
- `adapter-v3.json` (+ identity), the extended `scripts/command_audit.py` output, the E1/E2 report.
- `adapter-check-v3-*.json`, `liveness-check-v3-*.json`, and a FINDING closing each workstream.
