# Handoff: body-agnostic, fidelity-first cyborg (start at P0)

This is a self-contained brief for a fresh agent. Read it end to end, then read the documents it
names before touching code. The repository is `/home/gb/projects/fly-drone` (git, branch `main`,
working tree clean at commit `26b626b`). Its sibling `/home/gb/projects/fly-playground` shares the
same connectome/body ancestry and is referenced below.

## 0. Your immediate task

Execute **P0 — the declared body adapter** from the design spec (§4), and land it behind a new
teacher-free causal gate. Do **not** start P1/P2 until P0's gate is written and passing (or the
honest negative is recorded). P0 requires no contract change and no long training run.

Before writing code, read, in this order:

1. `AGENTS.md` — the agent brief and the rules that are easy to break.
2. `docs/superpowers/specs/2026-09-19-body-agnostic-fidelity-cyborg-design.md` — the new direction.
3. `docs/overview/README.md` — the goal contract, architecture, training, evaluation.
4. `docs/free-roam.md` — the current free-roam method you are replacing as the default.
5. `docs/results/encoder-v6/DEPRECATION.md` — why the learned bridge was deprecated.
6. `docs/external-prior-art.md` §1 and `docs/references.md` — what others do and what is missing.

## 1. The situation

This project is a research simulator in which a frozen MaleCNS (166,700-neuron) connectome is the
brain of a simulated quadrotor. Historically: camera pixels → an engineered encoder (v4, plus
learned v5/v6) → input cells → frozen LIF graph → 2,022 descending + VNC-motor traces → a learned
MLP decoder → `[vx, vy, vz, yaw_rate]` → a cascaded PID → four rotors. The illustrative fly in the
viewer has no effect on the drone.

The sibling `fly-playground` is where the same connectome drives a browser fly body; its Rust LIF
core (`crates/fly-sim`) and fly-drone's `crates/brain-core` are byte-identical copies, and
fly-drone's `web/src/fly/*` is a hash-pinned reference copy of playground's body equations. Do not
edit those copies.

## 2. The goal (documented and reframed)

The original documented goal was: _"The fly's connectome is the brain; the drone is its body — a
cyborg."_ As of commit `26b626b` the goal was generalised to be body-agnostic:

> The connectome is the brain; the body is an embodiment, not the intelligence. Today the body is a
> simulated quadrotor (the cyborg milestone); the fidelity reference is the fly's own biomechanical
> body (`flybody`/NeuroMechFly); the deployment target is a physical drone. Only descending/motor
> (and, once added, ascending) neurons touch the body adapter; the wiring stays frozen; any fitted
> dynamics or added feedback is a declared, versioned contract change; the body supplies reflexes,
> never decisions; every skill must be causal.

The contract clauses are unchanged and must be preserved: only neural activity reaches the bridge;
the connectome is frozen; one bridge, no mode switch; the body executes; skills are causal; labels
are honest. See `docs/overview/README.md` §1.

## 3. Why we pivoted (the drift)

The old default made the learned decoder, not the connectome, the brain:

1. **An engineered teacher injected the behaviour.** `python/fly_drone/teacher.py` blends four
   geometry-driven drives (evade/avoid/beacon/explore); DAgger + PPO/SAC fit a decoder to imitate
   it. The decoder learned the teacher; the connectome was a feature source.
2. **The connectome was open-loop about its own body.** No ascending/proprioceptive/haltere
   feedback, so the decoder absorbed all body mismatch and model error.
3. **The dynamics were the coarsest possible.** Uniform LIF signs, silent modulators, no per-neuron
   tuning.

Evidence of the failure: v5 and both v6 decoder rounds collapsed to blind flailing (visible dodge ≈
ghost dodge), and v6's learned encoder closed negative — see
`docs/results/encoder-v6/CONCLUSION-2026-09-18.md` and `SAC-ROUND2-2026-09-19.md`. The user's
directive: stop training the drone to fit our simulation; make the connectome genuinely in control.

## 4. The new direction (spec summary)

The spec is `docs/superpowers/specs/2026-09-19-body-agnostic-fidelity-cyborg-design.md`. Decisions
already made by the user:

- **Hybrid A+B**: do the fidelity-first method on the drone **now**; scope the fly body
  (`flybody`/NeuroMechFly, via fly-playground's articulation plan) as a **later** phase (Scope A).
- **Contract changes authorized**: ascending/proprioceptive feedback (C1) and
  connectome-constrained fitted dynamics (C2). Both must be **additive, versioned identities**
  (the `sign-v2` pattern), never edits to the canonical bundle.
- **Do not delete all prior work**: the RL/SAC/DAgger learned encoder+decoder code stays in the
  tree as a **deprecated, opt-in path**. v6 is the retained learned version; older v5 artifacts were
  removed.
- **Cleanup already done**: the `runs/` tree (5.5 GB) was deleted; accepted v4 room actors and the
  best v6 pair were preserved under `docs/results/actors/`.

The v1 waves, in order:

| Step | What                                        | Contract          | Compute               |
| ---- | ------------------------------------------- | ----------------- | --------------------- |
| 0    | Goal reframing + prior art + spec           | —                 | **done in `26b626b`** |
| 0b   | Cleanup + preservation + deprecation record | —                 | **done in `26b626b`** |
| 1    | **P0 declared adapter + teacher-free gate** | none              | calibration           |
| 2    | P1 ascending feedback                       | C1 (new identity) | probes                |
| 3    | P2 connectome-constrained dynamics          | C2 (new identity) | fit — ask first       |
| 4    | P4 transfer / domain randomization          | none              | training — ask first  |
| 5    | P3 visual relay                             | none              | probes                |
| 6    | P5 evaluation additions                     | none              | eval runs             |
| 7    | P6 Scope A fly body                         | goal broadening   | separate plan         |

## 5. Current repository state (important gotchas)

- `runs/` **does not exist**. Running `fly-drone baseline`, `train`, `serve --accepted`, `smoke`,
  etc. will recreate it; that is expected and `runs/` is git-ignored.
- Preserved, committed artifacts:
  - `docs/results/actors/v4-closed-s04/actor.json`, `docs/results/actors/v4-loom-ft/actor.json`
    (accepted room actors; `accepted-policies.json` now points here).
  - `docs/results/actors/v6/clone/encoder.pt` + `clone.json`, `docs/results/actors/v6/round0/`
    (`decoder.json`, gate/validation reports). `current-policies.json` marks this pair
    `deprecated`.
- `python/fly_drone/current_pointer.py` now points its `FROZEN_V6_*` constants at
  `docs/results/actors/v6/...`; its `FALLBACK_DECODER` still names a deleted v5 path and will simply
  never resolve (harmless; do not reintroduce v5).
- The learned decoder is **still the only implemented bridge**. There is no `adapter.py` yet. Until
  you land P0, `serve`/`evaluate` on free roam will load the deprecated v6 pair and do nothing
  useful (runs/ is gone). That is the gap P0 fills.
- All 295 Python tests passed at `26b626b`; ruff clean. Pre-existing prettier drift exists in some
  unrelated docs (`docs/superpowers/plans/*`, `docs/results/encoder-v5|v6/*`, `docs/training.md`,
  `docs/sensory-model.md`) — **do not mass-reformat those**; just keep your own files clean.

## 6. P0 in detail (your deliverable)

Goal: make the default body bridge a **declared, calibrated adapter with no teacher in the
behaviour path**, and prove it with a gate. Keep the learned path available but non-default.

Required pieces:

1. **`python/fly_drone/adapter.py` (new).** A declared action codec from neural activity to the
   body command `[vx, vy, vz, yaw_rate]`. Use a small set of named populations/readouts (see
   `data/malecns/groups.json` readout roles and `BrainRuntime` in `python/fly_drone/brain.py`) and
   calibrate on a stimulus battery (FlyDrones-style ridge; reuse `python/fly_drone/calibration.py`
   and `scripts/stimulus_battery.py` if useful). No learned policy, no teacher labels. The codec's
   constants are fixed at build/calibration time and stored with an identity.
2. **The body adapter stays declared.** `python/fly_drone/plant.py` (`DronePlant.advance`) is the
   non-learnable setpoint+PID path; do not add decision logic to it.
3. **A teacher-free causal gate.** The declared policy must run the existing free-roam evaluation
   machinery with **no teacher present anywhere** and be reported honestly. Reuse the causal
   structure in `python/fly_drone/roam_eval.py` (conditions, ablations, ghost, paired bootstrap).
   Add the gate as a new entry point/flag rather than weakening `roam_eval.ACCEPTANCE` (that is
   never relaxed). Recommended shape: `fly-drone adapter-check` (new subcommand in
   `python/fly_drone/cli.py`) that runs the declared adapter under the standard conditions and
   prints a pass/fail against the same A-criteria, explicitly labelled teacher-free.
4. **Keep the learned path.** `training.py`, `sac.py`, `distill.py`, `teacher.py`,
   `crates/brain-core/src/policy.rs` stay. Selection between declared and learned bridges must be
   explicit (CLI flag / config), never a silent fallback or scenario switch — the "one bridge, no
   mode switch" clause still applies within a run.
5. **Tests.** Add focused tests (`tests/test_adapter.py` and extend `tests/test_roam_eval.py` or
   similar): codec determinism, identity binding, that the declared path contains no teacher call,
   and that the gate fails loudly (not silently) when no behaviour emerges.

Exit criteria:

- `adapter-check` runs end-to-end on the simulated drone with the frozen connectome and full
  causal conditions, and prints a stable report.
- Either it passes the A-criteria (record the evidence under `docs/results/`), **or** it honestly
  records a negative ("the connectome + declared adapter cannot fly the drone"), with the numbers.
  **A negative is an acceptable P0 outcome and must not be hidden by leaning on the teacher.**
- Existing accepted v4 actors and `roam_eval.ACCEPTANCE` are untouched; all tests pass.

## 7. After P0 (context only — do not start yet)

- **P1 (C1):** add ascending/proprioceptive/haltere and optic-flow feedback as new annotated input
  roles, with a new `bundle_hash` identity and a silencing test. Candidate cells and reused input
  roles are named in spec §5 and `docs/connectome-navigation-findings.md` §2.
- **P2 (C2):** fit per-neuron leak/threshold (then optionally per-synapse scale) to Shiu/Lappalainen
  recordings with the wiring frozen; emit an additive bundle. Long compute — ask the user first.
- **P4/P3/P5/P6** per spec §7.

## 8. Rules that are easy to break (from AGENTS.md)

- `roam_eval.ACCEPTANCE` and pre-registered thresholds are **never relaxed** without the user.
- The legacy room MJCF and accepted actors stay reproducible (`test_legacy_room_mjcf_unchanged`).
- Body feedback and fitted dynamics are **additive, versioned identities requiring explicit user
  sign-off**; the canonical bundle and accepted actors never change.
- **Ask the user before long** data collection / DAgger / PPO / SAC / fit runs.
- **Commit and push after each completed step.**
- Keep parallel workers modest (≤ 6 on this dev machine).
- Do not edit the hash-pinned `web/src/fly/*` reference copies or the canonical `data/malecns`
  bundle. Fly-body work belongs in `fly-playground`.

## 9. Environment and commands

A ROS install pollutes `PYTHONPATH`; strip it:

```bash
cd /home/gb/projects/fly-drone

# tests + lint (the acceptance loop for every step)
env -u PYTHONPATH .venv/bin/python -m pytest -q
.venv/bin/ruff format python tests && .venv/bin/ruff check python tests

# web/docs formatting (only your files; see §5)
yarn --silent prettier --write README.md AGENTS.md docs/**/*.md python/...   # be targeted

# inspect the current CLI
env -u PYTHONPATH .venv/bin/fly-drone --help

# relevant existing commands
env -u PYTHONPATH .venv/bin/fly-drone evaluate --task free_roam --policy <actor.json>
env -u PYTHONPATH .venv/bin/fly-drone assay
env -u PYTHONPATH .venv/bin/fly-drone calibrate
```

Python tools run through `scripts/venv.sh` for the `yarn` scripts; activating the venv is only
needed to call `fly-drone` directly. `.venv/bin/python` also works.

## 10. Pitfalls and honest failure modes

- **Don't reintroduce the teacher to make a gate pass.** Teacher labels are training-only and
  deprecated as the default; a declared adapter that needs a teacher is not a declared adapter.
- **Don't "improve" the dynamics inside the decoder.** Any fitted dynamics is P2, additive and
  versioned.
- **The connectome may simply not be sufficient.** If the honest result is that a declared bridge
  cannot fly the drone, that is a legitimate, reportable finding — write it up with the causal
  numbers and stop; do not silently fall back.
- **The body is alien.** The quadrotor is not a fly; some engineered translation is unavoidable.
  Keep it declared, calibrated, fixed, and small — that is the whole point of P0.
- **Watch identity handling.** New bridges carry `encoder_version`/`bundle_hash`; the runtime
  rejects mismatches. Read `python/fly_drone/brain.py` and `python/fly_drone/identity.py` before
  adding an identity.
- **Two repos.** If you find yourself wanting to change the fly body, stop and check Scope A and
  `fly-playground/docs/2026-09-13-connectome-articulation-plan.md`; that is a separate plan.

## 11. Open decisions to confirm with the user (spec §12)

1. Where preserved actors live — currently `docs/results/actors/` (already chosen and committed);
   keep it.
2. P1 input granularity — labelled ascending cells where a defensible mapping exists, populations
   otherwise, each with a confidence label (recommended).
3. P2 fitting target — leak/threshold first; escalate to per-synapse scale only on evidence
   (recommended).

## 12. Definition of done for this handoff

- P0 code landed (`adapter.py`, CLI gate, tests), all tests green, ruff clean.
- `adapter-check` evidence committed under `docs/results/` (pass **or** documented negative).
- Spec updated only to tick P0 status; no contract clause changed beyond what the spec authorizes.
- `README.md`/`AGENTS.md`/`docs/overview/README.md`/`docs/architecture.md` remain consistent with
  the body-agnostic goal.
- Committed and pushed with a concise message.
- Stop before P1 and report: what P0 measured, whether the declared bridge flies, and the proposed
  P1 design for sign-off.
