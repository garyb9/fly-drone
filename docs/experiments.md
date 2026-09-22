# Managed local experiments

The approved [roadmap](superpowers/specs/2026-09-22-fidelity-first-roadmap-and-agent-handoff.md)
authorizes two hours per run, six experiment-hours per stage, and at most six workers total.
The supervisor serializes experiments across output roots in this repository. Child programs
must honor their declared worker count; this is not a CPU or memory container.

Start the registered diagnostic in a terminal (foreground supervision):

```bash
env -u PYTHONPATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBGL_ALWAYS_SOFTWARE=1 \
  .venv/bin/fly-drone experiment start docs/results/roadmap/s1-a1-pilot-manifest.json
```

In another terminal, inspect or stop it:

```bash
env -u PYTHONPATH .venv/bin/fly-drone experiment status s1-a1-pilot
env -u PYTHONPATH .venv/bin/fly-drone experiment stop s1-a1-pilot
```

Status lives in `runs/experiments/<run-id>/status.json`, updates during execution, and includes
the active trial, completed count, elapsed seconds and last heartbeat. Trial logs and progress
events are adjacent. Each A1 trial log additionally prints completed scenario counts. Complete
trial JSON is in `runs/roadmap/S1/`; these larger trajectories are not committed.

After an explicit resume decision:

```bash
env -u PYTHONPATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBGL_ALWAYS_SOFTWARE=1 \
  .venv/bin/fly-drone experiment resume s1-a1-pilot
```

Completed trials are hash-checked and skipped. Partial output is preserved; interrupted trials
restart. Resume refuses changed source/artifacts/manifests. Cumulative time is not reset. An
unexpected supervisor death requires inspection of owned workers before any restart; do not
delete the ledger to evade this check. Keep the source tree unchanged during a run; evaluate
changes in a new identity rather than mixing revisions in one report.

The pilot contains development seed 0, the development manifest seeds 1–7, and confirmation seeds
8–15. All three share stage S1 and its aggregate budget. No automatic chaining is installed:
the supervising agent reports between runs and launches the next only after checking the result.
A user stop cancels this chain until explicitly resumed.

The A1 manifests intentionally have no automatic capability criterion. After every trial completes,
the supervisor reports `incomplete` with `execution complete; scientific decision requires review`.
This distinguishes collected diagnostic evidence from scientific acceptance; inspect completed count
and reason. Nonzero subprocess exits, missing outputs, cancellation and exhausted budgets are
recorded separately. Successful Python execution never implies that the drone can seek a beacon.

## Manifest contract

Required keys: `run_id`, `stage`, `hypothesis`, `decision_rules`, `identities`, `trials`.
Optional limits: `workers` (1–6), `max_seconds` (>0, ≤7200), `cwd`. Each trial has a unique `id`,
an argument-list `argv` (no shell), and a new JSON `output` path. An optional `criterion` names a
dotted JSON `field`, `op` (`eq`, `gte`, `lte`) and `value`; its meaning must be preregistered.
Never point output at an existing artifact. Source and declared artifact identities are captured
before execution, checked at completion and required to match on resume.

## Evaluation completeness

New `adapter-check`/free-roam evaluation reports retain unchanged `acceptance` calculations and
add `provenance` and `completeness`. Use `completeness.eligible_for_promotion` for promotion,
not `acceptance.passed` alone. Missing conditions/seeds/probes/near-threat sides, changed source,
or absent live A7 timing prevent promotion. Offline throughput is not live end-to-end timing.
