# Frozen v6 pair — pin and command-path smoke (plan 08 Task 0)

**Date:** 2026-09-18. **Scope:** plan 08
[`2026-09-18-fly-drone-08-frozen-v6-free-roam.md`](../../superpowers/plans/2026-09-18-fly-drone-08-frozen-v6-free-roam.md)
Task 0. Docs + two small fixes; no training; `roam_eval.ACCEPTANCE` untouched.

## The pinned pair

| Half | Artifact | Identity |
| --- | --- | --- |
| Encoder (frozen) | `runs/v6/clone/encoder.pt` | `learned-v6:01280e414169ff9c` (light r 0.998, loom r 0.934 held-out) |
| Decoder (round 0) | `runs/v6/round0/decoder.json` | `encoder_version = learned-v6:01280e414169ff9c`, `dataset_hash = 60cb1821…`, export error 3.2e-6 |

Round-0 gate (30 seeds, level 2, 60 s, `runs/v6/round0/gate30-round0.json`): **beacons 2.00/min
PASS** (v4 reference 1.933, gate 1.547), collisions 1.0/min. E1 loom 0.720 (fail 0.8; v4 0.732),
E2 pass (frozen baseline, reported only).

Pair is **valid but not selective**; it is the best valid v6 pair and the base for decoder-only
free roam (Option 3).

## Smoke results (tiny settings, `runs/roam/v6-smoke/`)

| Command | Result |
| --- | --- |
| `roam-collect --encoder` | OK — 50 samples, all carrying `learned-v6:01280e414169ff9c` |
| `roam-fit --encoder` | **fixed** (see below) — held-out R² reported, export error 1.0e-6 |
| `roam-screen … --encoder` | OK |
| `sac-validate --decoder --encoder` | OK (E1 null at 2 s / no positive frames, expected) |
| `evaluate --task free_roam --encoder` | OK — A1–A7 + probes produced |
| `sac-round bypass --encoder` | OK |
| `sac-init-decoder` → `sac-round decoder --init` | OK — export error 4.3e-6 |
| `sac-export --learner decoder` | OK |

## Fixes landed (smoke broke these)

1. **`roam-fit` had no `--encoder`.** `distill.fit` always built a v4 `BrainRuntime`, so it rejected
   every learned-encoder `.npz` (`collected with encoder learned-v6:…, expected …-v4`). Added
   `distill.fit(..., encoder=None)` and the `roam-fit --encoder` flag; the loader now validates the
   recorded version against `brain.encoder_version`.
2. **`training.export_actor` hard-coded `encoder_version = ENCODER_VERSION` (v4).** A clone fit under
   a learned encoder exported an actor the runtime then refused to load. It now writes
   `brain.encoder_version`; for a v4 brain this is `ENCODER_VERSION`, so legacy exports are
   byte-identical.

## Plan 08 corrections (mechanics)

- **Task 1:** `roam-fit <npz> --encoder runs/v6/clone/encoder.pt --output …` now works as written.
- **Task 3:** screens are unchanged; `roam-fit` for each iteration needs `--encoder`.
- **Task 4 warm start:** the DAgger actor's `warm-ppo.zip` **cannot** be loaded as a decoder-round
  `--init`: its observation space is `Box(2022)` while a decoder round under a v6 encoder uses
  `Dict(dn, geometry)`. The correct path is
  `sac-init-decoder <npz> --encoder <clone.pt> --output <init>` (which builds the SAC spaces and
  clones the labels), then
  `sac-round decoder --encoder <clone.pt> --init <init>/decoder.zip`.
- **Decoder anchor:** `sac-round decoder` currently has no anchor (`sac.anchor_penalty` applies to
  the spatial encoder only). The behaviour-cloning anchor agreed for the decoder round is plan 08
  Phase 4 work, not yet implemented.
