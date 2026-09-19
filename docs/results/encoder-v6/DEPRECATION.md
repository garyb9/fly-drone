# Encoder v6 (learned encoder) — deprecated

**Status:** deprecated 2026-09-19. The learned encoder + decoder path is **retained in code** for
reference and possible part reuse, but it is **no longer the default bridge** and no longer
receives new training. The default bridge is now the declared, calibrated adapter (spec P0) in
[`../../superpowers/specs/2026-09-19-body-agnostic-fidelity-cyborg-design.md`](../../superpowers/specs/2026-09-19-body-agnostic-fidelity-cyborg-design.md).

## Why it was deprecated

The v6 goal was a learned retinotopic encoder that raised loom selectivity (E1) above the
pre-registered 0.8 bar by driving the optic lobe with 816 directional currents instead of v4's 8
scalars. It closed **negative**:

| Failure                                                                                     | Record                                                                      |
| ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| E1 loom selectivity 0.720 (bar 0.8; v4 0.732) — no improvement                              | [`CONCLUSION-2026-09-18.md`](CONCLUSION-2026-09-18.md)                      |
| Alternating encoder SAC collapsed (saturated currents / near-constant)                      | `RESULTS-2026-09-16.md` §6–§7                                               |
| Joint training drifted / collapsed under both α regimes                                     | [`JOINT-2026-09-17.md`](JOINT-2026-09-17.md) §1–§7                          |
| Both decoder-only rounds destroyed the clone → blind flailing (visible dodge ≈ ghost dodge) | [`SAC-ROUND2-2026-09-19.md`](SAC-ROUND2-2026-09-19.md)                      |
| The second retinotopic axis has no visual meaning (no cell↔ommatidium join in the data)     | [`CONCLUSION-2026-09-18.md`](CONCLUSION-2026-09-18.md) "structural finding" |

The deeper reading, and the reason for the pivot: the learned bridge was absorbing the mismatch
between a coarse, open-loop LIF model and an alien quadrotor body, so RL fitted the simulator rather
than exercising the connectome. The body-agnostic spec replaces that with a declared bridge, body
feedback (C1) and connectome-constrained dynamics (C2).

## What is retained in code

- `python/fly_drone/encoder.py` (`LearnedEncoder`, v5), `spatial_encoder.py` (`SpatialEncoder`, v6),
  `retinotopy.py` (maps, identities), `spatial_policy.py`, `spatial_clone.py`, `joint_policy.py`,
  `predictive.py`, and the encoder/clone/joint paths in `sac.py`.
- CLI: `encoder-collect`, `encoder-clone`, `sac-round encoder|joint|bypass`, `spatial-maps`,
  `encoder-checks`.
- Tests: `tests/test_brain_v5.py`, `test_brain_v6.py`, `test_encoder.py`,
  `test_spatial_encoder.py`, `test_retinotopy.py`, `test_spatial_clone.py`, `test_spatial_policy.py`,
  `test_sac_v6.py`, `test_joint.py`.

Removing any of this is out of scope: it stays as a deprecated, opt-in path.

## Preserved artifacts

The best valid v6 pair (valid but not selective) is preserved under
`docs/results/actors/v6/` so `serve --encoder` and replay still work after the `runs/` cleanup:

| Half              | Path                                               | Identity                                        |
| ----------------- | -------------------------------------------------- | ----------------------------------------------- |
| Encoder           | `docs/results/actors/v6/clone/encoder.pt`          | `learned-v6:01280e414169ff9c`                   |
| Decoder (round 0) | `docs/results/actors/v6/round0/decoder.json`       | `encoder_version = learned-v6:01280e414169ff9c` |
| Gate report       | `docs/results/actors/v6/round0/gate30-round0.json` | 30-seed level-2 gate PASS 2.00                  |

`accepted-policies.json` still points at the v4 room actors; `current-policies.json` now marks the
free-roam `learned-v6` pair `deprecated`.

## If v6 is ever revived

The prerequisite is the real cell↔ommatidium join (e.g. FlyWire optic-lobe column annotations), not
another training scheme — see [`CONCLUSION-2026-09-18.md`](CONCLUSION-2026-09-18.md) "Consequence".
