# Gated joint encoder+decoder training (M4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Status:** draft for user review (2026-09-17). Not approved; no code written.

**Goal:** Train the v6 spatial encoder and the decoder **together** in one SAC loop, so neither is a
frozen partner for a whole round, and prove the pair has gained causal skill **without** turning the
frozen connectome into a wire. This is milestone M4 of the retinotopic sensing spec
([§5, §8](../specs/2026-09-16-retinotopic-sensing-v6-design.md)).

**Why now.** The alternating scheme (plan 06, Task 11) failed five times: the frozen partner makes the
learning signal for the moving half weak and indirect, and every encoder round drove the encoder's
output to a near-constant (saturate, centre, or flat) despite an anchor and a predictive auxiliary
objective. The v6 spec already names M4 as the fix — "removes the alternating non-stationarity —
**only** with E1/E2 and E3 as hard anti-wire gates".

**Spec:** [`2026-09-16-retinotopic-sensing-v6-design.md`](../specs/2026-09-16-retinotopic-sensing-v6-design.md)
§5 (training), §6 (evaluation), §8 (sequencing). Read plan 06 first.

## Global constraints

- The connectome is frozen and **non-differentiable**: no gradient flows through it, and no wiring,
  weight, sign or parameter changes. Joint training means both actors move in the same loop, not a
  backprop through the brain.
- Interfaces are unchanged: the encoder's only input is its own eye stack; the decoder's only input
  is the 2,022 DN + VNC traces. No pose, target or pixels reach the decoder.
- **Anti-wire gates are hard.** Joint training is only accepted if E1, E2 **and** E3 hold; a pair
  that wires the decoder to the encoder through the brain is rejected however good its reward.
- `roam_eval.ACCEPTANCE` and E1/E2 bars are never relaxed. E1 is read on the **motion ∪ loom**
  union for v6, with loom-only and motion-only reported alongside (added `dfc1ed4`).
- v4/v5 reproduce bit-for-bit; the legacy room test stays green.
- Workers ≤ 6; `env -u PYTHONPATH .venv/bin/python …`; ask the user before every long run; no
  `fly_drone/*.py` edits while a `sac-round`/`roam-*` process is live.
- Commit and push after each task; no agent trailer.

## Design

One SAC model whose **actor has two independent heads** reading two different observations:

```
                eyes (6x48x64)                dn (2022) + geometry
                     |                              |
        [SpatialEncoderNet]                [KeyNormalizer(dn) -> MLP]
                     |                              |
             816 logits (currents)           latent -> 4 velocity means
                     \_____________________________/
                                    |
                        action = [currents(816), velocity(4)]   (820)
                                    |
   env: currents -> FROZEN CONNECTOME -> DN traces -> (obs for next step)
        velocity -> plant PID -> rotors
```

- **`JointActor`** (new, `python/fly_drone/joint_policy.py`): keeps two feature paths and two mean
  heads, concatenates `mu` and clamped `log_std`, so SB3's `SquashedDiagGaussianDistribution` samples
  the full 820-dim action. The encoder path reuses `SpatialFeaturesExtractor`; the decoder path reuses
  `KeyNormalizer` on `dn` (the same standardisation `set_dn_stats` writes).
- **`JointSACPolicy(AsymmetricSACPolicy)`**: `make_actor` returns `JointActor`; `make_critic` keeps
  `CriticExtractor` (training-only, sees all keys). One critic, one reward — the coupling is the
  point.
- **`JointRoamEnv`** (new, in `sac.py`): an external-v6 brain (patch roles, currents set by the
  action) plus the plant driven directly by the action's last 4 dims (the learned decoder replaces
  `brain.infer`). Observation is `{eyes, dn, geometry}` exactly as the alternating encoder env.
- **Stabilisers carry over:** the predictive objective (predict next DN from current DN + currents)
  and the strong anchor to the round-0 clone, both on the encoder path only.
- **Entropy regime:** the action is 820-dim, so `ent_coef_init(820)` reuses the dimension scaling
  from `sac.py` (≈9.8e-5). **Open decision O2** below: one α for both heads under-explores the
  4 velocity dims; a per-head α needs a custom actor loss.
- **Export:** one run yields both artifacts. `sac-export --learner joint` writes `encoder.pt`
  (`SpatialEncoder.from_actor`) and `decoder.json` (`export_decoder` on the decoder head), pinning
  the same `learned-v6:` version.

## Gates (all must hold)

| Gate | What | Source |
| --- | --- | --- |
| G1 foraging | round guard: beacons ≥ ½ round 0, ghost ≤ 0.3, E2 pass | `roam_eval.round_eligible` |
| G2 E1 | union AUC strictly > the best alternating round's union AUC | pre-registered 0.8 reported |
| G3 E2 | light margin ≥ 0.05, loom margin ≥ 0.1 | `ENCODER_CHECKS` |
| G4 E3 | a bypass decoder reading the currents is **not** better than the full joint pair | `roam_eval.bypass_comparison` |
| G5 joint > alternating | the joint pair beats the alternating pair on the guard **and** G2, without failing G3/G4 | v6 spec §8 M4 |

If any gate fails, revert to the alternating round-0 pair (`runs/v6/round0`, gate-passing) and record
the failure; no threshold moves.

## File structure

| File | Responsibility | Change |
| --- | --- | --- |
| `python/fly_drone/joint_policy.py` | `JointActor`, `JointSACPolicy`, action split helpers | create |
| `python/fly_drone/sac.py` | `JointRoamEnv`, `joint` learner in `learner_spaces`/`build_sac`/`export_checkpoint`, entropy regime | modify |
| `python/fly_drone/spatial_policy.py` | reuse `SpatialFeaturesExtractor`; expose the encoder net for the joint actor | modify |
| `python/fly_drone/cli.py` | `sac-round joint`, `sac-export --learner joint` | modify |
| `tests/test_joint.py` | policy shapes, action split, env step, one train step, export round-trip | create |
| `docs/results/encoder-v6/` | joint run results, E1/E2/E3 report | create |

## Tasks

### Task 1: Joint policy and action space
- [ ] `JointActor`: encoder head (eyes → 816 logits) + decoder head (dn → 4 means), one clamped
      `log_std` per part; `get_action_dist_params` concatenates. Tests: mean shape (N, 820); changing
      `eyes` moves only the first 816, changing `dn` only the last 4; a fresh policy round-trips
      through `save`/`load`.
- [ ] Commit: `Add a two-head joint actor for gated joint training`.

### Task 2: `JointRoamEnv`
- [ ] Split the 820 action: `currents = action[:816]`, `velocity = action[816:]`; feed currents to the
      external-v6 brain, velocity to the plant. Observation `{eyes, dn, geometry}`; metabolic cost as
      in the encoder env. Tests: action split; equal-split currents reach the same cells as the
      encoder env; a step runs and returns the right observation keys.
- [ ] Commit: `Add the joint encoder+decoder environment`.

### Task 3: Training, export and load plumbing
- [ ] `learner_spaces("joint")`, `build_sac("joint", ...)` → `JointSACPolicy`, `ent_coef_init(820)`;
      `train_round(learner="joint")` with the anchor + predictive objective on the encoder path;
      `sac-export --learner joint` writing both artifacts. Tests: one `train` step changes both heads;
      save/resume; export parity for the decoder and a loadable `learned-v6:` encoder.
- [ ] Commit: `Train and export the joint encoder+decoder`.

### Task 4: Smoke
- [x] 9k-frame joint round, 6 workers: exited 0; `predictive_loss` 0.219, `ent_coef` 1.4e-4
      (dimension-scaled), anchor = clone, both heads warm-started. Commit: `Add the joint-training
      smoke result` (`edce7d3`, `babdc4a`).

### Task 5: One long joint run (gated, needs user approval)
- [ ] Run to a pre-agreed budget (recommend 300k frames); `sac-validate`; round guard.
- [ ] **Gates G1–G3.** Stop and report if the guard or E2 fails.

### Task 6: E3 bypass and the comparison (G4, G5)
- [x] **6a — v6 bypass support.** `learner_spaces`/`SacRoamEnv` carry the flat 816 currents for a
      bypass round under a v6 encoder (`spatial_inputs`), `_infer_spatial` recognises a v6 encoder
      for `bypass`, `CriticExtractor` sizes the currents key from the space, and `distill`'s
      `bypass:` controller accepts `SpatialEncoder` and flattens the currents. Commit:
      `Support the brain-bypass control on the v6 currents`.
- [ ] **6b — run E3:** `sac-round bypass` on the joint encoder; `roam-screen` full vs bypass;
      `bypass_comparison`. Then compare against the best alternating pair on guard + E1 union.

### Task 7: Report
- [ ] Write `docs/results/encoder-v6/JOINT-<date>.md` with G1–G5, all E1/E2 numbers (union/loom/
      motion), and the honest verdict (accepted / reverted).

## Open decisions (need the user)

1. **O1 — Anti-wire gate strictness.** Is "union E1 strictly beats the best alternating round" the
   right G2, or do you want a margin? And must G4 (bypass not better) hold jointly with G2, or is a
   bypass tie acceptable?
2. **O2 — Entropy across the two heads.** One α over the 820-dim action is dimension-scaled but
   under-explores the 4 velocity dims. Options: (a) single scaled α (simplest, may starve the decoder
   head); (b) separate α per head via a custom actor loss (more code); (c) warm-start the decoder head
   from the round-0 decoder and freeze it for the first N frames. Recommend (a) first, with (b) if the
   decoder head does not move.
3. **O3 — Budget and kill rule.** Frames per run, and the exact condition that reverts to the
   alternating round-0 pair.
4. **O4 — Keep the anchor/predictive?** They stabilise the alternating encoder; in joint mode the
   decoder also moves, so the anchor's target (the round-0 clone) may be too conservative. Recommend
   keeping the predictive objective and a weaker anchor (`ANCHOR_WEIGHT=0.1`).

## Decisions (adopted from the recommendation, user 2026-09-17)

- **E1 group:** read on the **union (motion ∪ loom)**; report loom-only and motion-only alongside;
  the 0.8 bar and the pre-registered loom `passed` are unchanged.
- **O1:** G2 = union E1 **strictly beats the round-0 union** (baseline to be measured by re-running
  `encoder-checks` on round 0); G4 (bypass not better) must hold jointly with G2.
- **O2:** (a) single dimension-scaled α over the 820-dim action first; switch to a per-head α only if
  the decoder head fails to move after warm-up.
- **O3:** one joint run of **300k frames**; revert to `runs/v6/round0` (and record the failure) if any
  of G1–G5 fails.
- **O4:** keep the predictive objective; weaken the anchor to `ANCHOR_WEIGHT=0.1`.
- **Task 12** (curriculum) stays after M4.

## Risks

| Risk | Mitigation |
| --- | --- |
| Encoder wires the decoder through the connectome | G2/G3/G4 hard gates; bypass comparison; reject on fail |
| One α starves the decoder head | O2; check both heads move after warm-up |
| Joint drift/collapse | predictive objective + anchor (as in plan 06, which passed the 100k gate) |
| Long-run cost | smoke first; ask before the run; ≥6 workers cap |
| Non-stationarity returns | one critic, one reward; both heads updated every gradient step |
