# Frozen-v6 free-roam phase — summary (2026-09-19)

**Scope:** plan 08 — a free-roam actor whose only learner is the decoder, on the frozen v6 pair
(encoder `learned-v6:01280e414169ff9c` + `runs/v6/round0/decoder.json`), with the harvested causal
controls. The connectome and encoder stay frozen; `roam_eval.ACCEPTANCE` and the E1/E2 bars were
never changed.

**Verdict:** **no accepted free-roam actor.** The honest negative result of the whole v6 line stands:
the learned encoder is closed (phase 1), and decoder-only RL on the frozen pair does not learn a
causal dodge — it collapses foraging. The best valid artifacts are the frozen v6 pair and the
DAgger students `it2`/`it3t`.

## What worked

| Item | Evidence |
| --- | --- |
| Frozen v6 pair pinned and loadable end-to-end | `FROZEN-PAIR-2026-09-18.md`; all command paths smoked |
| Two real gaps found and fixed | `roam-fit --encoder` (was v4-only) and `training.export_actor` encoder identity (was hard-coded v4). Both v4 paths bit-identical |
| **E3 bypass gate (G2): passes** | bypass reads the 816 currents but does not beat the full pair (beacons 1.2 vs 1.8, collisions 3.3 vs 2.5). `E3-BYPASS-2026-09-18.md` |
| **Threat curriculum works directionally** | threat-only DAgger (`it3t`) gave the first **causal dodge** signal: dodge 0.31, ghost 0.14, gap +0.17, vs ≤0.05 for all earlier students. `DAGGER-2026-09-18.md` |
| Reusable tooling, tested | potential-based **threat shaping** + level-4 "threats without pillars" (`7c1e64e`), `sac-round --level` (`fdfe0de`), in-loss **BC anchor** (`4f24cdd`), reward decoder anchor (`45fc4d4`) |
| Round guard did its job | every collapse was caught by `roam_eval.round_eligible` (beacons, ghost, E2); nothing degenerate was accepted |

## What did not work

| Route | Outcome |
| --- | --- |
| G0 realisability at the literal 0.5 bar | fails (avoid 0.29, beacon 0.41, explore −1.13, threat unmeasurable/negative). `explore` is random by construction and the v5 DAgger never met 0.5 either — recorded as a **diagnostic**, not a gate (`REALISABILITY-2026-09-18.md`) |
| DAgger 0–3 (level mix 0–3) | foraging holds (~1.7/min) but **no causal dodge**; the threat drive is only ~1% of frames; the pure-student iter 3 collapses foraging to 0.5 (`DAGGER-2026-09-18.md`) |
| Decoder SAC `sac1` (reward anchor 0.2, from it3t) | beacons **0.00**, ghost 0.33, `round_eligible=false` (`SAC-ROUND1-2026-09-18.md`) |
| Decoder SAC `sac2` (dense threat shaping, from it2) | beacons **0.30**, ghost 0.54 (blind flail); shaping slowed but did not stop the collapse (`SAC-ROUND2-2026-09-19.md`) |
| Decoder SAC `sac3` (in-loss BC anchor, 60k pilot) | **stopped by request**; early trace after unfreeze (−150 at 9k → −376 at 18k) was not promising. Inconclusive, no validation |
| v6 learned encoder | already closed negative before this phase (`CONCLUSION-2026-09-18.md`) |

**Mechanism:** the actor is frozen for the warm-up; degradation always begins when it unfreezes,
i.e. the RL actor update destroys the clone. A reward-side anchor was too weak; shaping did not
change the failure; the in-loss BC anchor is the untested remaining fix (pilot inconclusive). The
free-roam reward also has no positive term for dodging, and threats are sparse — dense shaping
alone did not rebalance it.

## Current wiring (defaults)

- `docs/results/current-policies.json` → **frozen v6 pair**: encoder `runs/v6/clone/encoder.pt`
  (`learned-v6:01280e414169ff9c`) + decoder `runs/v6/round0/decoder.json`, status `interim`,
  gate report `runs/v6/round0/gate30-round0.json`. `current_pointer.scan` now prefers the v6 pair
  over the v5 `it0` fallback (`45fc4d4`… this commit; test added).
- `docs/results/accepted-policies.json` is **unchanged** — only the v4 `visual` and `looming`
  actors. No free-roam actor is accepted.
- E1 (frozen) loom 0.720 / union 0.715, reported only; E2 passes.

## Data removed (phase cleanup)

The large transient/failed artifacts were deleted (git-ignored, regenerable): `runs/v6/joint`,
`round1*`, `round0-data-*`, `smoke`, `clone-seedmotion`/`clone-test`; `runs/roam/v6-d*.npz`,
`v6-realise*.npz`, `v6-smoke`, `v6-sac1/2/3`, `v6-bypass`. `runs/` shrank 20 GB → 5.5 GB. Kept:
the frozen pair, the small DAgger students (`v6-it0…it3`, `it3t`) and their screens/fits, and all
result records. `runs/v5` was left untouched.

## Recommended next steps

1. **In-loss BC pilot was stopped mid-run** — the one untested, targeted fix. If free roam is
   resumed, finish it (or use a KL variant) before concluding decoder-only RL cannot protect the
   clone.
2. **Reward, not just shaping:** add a positive dodge term or use time-to-contact so the policy is
   not forced to trade foraging against collisions.
3. **Better anchor target:** `it3t` dodges but forages weakly (0.80); `it2` forages (1.6) but does
   not dodge. A student that does both is the prerequisite for anchoring without a trade-off.
4. **Fallback already available:** `it2` is `round_eligible` (beacons 1.60, ghost 0.15, E2 pass) —
   a free-roam v1 at levels 0–2 (forage + obstacle avoidance, no thrown threats) could be accepted
   as a scoped exception, with A3 deferred.
5. **Do not revive v6 sensing** without the real cell↔ommatidium join.

## Commits this phase

`0b10273` M5 spec+plan · `636c086` pair pin + learned-encoder fixes · `1dc2362` G0 record ·
`02174a2` DAgger driver · `d8c4a3b` DAgger result · `a48edc5` threat DAgger ·
`0d47715` threat DAgger result · `45fc4d4` decoder reward anchor · `3230b7d` plan correction ·
`210525d` E3 bypass result · `b1a5ce9` sac driver · `ba96ef9` sac1 result · `7c1e64e` threat
shaping + level 4 · `fdfe0de` `--level` + shaped driver · `463cb4b` sac2 result · `4f24cdd`
in-loss BC anchor · `2e38b12` BC pilot driver.
