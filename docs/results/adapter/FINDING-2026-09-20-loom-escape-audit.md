# Why loom/escape fails: the brain senses the threat, the bridge doesn't use it

**Status:** recorded 2026-09-20. Free audit, level 3, codec v2 vs the accepted RL teacher (3 × 120 s
each). No threshold or bridge changed. Follows the v2 acceptance failure
([`FINDING-2026-09-20-v2-acceptance.md`](FINDING-2026-09-20-v2-acceptance.md)).

## 1. The loom pathway itself works

Per-frame sensors during free roam (max of `loom_l`/`loom_r`), the connectome's own `escape`
readout, and the emitted command, aggregated over threats that came within 2.5 m:

| | v2 seed 0 | v2 seed 1 | v2 seed 2 | teacher 0 | teacher 1 | teacher 2 |
| --- | --- | --- | --- | --- | --- | --- |
| near threats | 1 | 2 | 3 | 5 | 5 | 4 |
| near-dodge rate | 0.0 | 0.5 | 1.0 | 1.0 | 1.0 | 0.75 |
| near peak `loom` | 2.0 | 2.0 | 2.0 | 1.99 | 1.91 | 1.95 |
| near peak `escape` | 0.87 | 0.93 | 0.96 | 0.96 | 0.94 | 0.92 |
| near peak `vz` command | 1.0 | 1.0 | 1.0 | 0.6 | 0.8 | 1.0 |
| **near peak `vy` command** | **0.24** | **0.38** | **0.39** | **1.0** | **1.0** | **1.0** |
| collisions (120 s) | 16 | 14 | 9 | 1 | 0 | 2 |
| of which wall | 14 | 12 | 9 | 0 | 0 | 0 |

The dark-area loom cue saturates and the **connectome's escape readout reaches ~0.9 in both the
declared bridge and the teacher** — the LC4/LPLC2 → giant-fibre route fires. Sensing and the brain's
loom response are not the problem.

## 2. What actually differs

1. **The bridge under-commands the lateral escape.** The v2 codec's sidestep is
   `vy = side_fraction · yaw` with `side_fraction = 0.5`, so lateral command is a fraction of the
   *steering* command and caps near ~0.4, while the teacher commands full `vy = 1.0`. The brain's
   escape is present but the bridge turns it into a **climb** (`vz` peaks at 1.0) plus a weak
   steering flip — the wrong axis for a horizontally-aimed intercept.
2. **There is no obstacle avoidance at all.** v2 collides 9–16 times per 120 s and **all but one or
   two are walls**; the teacher touches nothing. A wall ahead gives a symmetric (steer ≈ 0) loom
   response, which the codec maps to climb, not a turn, so the drone flies into walls. Wall contacts
   dominate A2 (6.67 collisions/min) and swamp A4's loom-silencing test — silencing loom barely
   changes wall-driven collisions, so the ratio stays ≈ 0.9.
3. **The loom cue is not threat-selective.** It saturates at 2.0 on turn-sweeps/edges (28–30 % of
   frames ≥ 0.22 even between threats), matching the documented limit in
   [`sensory-model.md`](../../sensory-model.md) §2/§6: the v4/v5 8-scalar loom cue caps the E1
   selectivity AUC at 0.686 (< 0.8), and Mi1/Tm3/T4/T5 do not propagate to the loom circuit.

## 3. What the prior art and docs suggest

- [`sensory-model.md`](../../sensory-model.md) §2: the loom cue is a hand-built dark-area-growth
  scalar injected into LC4/LPLC2; the documented fix for its poor selectivity is the **v6 spatial
  (retinotopic) front-end**, so the optic lobe computes motion and looming itself.
- [`external-prior-art.md`](../../external-prior-art.md): **FlyDrones** routes optic flow →
  T4/T5/LPLC2/LC4 and reads the **descending giant-fibre neurons DNg02/DNp03/DNp01** (not motor
  cells) to the sticks; **fly.ai** injects LPLC2/LC4/LC10a and reports the left-LC4+LPLC2 →
  left-giant-fibre-DNp01 laterality. Both read the *descending* escape pathway and use it for
  steering, not climb.
- Our bridge reads `escape` off giant-fibre **motor cells** and weights it (with `side_fraction ·
  yaw`) into a climb-dominant command. The evidence above says the downstream use, not the upstream
  signal, is what fails.

## 4. Proposed C4 (declared, additive, versioned, silenceable)

1. **C4a — use the escape/loom response laterally.** Map the connectome's loom/escape and its side
   (via the neural steering/loom response) to a decisive lateral escape, so a looming object is
   steered *away from* rather than climbed over. Falsifier: near-threat peak `|vy|` and A3
   dodge/balanced/ghost.
2. **C4b — restore loom selectivity.** A declared spatial front-end (the documented v6 route, or the
   P3 relay extended to LC4/LPLC2) so the loom signal separates threats from turn-sweeps. Falsifier:
   the documented E1/E2 selectivity gates, then A4 and wall collisions.
3. **C4c (option) — read the descending escape neurons** (the DNp01/giant-fibre descending cells)
   instead of, or alongside, the motor-cell aggregate, as the prior art does.

Evidence ranks **C4a first** (the signal already exists and the teacher proves the body can execute
it); C4b is required before wall avoidance can be causal. Each addition needs a spec and sign-off;
`ACCEPTANCE` stays untouched.
