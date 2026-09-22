# Fly / Drone — agent brief

## The goal (read this first)

**The fly's connectome is the brain; the body is an embodiment, not the intelligence.** Today the
body is a simulated quadrotor — the cyborg milestone. The fidelity reference is the fly's own
biomechanical body (`flybody`/NeuroMechFly); the deployment target is a physical drone. The full
MaleCNS connectome (166,700 neurons, frozen) should fly that body: the body moves because of what
its descending and motor neurons do. The body never runs a hidden script, planner or mode switch
that makes the decisions for it. The direction and its staged contract changes are specified in
[`docs/superpowers/specs/2026-09-19-body-agnostic-fidelity-cyborg-design.md`](docs/superpowers/specs/2026-09-19-body-agnostic-fidelity-cyborg-design.md).

What that means in practice, and what every change must preserve:

| Principle                 | Concretely                                                                                                                                                                                                                                                                                                                                                                                             |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| The brain decides         | The only input to the body bridge is neural activity (2,022 descending + VNC motor traces). No pose, task id, target position or pixels.                                                                                                                                                                                                                                                               |
| The body executes         | The body supplies reflex-level stabilisation (PID, mixer, rotors) — the role the fly's own VNC and halteres play. It never chooses where to go.                                                                                                                                                                                                                                                        |
| The brain stays the fly's | Wiring, weights, signs and neuron parameters are frozen. Learning lives in the bridge. The default bridge is a **declared adapter** (a calibrated codec, no teacher); a learned decoder/encoder path is retained but deprecated (v6 = per-patch maps; see `docs/sensory-model.md` §6–§7). Any fitted dynamics or added body feedback is a declared, additive, versioned contract change, not training. |
| One brain, one bridge     | One bridge for free roam. No switching decoders by scenario or state.                                                                                                                                                                                                                                                                                                                                  |
| Behaviour must be causal  | A skill counts only if silencing the pathway that carries it (vision, loom, light, feedback) removes it, and a blind (ghost) condition cannot pass by chance.                                                                                                                                                                                                                                          |
| Honest labels             | Any label uses simulator geometry only for what the eyes could actually see.                                                                                                                                                                                                                                                                                                                           |

Where we are going: foraging, obstacle avoidance and threat dodging in open, unseen arenas, all
attributable to neurons, with no engineered teacher in the behaviour path (free roam,
`docs/free-roam.md`); then richer senses and feedback into the connectome; then a fly-like body and
onboard hardware.

Start with [`docs/overview/README.md`](docs/overview/README.md) (architecture, math, training,
roadmap) and open [`docs/overview/architecture.html`](docs/overview/architecture.html) in a browser
for the diagrams.

## Rules that are easy to break

- The approved execution order is the [fidelity-first roadmap](docs/superpowers/specs/2026-09-22-fidelity-first-roadmap-and-agent-handoff.md):
  bounded A1 audit, then whole-body articulation/fidelity, then integrated capability and deployment.
- The user authorized autonomous local experiments under that roadmap: ≤2 hours per run (including
  resumes), ≤6 experiment-hours per stage, ≤6 workers total, with live progress and cancellation.
  This standing authorization covers specified runs; ask for a new contract, increased budget,
  or otherwise out-of-scope action. Teacher-driven behavioral training is not authorized.

- Pre-registered acceptance thresholds (`roam_eval.ACCEPTANCE`) are never relaxed without the user.
- The legacy room MJCF and accepted actors stay reproducible (`test_legacy_room_mjcf_unchanged`).
- Body feedback and fitted dynamics are additive, versioned identities requiring explicit user
  sign-off (see the body-agnostic spec); the canonical bundle and accepted actors never change.
- Ask the user before long data collection / DAgger / PPO runs.
- Commit and push after each completed step.
- Keep parallel workers modest (≤ 6 on the dev machine) so RAM is not exhausted.

## Commands

A ROS install pollutes `PYTHONPATH`; strip it:

```bash
env -u PYTHONPATH .venv/bin/python -m pytest -q
.venv/bin/ruff format python tests && .venv/bin/ruff check python tests
env -u PYTHONPATH .venv/bin/fly-drone roam-screen teacher random --ablations none ghost \
    --seeds 10 --seconds 60 --level 3 --workers 6 --output runs/roam/screen.json
```
