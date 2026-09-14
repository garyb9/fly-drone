# Fly / Drone — agent brief

## The goal (read this first)

**The fly's connectome is the brain, the drone is its body — a cyborg.** The full MaleCNS
connectome (166,700 neurons, frozen) should autonomously fly the drone: it sees through the
drone's cameras, and the drone moves because of what its descending and motor neurons do. The
drone never runs a hidden script, planner or mode switch that makes the decisions for it.

What that means in practice, and what every change must preserve:

| Principle                 | Concretely                                                                                                                                                                                                                                                       |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| The brain decides         | The only input to the decoder is neural activity (2,022 descending + VNC motor traces). No pose, task id, target position or pixels.                                                                                                                             |
| The body executes         | The drone supplies reflex-level stabilisation (PID, mixer, rotors) — the role the fly's own VNC and halteres play. It never chooses where to go.                                                                                                                 |
| The brain stays the fly's | Wiring, weights, signs and neuron parameters are frozen. Learning lives in the decoder and, for free roam, the v5 sensory encoder (camera images only, 8 uniform population currents in [0, 2]; see docs/sensory-model.md §6). Legacy actors stay on encoder v4. |
| One brain, one decoder    | One actor for free roam. No switching decoders by scenario or state.                                                                                                                                                                                             |
| Behaviour must be causal  | A skill counts only if silencing the pathway that carries it (vision, loom, light) removes it, and a blind (ghost) condition cannot pass by chance.                                                                                                              |
| Honest labels             | Teachers use simulator geometry only for what the eyes could actually see.                                                                                                                                                                                       |

Where we are going: foraging, obstacle avoidance and threat dodging in open, unseen arenas,
all attributable to neurons (free roam, `docs/free-roam.md`); later, richer senses and feedback
into the connectome, and onboard hardware.

Start with [`docs/overview/README.md`](docs/overview/README.md) (architecture, math, training,
roadmap) and open [`docs/overview/architecture.html`](docs/overview/architecture.html) in a browser
for the diagrams.

## Rules that are easy to break

- Pre-registered acceptance thresholds (`roam_eval.ACCEPTANCE`) are never relaxed without the user.
- The legacy room MJCF and accepted actors stay reproducible (`test_legacy_room_mjcf_unchanged`).
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
