# External prior art: connectome-driven drones, and what this project takes

This doc answers one question for every external project that puts a fly connectome in a body:
**does it help the goal in [`overview/README.md`](overview/README.md) — a frozen MaleCNS connectome
flying the drone, with every skill attributable to neurons?** It records the survey, the exact
comparison, the open gap, and the pieces this project adopts. It changes no code, no
`roam_eval.ACCEPTANCE` threshold and no encoder identity.

The harvest design (what gets ported, how, with which guardrails) is
[`superpowers/specs/2026-09-18-prior-art-harvest-design.md`](superpowers/specs/2026-09-18-prior-art-harvest-design.md).
Links are collected in [`references.md`](references.md#connectome-driven-bodies-prior-art).

## 1. Verdict

As of 2026-09-19, **nobody has solved the goal**. Eon Systems' embodied fly (2026) now closes a
whole-brain→fly-body loop without task training, which is the closest external work to _natural_
behaviour — but for a fly body, not a drone, with no camera→Mi1/Tm3 encoder and no causal controls.
Several projects solve overlapping subsets — a frozen whole connectome in software,
camera-to-visual-cell encoding, a learned readout, or a simulated body — but no project combines
all of the following, and no project has a peer-reviewed, independently reproduced flight:

1. a **learned** camera-to-visual-neuron encoder (rather than precomputed optic flow or feature
   detectors) driving **Mi1/Tm3 and LC4/LPLC2**;
2. the **full** 140–170k-neuron LIF network in the closed loop;
3. **only** a decoder over descending + VNC-motor traces to quadrotor setpoints, with the
   connectome frozen;
4. a **logged real flight** whose only non-biological controller is the stock attitude loop.

This project's pre-registered causal tests (`roam_eval.ACCEPTANCE`, A1–A7) are also stricter than
any surveyed project's evaluation. That is the moat, not the model.

## 2. Comparison

| Project                             | Frozen full CNS                          | Camera → labeled visual cells                                              | Learned decoder → body                                 | Real flight                            | Causal/ablation controls        | Solves our goal?                        |
| ----------------------------------- | ---------------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------ | -------------------------------------- | ------------------------------- | --------------------------------------- |
| **FlyDrones** (SpikeCalls)          | ✅ MaleCNS 166k LIF                      | ⚠️ per-cell optic flow → T4/T5/LPLC2/LC4; no Mi1/Tm3 from pixels (roadmap) | ⚠️ ridge readout from 6 DNs → RC sticks                | ❌ adapters written, not flight-tested | ❌                              | Closest, partial                        |
| **fly.ai** (alextitonis)            | ✅ 166,700 / 25.6M                       | ⚠️ injects feature detectors (LPLC2/LC4/LC10a), not eyes/Mi1/Tm3           | ⚠️ reservoir readout, no drone                         | ❌                                     | ⚠️ scrambled-wiring control     | Best oracle + negative results          |
| **dylankainth/flybrain**            | ⚠️ FlyWire female 139k                   | ✅ opt-in retinotopic eye map (Buchner 1971) → R1-6                        | ⚠️ hand population code, no training                   | ❌ bench-test advised                  | ❌                              | Take the retinotopy                     |
| **fly-self-driving** (suanmiao)     | ❌ trains gains/leaks                    | ✅ pixels → optic-lobe sensory → VNC motor                                 | ✅ DAgger on unseen streets                            | ❌ simulated car                       | ✅ shuffled-graph + state-carry | Best method sibling                     |
| **FlyGM** (arXiv 2602.17997)        | ❌ trains whole graph                    | ⚠️ proprio+video → afferents                                               | ✅ RL on flybody (walk+flight)                         | ❌ sim fly                             | ✅ rewired/ER/MLP baselines     | Different thesis (arch prior)           |
| **DOOMFLY**                         | ⚠️ +plasticity (failed gates)            | ⚠️ R1-6 proxies                                                            | ✅ fixed DN→buttons                                    | ❌                                     | ❌                              | Negative results                        |
| **flybrain-robot-bridge**           | ❌ mock, MaleCNS stub                    | ❌ hand groups                                                             | ❌                                                     | ❌                                     | ❌                              | Ignore                                  |
| **Eon Systems embodied fly** (2026) | ✅ FlyWire 138k + connectome-constrained | ⚠️ brain-wide sensory drive, body vision                                   | ⚠️ connectome-constrained motor→body, not task-trained | ❌ simulated body                      | ❌                              | Closest natural fly-in-a-body; no drone |
| **fly-brain-full** (rndlabsoy)      | ✅ 138,639 FlyWire v783                  | ⚠️ vision/olfaction/gustation into afferents                               | ⚠️ emergent, no training                               | ❌ simulated fly                       | ❌                              | Hobby-scale embodiment                  |
| **flybrain** (snedea)               | ✅ 139,255 FlyWire FAFB                  | ⚠️ scripted sensory drives                                                 | ⚠️ emergent, no training                               | ❌ browser fly                         | ❌                              | Emergent-behaviour demo                 |

Legend: ✅ real and working · ⚠️ partial, modelled, or not validated · ❌ absent.

## 3. Source by source

### FlyDrones (SpikeCalls) — the closest match

`camera → per-cell optic flow → T4/T5, LPLC2, LC4, R1–R6 → MaleCNS signed LIF → DNg02/DNp03/DNp01
→ throttle/yaw/forward → safety → drone`. Frozen MaleCNS v1.0 (~166k neurons, ~25M directed
connections after ≥3-synapse filter), Shiu-2024 LIF parameters. `flydrones calibrate` fits a ridge
readout over a 7-stimulus battery; only the readout is fitted. Drone adapters (Tello, Crazyflie,
MAVLink, ESP32/Betaflight) are written against the official SDKs but **the authors state they have
not been flight-tested**. The browser demo runs MiniFly (850 neurons), not the full brain.

- **Taken:** the 7-stimulus calibration battery (workstream D); the honest statement that the bridge
  is engineered; the roadmap target of driving early vision (Mi1/Tm3) from pixels — which is exactly
  this project's v6 direction.
- **Not taken:** per-cell optic-flow preprocessing (our v6 learns the encoder and lets the optic lobe
  compute motion); the hand-mapped DN→stick readout (our decoder is learned over all 2,022 traces).

### fly.ai (alextitonis) and the `flybrain` package

A task-agnostic "fly reservoir": frozen 166,700-neuron / 25,582,938-connection MaleCNS LIF, an
encoder the user writes (drive named neuron types via `brain.step(inject=...)`), and a readout fitted
on PCA of neural activity (ridge/logistic). Injects **feature detectors** (LPLC2/LC4/LC10a), not the
eye. Its README publishes five findings, including two directly useful negative results: with Fly64's
tonic/gain settings the whole network sits at threshold and vision does nothing; and the
photoreceptor→lamina relay is lost because photoreceptors are histaminergic (inhibitory) onto graded
lamina neurons. It also publishes a laterality check: left LC4+LPLC2 → left giant fibre DNp01
**+17 to +25 spikes/s**, right unchanged; LC10a → left DNa02 +1.4 to +3.7. No drone, but the cleanest
independent reference implementation of the same connectome.

- **Taken:** the independent oracle for laterality/propagation cross-checks (workstream C); the
  validation that injecting past the photoreceptors (at Mi1/Tm3/LC4/LPLC2) is the right call; the
  transmitter-sign conventions to investigate (workstream E).

### dylankainth/flybrain — retinotopic eye map

Runs the female FlyWire connectome (139,255 neurons) on a Tello with no training and a hand
population code over ~1.5k descending neurons. Its distinctive asset is an **opt-in retinotopic eye
map**: the measured ommatidial viewing directions of Buchner (1971), digitized by the Straw lab
(`strawlab/drosophila_eye_map`, BSD), pinhole-projected into a forward camera and sampled with a ~5°
acceptance window. The repo is explicit about scope: the eye **geometry** is real measured data; the
**cell↔ommatidium identity** is modelled (retinotopic order, not the connectome's visual columns);
one Tello camera only covers a frontal ~66°×50° cone.

- **Taken:** the Buchner/Straw eye geometry and pinhole sampler, recomputed for this project's two
  splayed 64×48 ±0.75 rad cameras, to verify the v6 Tm4 PCA visual-axis correspondence that the v6
  spec flags as unverified (workstream A).
- **Not taken:** their R1-6 cell assignment (we inject at Mi1/Tm3/LC4/LPLC2, and our v6 encoder is
  learned, not a luminance map).

### fly-self-driving (suanmiao) — the best methodological sibling

A frozen MaleCNS adjacency (165,122 neurons, 25,563,197 synapses) run as a recurrent network driving
a simulated street from a 64×32 windscreen. It **trains one gain per synapse and one leak per neuron,
so the internals change** — a different thesis from ours — but two methodological results transfer:
carrying neural state across decisions (never resetting) took the same graph from 3/20 to 17/20
streets, and a randomly rewired graph reaches only 16/20 vs 20·19·19/20 for the real wiring. DAgger
and closed-loop evaluation on unseen layouts confirm this project's recipe.

- **Taken:** the state-continuity regression test (workstream A-adjacent) and the shuffled/rewired
  graph control (workstream B).

### FlyGM (Jin, Zhu, Zhang, Sui — arXiv 2602.17997)

Instantiates the whole-brain connectome as a directed message-passing graph and trains the **entire**
controller (shared update MLP + per-neuron intrinsic descriptors) with imitation + PPO on the
`flybody` biomechanical fly in MuJoCo, covering walking, turning and flight. Its controlled
comparison — connectome vs degree-preserving rewiring vs Erdős–Rényi vs MLP — shows the wiring is a
real structural inductive bias, and even an unweighted connectome beats the non-connectome baselines.

- **Taken:** the rewired / random-graph / MLP baseline design for this project's causal control
  (workstream B) and the citation that the wiring carries information beyond parameter count.
- **Not taken:** everything that trains the brain. Our connectome is frozen and only the decoder
  learns; FlyGM's frozen ingredient is the adjacency, not the circuit.

### DOOMFLY (nftechie)

Frozen MaleCNS plus an experimental dopamine-gated plasticity rule on KC→MBON11, playing Doom. It
publishes its negative results: the v6 candidate **failed its visual, conditioning and survival
validation gates**. Preserved as a warning about plasticity claims, not a source to copy.

### flybrain-robot-bridge (Frankweb33)

A mock neural backend and a MaleCNS stub that exits `Not implemented yet`; no graph is loaded. No
hardware test performed. Nothing to take.

### Eon Systems embodied fly — closest to a _natural_ fly, wrong body for us

Announced 2026-03-08 with a technical deep dive 2026-03-10. Integrates the FlyWire connectome and a
connectome-constrained brain model (Lappalainen/Shiu) with **NeuroMechFly v2**, closing the loop
through a physics-simulated fly body and demonstrating multiple behaviours. This is the closest
external work to "a natural fly in a body" and the clearest demonstration that _fitted dynamics +
a real body_, not task-trained readouts, is the route to natural behaviour. It is a company claim,
not peer-reviewed, and it does not meet our goal: no camera→Mi1/Tm3 encoder, no quadrotor, no
pre-registered causal controls, no real flight.

- **Taken:** the validation of the connectome-constrained direction (this project's P2) and the
  reminder that the body must be fly-like for the motor output to mean what the connectome evolved
  to mean (Scope A). Not their model or code.

### fly-brain-full (rndlabsoy) and flybrain (snedea) — whole-brain bodies without training

Two open, hobby-scale demonstrations that a whole FlyWire network in a body can produce behaviour
with no task training: `fly-brain-full` (138,639 neurons, NeuroMechFly v2/MuJoCo, vision/olfaction/
flight) and `flybrain` (139,255 neurons, browser, "the fly is not scripted"). They corroborate the
"no behaviour training" thesis and are useful references for embodiment plumbing.

- **Taken:** design reference only; neither is integrated or validated by us.

### Foundational science

- **Shiu et al., Nature 2024** — the whole-brain LIF recipe (parameters, activation/silencing) this
  project and nearly every project above use.
- **Lappalainen et al., Nature 2024** — connectome-constrained networks: fit unknown per-neuron and
  per-synapse parameters to recorded activity with the wiring fixed, predicting living fly neurons.
  The scientific basis for this project's P2.
- **Vaxenburg et al., Nature 2025 (`flybody`)** and **NeuroMechFly v2 / FlyGym** — the fly body and
  sensorimotor loop; the fidelity reference for Scope A's fly-like body.
- **MaleCNS v1.0 (2026)** — the brain+VNC connectome; the reason the "descending neurons → body"
  demos appeared.

## 4. The open gap

A paper or repo that (a) injects a camera into **Mi1/Tm3 and LC4/LPLC2** rather than precomputed
optic flow or feature detectors, (b) simulates the **full** 140–170k LIF net in the loop, (c) uses
**only** DN/VNC traces (plus declared feedback) to drive quadrotor setpoints, and (d) shows a
**logged real flight** with the stock attitude loop as the only non-biological controller.
FlyDrones is built to be that project; as of 2026-09-19 its authors have not published the flight
logs they list for v0.2. Eon Systems closes the fly-body version of (b), but none of (a)/(c)/(d) for
a drone, and publishes no causal controls.

## 5. Discrepancies to reconcile

| Item                | This project                                                                     | Others                                                                                                     | Status                                                                                                      |
| ------------------- | -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Directed edge count | **10,520,431** edges with ≥3 synaptic contacts (`data/malecns/manifest.json:8`)  | FlyDrones / fly.ai / DOOMFLY: **25,582,938** "after typical filtering"                                     | **Unreconciled.** Same source file, ~2.4× difference. Investigate before citing edge counts (workstream C). |
| Transmitter signs   | ACh `+1`; GABA and Glu `−1`; unresolved/modulatory silent (`neuron-model.md:68`) | fly.ai: GABA, Glu **and histamine** inhibitory. FlyGM: ACh/Glu/ASP/**His** excitatory, GABA/Gly inhibitory | Three conventions disagree on glutamate and histamine. Investigate as an additive identity (workstream E).  |
| Photoreceptor relay | We inject at Mi1/Tm3/LC4/LPLC2, so the relay is bypassed                         | fly.ai finding #2: signal dies at the histamine/graded lamina relay                                        | **Consistent** — supports injecting past the photoreceptors. Recorded as validation, not a defect.          |

## 6. Extraction ledger

| Piece                                               | Source                                                                    | Lands in                                                      | Risk                                                   |
| --------------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------ |
| Buchner-71 ommatidial directions + pinhole sampler  | `dylankainth/flybrain` `flybrain_eye_map.py`, `eye_map/` (BSD, Straw lab) | `python/fly_drone/eye_geometry.py` + test                     | Vendored BSD data; splay/FOV differ from theirs        |
| 7-stimulus calibration battery + ridge readout      | FlyDrones `src/flydrones/calibrate.py`                                    | `python/fly_drone/calibration.py` (diagnostic)                | None; additive                                         |
| Degree-preserving rewiring + shuffled-graph control | FlyGM §4.1; fly-self-driving `train_street.py`                            | `scripts/make_rewired_bundle.py`, `scripts/rewired_report.py` | Heavy graph rebuild; diagnostic only                   |
| Independent MaleCNS oracle + laterality checks      | `flybrain` PyPI; fly.ai README findings 1–5                               | `scripts/check_fly_ai_oracle.py`                              | Optional dep, ~260 MB data                             |
| State-carry-between-decisions regression            | fly-self-driving (3/20 → 17/20)                                           | test on `brain.py`/`env.py`                                   | None                                                   |
| Transmitter-sign conventions                        | fly.ai (histamine), FlyGM §3.1                                            | additive `sign-v2` bundles                                    | Built; **S1 kept** (S2 identical, S3 escape-gain only) |

## 7. Contract guardrails

- `roam_eval.ACCEPTANCE` is never relaxed; the rewired and sign-v2 work is **diagnostic** and emits
  separate reports.
- The frozen v4 bundle, `ENCODER_VERSION` and every accepted actor stay bit-identical and never load
  an alternate bundle.
- Alternate bundles are new identities; the current `dataset_hash` covers only `graph.bin`, so it is
  hardened to cover `neurons.bin` and the manifest model block **before** any alternate bundle
  exists (harvest spec Step 0).
- New external dependencies are optional and skipped in the default test suite.
