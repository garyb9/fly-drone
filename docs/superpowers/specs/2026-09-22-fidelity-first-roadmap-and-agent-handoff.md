# Fidelity-first roadmap: canonical brain → whole fly body → physical cyborg

**Status:** approved for implementation by the user, 2026-09-22.
**Operational entrypoint:** `docs/HANDOFF.md`.
**Authority:** this is the execution roadmap approved in conversation. It preserves the
scientific contracts in `2026-09-19-body-agnostic-fidelity-cyborg-design.md` and supersedes
its ordering where stated below. Historical evidence and acceptance thresholds do not change.

## 1. Goal and user decisions

The full, frozen 166,700-neuron MaleCNS connectome is the brain. The body supplies
mechanics and declared stabilization, never behavioral decisions. Sensory observations
reach neurons; descending and motor activity drives the body. No teacher, task selector,
hidden planner, authored gait, or imported locomotion policy supplies the intelligence.

The goal has three independently demonstrated parts:

1. Faithful embodiment: identified neural activity drives an accountable biomechanical fly
   body, including sensory feedback.
2. Causal capability: one fixed bridge per embodiment supports seeking, obstacle avoidance,
   and threat evasion in unseen environments; pathway interventions remove the skills.
3. Physical deployment: the full brain controls a physical drone, eventually onboard,
   without transferring decisions into the flight controller.

User decisions:

- Run a bounded A1 light-direction audit first.
- Then prioritize **fly-body fidelity and whole-body articulation**, before completing drone
  free-roam capability. Include wings, six legs, head, antennae, and feeding articulation.
- Coordinated walking/grooming are research milestones, not consequences of joint motion.
- Permit staged sensory, feedback, and recording-fitted dynamics proposals. New biological
  contracts remain explicit approval points; approved contracts do not need repeated approval.
- Run approved local experiments autonomously with visible progress and cancellation.
- Maximum **two hours per run**, **six experiment-hours per stage**, **six workers total**.

A reproducible insufficiency finding is a valid stage result, not completion of the goal.

## 2. Baseline and required reading

Read `AGENTS.md`, `docs/HANDOFF.md`, the governing body-agnostic spec, the C4 spec and A0/A3/B0
findings, `docs/overview/README.md`, `docs/sensory-model.md`, and `docs/free-roam.md`.
For embodiment read sibling `fly-playground/docs/2026-09-13-connectome-articulation-plan.md`.
Treat `docs/hardware-estimate.md` as historical estimates, not hardware validation.

| Area | Evidence at approval | Consequence |
| --- | --- | --- |
| Canonical runtime | Full frozen MaleCNS | Preserve graph, ordering, signs, parameters, identity |
| Codec v2 | Fixes idling; beacon rate 0.033/min vs teacher 0.733/min | Alive in the ordinary sense, not capable |
| Codec v3 | Lateral escape mechanically strong; ghost dodge 0.205 | Visual dependence is not skill success |
| C4a | Dodge 0.341, balanced 0.154; L6 move bout 12.80 s | Rejected; no 50-seed confirmation |
| C4b | v4 E1 0.738; frozen v6 0.720, bar 0.8 | No adoptable selective loom candidate |
| Light | Encoder beacon AUC ~0.973 | Presence detection, not proof of descending direction |
| Feedback | Existing P1 identity weak positive | Reusable infrastructure, not established capability |
| Dynamics | Prior primarily varied noise | Recording-fitted leak/threshold remain untested |
| Liveness | L4 unresolved, L7 deferred on drone, v3 also fails L6 | Never report full liveness acceptance |
| Default | v3 kept by user despite failure | Preserve default; explicitly pin comparisons |

Two A1 hypotheses: normalization suppresses a small light response relative to loom, and
the escape-dependent steering multiplier reverses otherwise useful steering on false loom.
Neither is a diagnosis before measurement.

The sibling has substantial dirty/untracked articulation work. It combines NeuroMechFly
geometry and custom TypeScript physics, not a verified biomechanical fly backend. It includes
authored stepping, grooming, landing and other force sources. Preserve that work; reuse only
audited mechanics/mappings. A disabled controller flag does not prove all assists are absent.
Full cell count does not establish numerical equivalence of the sibling and canonical brains.

## 3. Invariants

- Canonical graph, weights, signs, order, tonic inputs and neuron parameters are immutable.
- Alternate sensory/model/feedback contracts have explicit designs, authorization, hashes,
  silencing controls and evidence. Accepted actors never silently load another identity.
- The action bridge receives neural activity only: no pixels, task IDs, rewards, target
  positions/bearings, or simulator state. Body feedback enters neurons, not the action codec.
- One fixed bridge per embodiment; no switching by task, scenario or inferred behavioral mode.
- The VNC is part of the brain. No CPG, gait timer, grooming waveform or landing planner
  replaces missing neural coordination.
- Activation decay, compliance, wing mechanics and declared stabilization may have state;
  they cannot choose destinations or sequences. Zero drive cannot sustain powered movement.
- Keep `ACCEPTANCE`, ablations, liveness definitions, legacy MJCF and accepted actors unchanged.
- Calibrate on declared stimuli/actuator mechanics, never teacher actions or task reward.
- Attribution/correlation alone is not causality. Missing evidence cannot support a pass.
- Teacher/scripted baselines are separate offline comparators, never candidate controllers.

## 4. Managed execution and authorization

This approved roadmap authorizes implementation, diagnostics and specified local experiments
inside its budgets. No per-run reconfirmation for an unchanged approved stage. It does not
authorize new biological contracts before their concrete designs are approved, budget increases,
paid compute, purchases, physical actuation, relaxed bars, destructive treatment of user work,
or teacher-driven behavioral training.

Implement `fly-drone experiment start <manifest>`, `status <run-id>`, `stop <run-id>`, and
`resume <run-id>` before long runs. Commands must expose the run-root location.

Manifests specify stage/hypothesis, prerequisites, authorization, exact command/trials, source
fingerprint, artifact identities, seeds, conditions, resource/time limits, outputs and
pre-registered scientific decision rules. Use argument arrays, not shell command interpolation.

Runtime artifacts: atomic `status.json`, append-only progress events/logs, durable completed
trials, checkpoints when available, and a final outcome distinguishing scientific pass/fail,
cancellation, budget exhaustion and incomplete evidence. Exit code zero is not a scientific pass.

- Heartbeat at least every ten seconds; supervising agent updates at least every sixty seconds
  and between long runs. Estimate ETA only from measured throughput.
- Two-hour cap includes resumed segments. Six-hour stage budget sums concurrent run elapsed
  time. No evasion by renaming/subdividing runs. At most six workers across the whole team.
- One heavy simulation/fitting job at a time by default; lightweight code/tests may run alongside.
- Stop prevents new trials, saves completed outputs and terminates only owned workers. Cooperative
  shutdown precedes bounded process-group termination. No unrelated PID termination.
- Resume verifies manifest/source/identity compatibility and skips validated completed trials.
  Resume at trial boundaries unless exact simulator/RNG restore has been validated.
- User stop prevents automatic launch of the next run until explicitly resumed.

Every stage ends with a finding: hypothesis, fixed/changed factors, identities, commands/seeds,
durations, uncertainty, negative controls, verdict, next permitted action and blocker.
Run relevant checks; commit and push each completed step. Stage only owned changes.

## 5. Dependencies and agent work allocation

`S0 evidence/run management → S1 A1 audit → S2 articulation audit → S3 biomechanics backend
→ S4 whole-body articulation → S5 sensory closure/coordination → S6 integrated capability
→ S7 generalization/transfer → S8 physical and onboard deployment`.

Failure routes to a diagnostic branch or explicit approval checkpoint, never a scripted substitute.

| Role | Ownership |
| --- | --- |
| Coordinator | Contracts, budgets, integration, evidence, handoff, commits |
| Neural/evaluation | A1, mapping, interventions, capability gates |
| Body | Mechanics, mappings, backend, articulation |
| Reviewer | Leakage, hidden assistance, reproducibility, completeness |

At most four agents initially, distinct file ownership, no simultaneous edits to shared APIs.
A single agent can execute sequentially. Six workers is a machine-wide cap, not per agent.

## 6. S0 — Reliable starting point

Record revisions, canonical hashes, codec identities, environment and tests. Inventory sibling
dirty sources without modifying them; HEAD alone cannot reproduce that worktree. Reconcile stale
summaries while retaining historical findings and the user-selected v3 default.

Implement managed experiments and an additive completeness check around evaluation reporting:
all requested conditions/seeds completed, all A5 probes and sides present, ghost near-threat
evidence present, identity/provenance valid, and A7 measured for realtime claims. Keep existing
acceptance arithmetic and old artifacts unchanged. Offline behavioral completeness and complete
realtime acceptance are distinct. New incomplete evidence cannot be promoted.

Exit: interrupted/resumed lightweight experiment verifies status, cancellation, accounting and
preservation; compatibility tests pass; baseline and blockers are recorded.

## 7. S1 — Bounded A1 transmission audit

Trace `rendered beacon → currents → neural response → readout → codec → heading change`.
Compare explicit v1/v2/v3 on identical neural inputs. Record currents, sided neural activity,
normalization/saturation, escape latch, commands, actual yaw, latency and bearing change.
Geometry is for stimulus generation/scoring only.

Battery: synthetic L/R/bilateral/absent light at sub-saturating and saturating strengths; rendered
bearings/distances; onset/sustained/disappearance/reacquisition; normal rotation without threat;
mixed light/loom with matched silencing; persistent state as well as reset trials. Start with
eight development and eight disjoint confirmation seeds. Freeze the exact battery before runs.

Start with existing steering readouts, then anatomically named output groups if needed. Freeze
candidate membership/sign before confirmation; no arbitrary task-fitted weighted search.

Decision branches:

- Direction missing in currents: sensory-direction problem.
- Direction in currents, absent in outputs: neural transmission/model problem.
- Direction in outputs, suppressed/reversed in commands: codec problem.
- Correct commands, poor realized turns: mechanics/timing problem.
- Visible steering works, collection fails: approach/search problem.

Reuse A5 steer/approach falsifiers. Synthetic sign alone is insufficient. Deliver
`a1-transmission-audit.json` and a finding. Specify any supported small codec correction as an
additive candidate, without changing defaults or starting a sweep. Carry it to S6. Proceed to
body fidelity after this bounded audit even if A1 is unresolved.

## 8. S2 — Existing articulation and force-source audit

Classify every applied force/torque, joint target, drive and override as passive mechanics,
neural coupling, stabilization, intervention, or behavioral assistance. Explicitly inspect
tripod stepping, grooming waves, surface landing/alignment, escape impulses, boundary forces,
feeding overrides and activity floors. Emit a source/actuator ledger.

Create a separate neural-only execution path; retain assisted demos with honest labels. In causal
trials disallowed sources contribute exactly zero, rendering consumes physical state, no visual
animation drives physics, bounds terminate instead of steering, and direct stimulation is tagged.
Do not assume `controllersEnabled=false` isolates these sources.

Exit: hidden-actuation tests, passive decay, physical/render correspondence and deterministic
replay pass; reusable modules identified; user work preserved.

## 9. S3 — Canonical brain and actual biomechanics

Default mechanical backend: pinned **flybody MuJoCo core body**, without its learned policies.
Use NeuroMechFly as articulation/contact/sensing reference, not its authored controllers.
Upstream: https://github.com/TuragaLab/flybody and https://neuromechfly.org/.

Body implementation belongs in fly-playground; fly-drone retains canonical runtime and evaluation.
Expose an importable body package without duplicating brains. Never modify pinned `web/src/fly/*`.
Pin assets, dependency versions, units, actuator definitions and licenses. Missing joints require
evidenced extensions, not silent behavioral substitutes.

| Boundary | Contract |
| --- | --- |
| Neural frame | Timestamp, identity, ordered descending/motor traces |
| Body codec | Fixed neural-frame to embodiment-command map |
| Body backend | Reset, fixed advance, eyes, snapshot/restore, capabilities, close |
| Body observation | Whitelisted gyro/joints/loads/contacts/actuators |
| Sensory adapter | Images/physical observations to declared input currents |
| Diagnostic truth | Geometry/labels for evaluators only |

Preserve drone command shape/units. Fly outputs are actuator activations/torques, not walk/groom/
land requests. Do not forward `DronePlant.state()` wholesale: it includes oracle target data.
Keep canonical 5 ms neural tick, verified mechanical substeps and deterministic scheduling. Hold/
integrate drive across physics steps. Camera geometry/rate changes have separate sensory identity.
Keep full graph and ordering; any new output map is versioned without changing legacy features.
Do not claim within-wingbeat spike timing at this neural timestep.

Exit: reproducible full-brain-to-physics loop, no locomotion policy, accounted forces, units/signs,
passive decay, replay and pinned identities. Motor stimulation proves coupling, not autonomy.

## 10. S4 — Whole-body articulation

Registry per actuator: neuron IDs/grouping, side evidence, nerve/muscle/joint correspondence,
confidence/source, transfer function, limits, calibration and unresolved ambiguity. Soma side is
not automatically output side; whole-leg means do not identify antagonists. Whole-body completion
requires requested subsystems supported or a user-approved scope exception.

| Package | Required evidence |
| --- | --- |
| Wings | Independent power/steering; unilateral response; sustained energy/lift ends on silencing |
| Six legs | Supported antagonists, limits, distinct signed flexor/extensor movement |
| Contact | Ground reaction forces explain support/translation; no scheduler chooses steps |
| Head | Motor-driven pose changes eye orientation |
| Antennae | Side-specific joints, no idle oscillator |
| Proboscis | Supported extension/retraction, no feeding override |
| Abdomen | Supported segment movement, no invented pumping |

Order wings → legs → head/antennae → mouthparts/abdomen → integrated contact. Parallelize independent
mapping/tests. Each assay compares baseline, direct motor drive, upstream stimulus when approved,
pathway silencing, restored and unrelated-cell controls. Predeclare sign, response window, effect
above noise and saturation limits; confirm on held-out seeds/strengths.

Exit: whole-body demonstration with synchronized neural/actuator/force/joint telemetry. Distinguish
actuator coupling, sensorimotor pathway and autonomous coordination; unsupported behavior stays absent.

## 11. S5 — Sensory closure and coordination

Prepare separately approved mappings for gyro/haltere, joint position/velocity/load, contact,
head-dependent vision and actual foot/mouth taste. Add other modalities only for a diagnosed gap.
Reuse P1 infrastructure, not its arbitrary rotor-to-limb convention. Each channel declares units,
range, time, cells, confidence, identity and silencing. No proximity shortcut for contact taste.

Coordination gates: posture/support; taste-to-proboscis; wing mechanics/perturbation recovery;
untethered flight if supported; coordinated stepping; grooming selection/repetition. Separate gates,
not promises. Examine antagonists/premotor coverage before concluding rhythms are absent.

If inputs do not yield useful outputs: verify identifiers, graph coverage, signs, units, timing,
strength, readout resolution/saturation and feedback; compare physiological evidence; only then
prepare a minimal alternate dynamics proposal. Establish recordings/correspondence before fitting.
Start with leak/threshold, fit neural responses (not actions/reward), hold out recordings and
interventions, freeze before behavior evaluation. No recordings means blocked, not task-fitted.

Exit: causal behaviors and precise missing dependencies reported. Re-enable L7 only when actual
body authority supports it; exclude teleports; never lower its detector to pass.

## 12. S6 — Integrated capability on both embodiments

Seeking: implement S1-supported additive correction; visible orientation → approach/collection →
reacquisition → unseen-target search → competing stimuli. Search comes from neural dynamics, never
scan timers/waypoints/explore constants. Fly results do not prove drone results; calibrate mechanics.

C4b stays parked until this package. Default research proposal: declared local expansion sensing
separating expansion from translation/rotation, with sensory-only inputs. v4/v6 remain negative
baselines; no unchanged reruns. Adoption requires E1 ≥0.8, existing E2 margins, useful warning time,
preserved light direction, turn/shadow/exposure controls and downstream neural response. Trained v6
is a separately approved alternative, not automatic teacher-policy revival.

After selective sensing: measure lateral/vertical dynamics in SI; isolate climb reduction versus
lateral gain; test additive escape against rejected tradeoff; include both sides and head-on.
Do not invent a direction for symmetric neural output with a hidden policy.

One bridge handles all stimuli. Free falsifier → 15 seeds ×120 s level-3 smoke →50 fresh confirmation
seeds only on eligible candidate. Earlier inspected seeds are diagnostic. No newly failing criterion;
old failures stay visible. Single-skill gain does not promote default or establish full capability.

## 13. Acceptance and claims

Code is authoritative; unchanged drone gates:

| Gate | Requirement |
| --- | --- |
| A1 | ≥0.6 teacher beacons, ≥2×best registered rival, positive lower paired CI |
| A2 | ≤0.5 collisions/min and ≤half better loom/ghost collision rate |
| A3 | Dodge and balanced ≥0.8; ghost ≤0.3 |
| A4 | Existing selective light/loom and cross-skill preservation rules |
| A5 | Every skill balanced ≥0.8 |
| A6 | Coverage ≥0.4, slow ≤0.1, yaw bias ≤0.25, no loss-of-control |
| A7 | Measured live ≥1× realtime |

Report liveness alongside capability. Before autonomous fly evaluation, register a separate
body-scaled protocol using verified mechanics and biological references; drone dimensions/speeds
are not biological bars. No post-result threshold selection.

Strong causal claims require stimulus→neurons→actuation→outcome, relevant silencing, restored and
unrelated controls, appropriate ghost/shuffle/zero controls and separation of general immobility
from skill loss. Retain rewiring and E3 bypass diagnostics. E3 has no numeric acceptance bar;
bypass parity/superiority requires investigation before strong connectome-computation claims.

## 14. S7 — Generalization and transfer

Freeze candidate before unseen families: S0 canonical unseen seeds, S1 denser, S2 larger room,
S3 shifted threat timing/speed, S4 connected split room/detour. Pre-register geometry, distributions,
seeds, visibility and difficulty; validate connectivity and actual shift. Old DAgger/SAC remedies
are superseded: failure returns to senses/neural dynamics/mechanics.

Proposed additive family gate (register before evaluation): all applicable capability/causal metrics
per family, beacon rate ≥0.6 S0 and ≥0.6 family teacher; never average a failed family away.

Version perturbations: exposure/blur/noise/contrast/timing, delay/jitter/dropout, estimation errors,
actuator lag/thrust/torque and parameter uncertainty. Perturb every relevant pathway including
declared sensing and feedback. Use measured hardware ranges when available; otherwise label as
robustness research, not validated transfer. Exit: frozen identity passes registered envelope.

## 15. S8 — Physical and onboard deployment

Before purchasing/flight, specify airframe/actuators, camera geometry/synchronization/exposure,
controller units/frames/limits, estimator isolation, timing/stale commands, compute/memory/power/
thermal/endurance measurements. Historical estimates are not board qualification.

Sequence full-graph compute benchmark → cameras/interface without propulsion → actuator system-ID
→ hardware-in-loop → constrained replica arena → physical causal trials → held-out layouts →
onboard parity/endurance. Ground compute is acceptable for first demonstration and labeled offboard.

No graph reduction for speed. Exact optimization keeps golden traces; changed RNG/numerics requires
separate backend identity and parity/behavior checks. Benchmark complete sensor-to-command latency.
Purchases/propulsion/flight require an approved operating envelope. Log all failsafe interventions;
hover/geofence/landing assistance never earns neural skill credit. Physical controls remain inside
that approved envelope. Onboard completion needs measured deadlines, power, thermal and endurance.

## 16. Verification, promotion and handoff

Verify canonical/accepted hashes, legacy MJCF, ordering/pins, no privileged bridge input, ablations,
actuator signs/limits/energy, force accounting, clocks/replay, cancellation/resume/budgets,
completeness/missing samples and backend numerical/physical-unit parity.

Use `env -u PYTHONPATH .venv/bin/python -m pytest -q`; format/check only owned files with ruff.
Run sibling/web checks relevant to changed components. Do not bulk-format dirty user files.

Keep distinct shipped, candidate, rejected and accepted identities. Preserve v3 until explicit
promotion. Never auto-update accepted pointers. Improved A1 plus failed avoidance is partial.

Every handoff records stage/package, goal, revisions/identities, authorization/remaining budget,
running job/status/stop command, completed evidence, failures/assumptions, exact next action and
forbidden actions. Commit compact manifests/findings; hash large external trajectories/checkpoints.

## 17. First assignment

Execute S0 then S1; publish the bounded finding; proceed to S2 and whole-body mechanics. Do not begin
with v6 retraining, escape gain sweeps, rejected C4a confirmation, imported locomotion policies or
replacement of the canonical neural runtime. First major delivery is accountable whole-body neural
embodiment. Final delivery remains causal behavior on that body and a physical drone.
