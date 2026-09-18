# External findings on fly navigation, and what they mean for this project

This doc reads a set of external, connectome-mining findings on _Drosophila_ navigation
(`pwang724/fly-circuit-exploration`, September 2026) and asks one question of each: **does it help
the goal in [`overview/README.md`](overview/README.md) — a frozen MaleCNS connectome flying the
drone, with every skill attributable to neurons?** Things that do not help are marked as such and
not scheduled.

It records no new biology and changes nothing. It is background for a possible future milestone
and a filter for what to ignore. External links are collected in
[`references.md`](references.md#connectome-mining-and-navigation-circuit-external). The sibling
survey of connectome-driven bodies (drones, robots, reservoirs) is
[`external-prior-art.md`](external-prior-art.md).

## 1. Provenance and what kind of claim each statement is

The source is a two-day, LLM-assisted pass over the MaleCNS v1.0 and hemibrain v1.2 connectomes
plus a literature sweep, published with an audit. It reports **no new experiments**. Every claim
there is labelled as one of:

| Label              | Meaning                                                              |
| ------------------ | -------------------------------------------------------------------- |
| already published  | Stated in the cited literature.                                      |
| measured in wiring | A synapse count or column offset computed from the two connectomes.  |
| simulated          | A toy rate model run by the authors, not a fly.                      |
| proposed, untested | A hypothesis about which cells do what, explicitly not yet recorded. |

The headline claim (fast synaptic weight storage) is **proposed, untested**: the cells that would
carry it (hΔH, hΔI) have never been recorded. The two recorded hΔ integrators (hΔG, hΔA) leak over
seconds and their mechanism (activity vs. synapse) is undetermined. This project must therefore
treat the whole thing as a source of _hypotheses and cell labels_, not as ground truth.

## 2. The four findings

### Finding 1 — the path-integration sum may live in synapse strengths (hΔH, hΔI)

A fly that leaves food and wanders in the dark returns in a straight line by summing its steps
(path integration). The step signal is known: hΔB, a fan-shaped-body bump whose column is the
travel direction and whose height is speed. Two ways to hold the running sum: in neural activity
(a recurrent integrator, the textbook answer) or in synapse strengths (each step strengthens the
synapses for its own direction). The wiring screen says hΔB lands column-matched on **hΔH**
(82% same column) and **hΔI** (48%), giving a synaptic store the right geometry; hΔJ receives more
hΔB synapses but in opposite columns, so its contributions cancel. A write gate exists on both
(FB5H dopamine on every hΔH cell; FB4M dopamine on every hΔI cell, itself fed by the velocity
cells PFNv/PFNd and hΔB). A reset exists on hΔH only (OA-VPM3 octopamine, uniform across
columns); hΔI lacks one. Outputs reach the goal neurons FC2 and the steering neurons PFL3. In a
toy simulation hΔH retains 86% of an ideal store, hΔI 56%, hΔJ 14%. Reading is not a recall step:
a store cell fires as hΔB drive × stored synapse strength, so the same walking reads and writes.

### Finding 2 — the stored vector points away from food; hΔM and hΔI invert it

The store holds "I am north of the food"; steering needs "walk south". Every hΔ cell has dendrites
in one column and axon 180° away, so a signal entering on the dendrite leaves rotated — but hΔB
lands on the **axonal** arbor of hΔH/hΔI/hΔA/hΔG (82–95%), so they relay the displacement
unrotated. Exactly two short routes to PFL3 rotate it: **FC2 → hΔM → PFL3** (178°) and
**hΔA → hΔI → PFL3** (195°). Direct routes dominate by synapse count and steer away; the rotated
routes are the return path. PFL2 also receives the rotated copy, matching its known "fires most
when facing away from goal" tuning. Neither hΔM nor hΔI has an assigned function.

### Finding 3 — the motion signals come from unnamed cells

The compass needs turning speed (through GLNO) and the vector system needs forward speed (through
PFNd). An unbiased input screen points at **PS196_b** (a posterior-slope type, 19% of GLNO input,
also feeding the compass-learning dopamine neuron ExR2, ExR4, LPsP and FB3A) and at **FB3A** (four
glutamatergic tangential cells, 12% of PFNd input, pooling optic flow, antennal mechanosensation
and ascending neurons). **AN07B037**, an ascending neuron, sits upstream of both channels, arguing
for a body-derived rather than efference-copy signal. None of these have published physiology.

### Finding 4 — the compass has a built-in brake

Every ring-attractor model treats the PEN shifters as one-way conveyors: read the heading bump at
position x, write it at x±1. The connectome shows each PEN receives about **three times** more
input from the position it writes to than from the one it reads, on both sides, in both
connectomes. At that strength a naive ring model stops rotating; because real flies turn normally,
the write-position synapses must be functionally weak or cancelled (e.g. by Δ7, ExR4, ExR6). The
defensible claim is that rotation gain falls steeply as the anchor strengthens, not that the model
is correct.

## 3. Relevance filter: what helps this project and what does not

Our contract is stricter than a connectome read: the only decoder input is the 2,022
descending/VNC-motor traces, the connectome is frozen, and skills must survive silencing of the
pathway that carries them. Against that:

| Finding                                          | Helps the goal?               | Why                                                                                                                                                                                                                                 |
| ------------------------------------------------ | ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 — synaptic store hΔH/hΔI                       | Conceptually yes; **blocked** | Homing after a beacon leaves view is a navigation skill we lack, and foraging implies return. But our model is static LIF with frozen weights and electrically silent dopamine/octopamine (see §4), so the store cannot exist here. |
| 2 — hΔM/hΔI return inverters                     | **Yes, as observability**     | Names the cells that would carry a homing readout and a sign. Useful now for diagnostics and attribution, even before any homing role exists.                                                                                       |
| 1 & 2 — named navigation cell sets               | **Yes, directly actionable**  | hΔA–M, FC2A/B/C, PFL1/2/3, EPB/PEN, PFNd/v all exist in our bundle; we can watch them and test structure without touching the decoder.                                                                                              |
| 3 — PS196_b / FB3A / AN07B037                    | Later                         | Only matters once body/ascending feedback or a speed estimate is on the table. Candidate labels for diagnostics now.                                                                                                                |
| 4 — compass brake (EPG→PEN 3:1)                  | Diagnostic only               | Behaviour does not need it. Our model has uniform signs, no per-type gains and one-tick delays, so it cannot reproduce the fitted-ring result either way.                                                                           |
| Sibling repos (fly-brain, flycoinrh, spectacles) | Later                         | References for a plasticity milestone and onboard GPU compute; nothing to adopt into the current build.                                                                                                                             |

**Bottom line.** Findings 1–2 do not change how we fly today. They sharpen _what to observe_ and
provide the clearest external justification yet for the one capability our connectome cannot
express: a state held in synapses.

## 4. Where our simulator stands (the honest gap)

Everything here follows [`neuron-model.md`](neuron-model.md); nothing is new.

- **No plasticity.** Weights are quantised synapse counts fixed at load (`graph.bin`, `w_norm`).
  There is no tag, no weight update, no decay. A vector summed into synapses cannot be represented.
- **Modulators are silent.** Dopamine, serotonin and octopamine keep their anatomical edges but
  inject zero direct current, because their predicted transmitter is neither ACh (+1) nor
  GABA/Glu (−1). The finding's write gate (FB5H/FB4M dopamine) and reset (OA-VPM3 octopamine) are
  exactly those silent cells. This is a declared modelling hypothesis, not measured physiology.
- **Uniform signs and gains.** Every inhibitory source is −1, every excitatory +1, no per-type
  gain. Finding 4's 3:1 ratio is present in the anatomy but cannot be "read" as behaviour here.
- **Readout is motor-only.** The decoder sees 2,022 descending/VNC-motor traces
  (`brain.py`). The central-complex cells in findings 1–4 are in the graph and influence those
  traces, but we neither name nor observe them today; `groups.json` exposes only
  `escape/wing/thrust/yaw_torque`.

So the correct statement to make to a reader is: _the frozen connectome may well contain a
path-integration circuit, but our model of it — static, sign-uniform, modulation-free — is not one
that can hold a synaptic vector, and we have not measured whether the relevant activity exists in
our simulation._ That is a limit of our model and of the evidence, stated plainly.

## 5. Proposed code improvements

These are goal-aligned and contract-preserving. They change no threshold, no decoder, no encoder,
no weight. They are deferred until the encoder-v5 SAC pipeline (`pipeline13-15.sh`, Tasks 13–15)
stops, because `sac-round`/`roam-*` re-import `python/fly_drone/*.py` at spawn
([`superpowers/specs/2026-09-15-training-improvements.md`](superpowers/specs/2026-09-15-training-improvements.md) §8).

1. **Navigation-circuit observability (`python/fly_drone/circuit.py`, new).** Resolve named
   navigation cell types from `cells.json`: `EPG`, `EPGt`, `PEN_a/b`, `hDeltaA…M`, `PFNd`,
   `PFNv`, `FC2A/B/C`, `PFL1/2/3`, `FB3A`, `FB4M`, `FB5A`, `FB5H`, `OA-VPM3`, `PS196_a/b`,
   `ExR2/4`, `GLNO`, `LNO2`, `SpsP`, `AN07B037`, `AN06B009`; expose per-side splits and an EPG
   heading proxy from the bump.
2. **Runtime readout (`brain.py`).** Add `BrainRuntime.circuit_readout()` beside `readouts`
   (`brain.py:132`). The decoder feature list (`brain.py:138`) is untouched, so the
   "only neurons reach the decoder" contract holds.
3. **`fly-drone circuit-probe` (`cli.py`).** (a) Under arena/sensory stimuli, report whether the
   EPG heading, hΔ and PFNd readouts track yaw and translation, and whether they change when the
   light/loom pathways are silenced. (b) A type-level in/out synapse count among these cells as an
   independent structural check of findings 1–4 on _our_ bundle. Column-offset replication
   (finding 1's 82%) additionally needs MaleCNS fan-shaped-body column labels, which our bundle
   does not carry; that is a separate optional import, not assumed here.
4. **Viewer + telemetry (`server.py:380`, web).** A `circuit` block on the telemetry frame and a
   panel showing the compass and vector populations. Extends the existing "who is flying"
   attribution by naming the navigation cells, not just motor types.
5. **Tests.** Type resolution counts, readout determinism against a fixed seed, and a probe smoke
   test on the existing fixture/assay path.

The payoff is that any future claim about heading or homing can be checked causally (silence the
pathway, watch the readout and the behaviour) rather than inferred from motor output alone.

## 6. Path integration: a future milestone, not scheduled work

The finding is the strongest external case for a capability our connectome cannot express. If it is
ever pursued, it is a **contract change** and requires explicit user sign-off. Outline only:

- **What it needs.** An opt-in, versioned plasticity mode: a frozen-parameter, three-factor rule
  (hΔB pre-activity × dopamine gate → fast weight tag; octopamine reset), off by default so every
  legacy and free-roam actor and result stays reproducible. "Frozen parameters" would mean the
  _rule's constants_ are fixed from the finding, not learned; the brain would still not be
  gradient-trained.
- **Why it is risky.** The biology is untested (no recording at the synapses, no dopamine/
  octopamine action on them); a strengthen-only rule saturates without a reset; the compass
  writing/reading semantics are themselves disputed (finding 4); and the whole thing must still
  pass a causal test — silence the store and homing must disappear, with a ghost/blind control.
- **How it would be validated.** Add a beacon "depot" the drone can leave; check straight return
  after the beacon goes out of view; ablate hΔH/hΔI/FC2/PFL3 and the modulator cells; require the
  ghost condition to fail. None of this is started.

Until then the brain stays strictly frozen, and path integration is recorded here as the clearest
candidate for a future "state in synapses" milestone.

## 7. References

Findings and links are collected in
[`references.md`](references.md#connectome-mining-and-navigation-circuit-external).

Prior work the findings build on, as named there: Maimon & Abbott 2026 (columnar vector memories
feeding FC2); Hulse et al. 2021 (fan-shaped-body anatomy, the two-arbor motif); Lu 2022 / Lyu 2022
(hΔB travel-direction bump and PFNd velocity); Janke 2025 (hΔG leaky integrator reset at food);
Avritzer 2026 (hΔA 7–10 s travel-direction memory); Hulse et al. 2023 (PS196 as GLNO's largest
input); Fisher 2022 (ExR2 gates compass learning by rotation speed); Mussells Pires 2024 /
Westeinde 2024 (FC2/PFL steering); Turner-Evans et al. 2020, Duan et al. 2025 and Eddy et al. 2026
(compass ring and its inhibition).
