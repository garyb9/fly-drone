# References

## Data and upstream code

- **MaleCNS v1.0**, FlyEM / University of Cambridge / MRC LMB / Google Research, CC-BY 4.0.
  <https://male-cns.janelia.org/download/>. See [`data-pipeline.md`](data-pipeline.md).
- **fly-playground**: source of the Rust LIF core, data bundle and fly body equations.
  <https://github.com/garyb9/fly-playground>. Its design, neuron-model and data-pipeline docs
  are the background for this project. Copy provenance is in
  [`source-provenance.json`](source-provenance.json).
- **MuJoCo-drones-gym** (tau-intelligence): CF2X parameters, force model and cascaded PID. Pinned
  as the `vendor/mujoco-drones` submodule.
  <https://github.com/tau-intelligence/MuJoCo-drones-gym>
- **MuJoCo** physics engine. <https://mujoco.org>
- **NeuroMechFly** fly mesh (`web/public/fly.glb`, Apache-2.0). See
  `web/public/FLY-ATTRIBUTION.md`.

## Neuron model

- W. Gerstner, W. Kistler, R. Naud, L. Paninski, _Neuronal Dynamics_, ch. 1.3 (leaky
  integrate-and-fire). <https://neuronaldynamics.epfl.ch/online/Ch1.S3.html>
- Box–Muller transform for Gaussian noise: G. E. P. Box, M. E. Muller, "A note on the generation
  of random normal deviates", _Ann. Math. Statist._ 29 (1958).

## Vision and looming

- LC4 / LPLC2 looming-sensitive visual projection neurons and the giant-fiber escape pathway are
  the anatomical basis for the looming role. The primary references cited in
  `data/malecns/sensory-mappings.json` are
  <https://www.nature.com/articles/nature13427> and
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC6533146/>.
- Time-to-contact `τ = θ/θ̇`: the classical optical variable for approach (see
  [`sensory-model.md`](sensory-model.md) §2).

## Control and physics

- J. Förster, _System Identification of the Crazyflie 2.0 Nano Quadrocopter_, ETH Zürich (2015):
  the origin of the rotor drag coefficients.
- D. Mellinger, V. Kumar, "Minimum snap trajectory generation and control for quadrotors", ICRA
  (2011): the differential-flatness attitude construction used by the cascaded PID.

## Learning

- J. Schulman et al., "Proximal Policy Optimization Algorithms" (2017).
  <https://arxiv.org/abs/1707.06347>
- J. Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage
  Estimation" (2016). <https://arxiv.org/abs/1506.02438>
- A. Y. Ng, D. Harada, S. Russell, "Policy invariance under reward transformations: theory and
  application to reward shaping", ICML (1999): potential-based shaping.
- A. Raffin et al., "Stable-Baselines3: Reliable Reinforcement Learning Implementations", _JMLR_
  22 (2021). <https://jmlr.org/papers/v22/20-1364.html>
- E. B. Wilson, "Probable inference, the law of succession, and statistical inference", _JASA_
  22 (1927): the binomial interval used in [`training.md`](training.md) §6.

## Connectome mining and navigation circuit (external)

Connectome-mining findings on central-complex navigation, and the sibling whole-brain emulators.
These are **not** validated biology; each source page labels its claims as published, measured in
wiring, simulated, or proposed and untested. See
[`connectome-navigation-findings.md`](connectome-navigation-findings.md) for the relevance filter
and what, if anything, this project adopts from them.

- **Fly circuit exploration** (pwang724), MaleCNS v1.0 + hemibrain navigation screen, September 2026. <https://github.com/pwang724/fly-circuit-exploration>.
  - Findings index: <https://pwang724.github.io/fly-circuit-exploration/findings/index.html>.
  - Panoramic circuit view: <https://pwang724.github.io/fly-circuit-exploration/circuit.html>.
  - Finding 1, synaptic store at hΔH/hΔI:
    <https://pwang724.github.io/fly-circuit-exploration/findings/01-synaptic-store.html>.
  - Finding 2, hΔM/hΔI return inverters:
    <https://pwang724.github.io/fly-circuit-exploration/findings/02-return-inverter.html>.
  - Finding 3, unnamed velocity inputs (PS196_b, FB3A):
    <https://pwang724.github.io/fly-circuit-exploration/findings/03-velocity-sources.html>.
  - Finding 4, EPG→PEN compass brake:
    <https://pwang724.github.io/fly-circuit-exploration/findings/04-compass-brake.html>.
- **fly-brain** (eonsystemspbc): MaleCNS/FlyWire whole-brain LIF emulation across Brian2,
  Brian2CUDA, PyTorch, NEST GPU and GeNN. <https://github.com/eonsystemspbc/fly-brain>.
- **flycoinrh** (fruitflydev): MaleCNS connectome driving a browser; the one place it lets a weight
  move is dopamine-gated Kenyon-cell→MBON depression.
  <https://github.com/fruitflydev/flycoinrh>.
- **fly-brain-spectacles** (PtPavloTkachenko): MaleCNS on Snap Spectacles, with a bit-exact Metal
  GPU brain kernel and an explicit "real vs. engineered" decisions log.
  <https://github.com/PtPavloTkachenko/fly-brain-spectacles>.

## Connectome-driven bodies (prior art)

Projects that put a fly connectome in a body. Each is surveyed, with what this project takes and what
it rejects, in [`external-prior-art.md`](external-prior-art.md); the harvest design is
[`superpowers/specs/2026-09-18-prior-art-harvest-design.md`](superpowers/specs/2026-09-18-prior-art-harvest-design.md).

- **FlyDrones** (SpikeCalls): frozen MaleCNS as a drone pilot; optic flow → T4/T5/LPLC2/LC4, ridge
  readout → RC sticks. <https://github.com/SpikeCalls/FlyDrones> (live browser demo:
  <https://spikecalls.github.io/FlyDrones/>).
- **fly.ai** (alextitonis): frozen MaleCNS reservoir (`flybrain` on PyPI), feature-detector input and
  a PCA+linear readout; publishes laterality checks and negative results. Source:
  <https://github.com/alextitonis/fly.ai>; package: <https://pypi.org/project/flybrain/0.1.0/>.
- **dylankainth/flybrain**: FlyWire female brain on a Tello, with an opt-in retinotopic eye map from
  Buchner (1971) ommatidial directions. <https://github.com/dylankainth/flybrain>.
- **fly-self-driving** (suanmiao): frozen MaleCNS adjacency (trained per-synapse gains) driving a
  simulated car with DAgger; shuffled-graph control. <https://github.com/suanmiao/fly-self-driving>.
- **FlyGM** (Jin, Zhu, Zhang, Sui): whole-brain connectome as a graph policy trained with RL on
  `flybody`. <https://arxiv.org/abs/2602.17997> · <https://lnsgroup.cc/research/FlyGM>.
- **DOOMFLY** (nftechie): frozen MaleCNS plus a dopamine-gated plasticity rule in a Doom arena;
  failed its validation gates. <https://github.com/nftechie/doomfly>.
- **flybrain-robot-bridge** (Frankweb33): a mock neural backend with a MaleCNS stub.
  <https://github.com/Frankweb33/flybrain-robot-bridge>.
- **Straw lab `drosophila_eye_map`**: digitized Buchner-1971 ommatidial directions (BSD).
  <https://github.com/strawlab/drosophila_eye_map>.
- **flyhard** (MarkUnthank): the connectome-as-network recipe fly-self-driving builds on.
  <https://github.com/MarkUnthank/flyhard>.
- **Eon Systems embodied brain emulation**:
  <https://eon.systems/updates/embodied-brain-emulation>.
