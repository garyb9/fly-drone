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
