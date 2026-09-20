# C3b (excitability re-derivation) dropped — no defensible resting baseline

**Status:** recorded 2026-09-20. Decision, not a run. This is B2 of
[`../../superpowers/specs/2026-09-20-ongoing-state-and-faithful-readout.md`](../../superpowers/specs/2026-09-20-ongoing-state-and-faithful-readout.md) §8.

B2 asked for the per-neuron `threshold_scale` (currently the hand-picked `0.5` in
`dynamics.py:EXCITABLE_PARAMS`) to be re-derived so the **resting** network matches a stated,
cited baseline-activity statistic, or dropped with a reason if no such statistic exists. It does
not survive that test.

## Why there is no value to derive

A global `threshold_scale` shifts the excitability of every neuron uniformly. The only published
whole-brain characterisations of *Drosophila* resting-state activity describe the opposite of a
uniform shift:

- Li, Ping, Zhang & Wang, **"Connectome-constrained modeling identifies neurons and synapses
  that sustain spontaneous activity in *Drosophila*"**, bioRxiv 2026
  (<https://doi.org/10.64898/2026.08.21.745055>): spontaneous activity is "not distributed
  uniformly across the connectome, but is organized by a compact neuropil core", sustained by a
  "highly sparse, brain-spanning ensemble of inhibitory hub neurons".
- Tainton-Heap et al. and the near-whole-brain calcium recordings (Mann et al. 2017;
  Aimon et al. 2019) report spontaneous activity as *sparse* and state-dependent, not a single
  population fraction.

None of these pins a per-neuron threshold or a whole-brain rate that maps onto this simulator's
`mean_feature_activity`. The scalar `0.5` has no anchor, and scaling every threshold would
flatten exactly the sparse, structured resting dynamics the connectome is reported to support —
the opposite of a faithful derivation. Per B2.3, an unanchored scalar chosen because behaviour
improved would be fitting the brain to the task through the back door, which the parent spec's §1
exists to prevent.

## Decision

- **C3b is not built.** No `threshold_scale` is adopted as a C3 identity. The canonical bundle's
  per-neuron dynamics remain untouched.
- The existing `EXCITABLE_PARAMS` / `data/malecns-excitable` machinery and the P3
  `adapter-excitable.json` are **kept as recorded exploratory artifacts** (their gate is already a
  recorded negative/partial), but `threshold_scale = 0.5` is documented as a hand-picked P3
  tuning knob, **not** a prior.
- **Pre-registered falsifier, if a usable baseline is ever published:** the value must be derived
  from a stated resting statistic with a citation, recorded in the bundle marker, and the derived
  bundle must reproduce that statistic under `scripts/readout_audit.py`'s rest run. No behavioural
  score may appear in the derivation. Until then, B2 stays dropped.

C3a (tonic drive on identified motoneurons) is built separately and carries its own
`silence_tonic` causal gate.
