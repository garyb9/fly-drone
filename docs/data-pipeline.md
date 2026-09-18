# Data: the MaleCNS bundle

The connectome in `data/malecns/` is **built by fly-playground's pipeline and copied here
unchanged**. This repository does not re-run the pipeline. It verifies the files and binds
every policy to them.

## 1. Files

| File                    | Size   | Contents                                                                                                        |
| ----------------------- | ------ | --------------------------------------------------------------------------------------------------------------- |
| `neurons.bin`           | 3.9 MB | 166,700 × 24-byte records: body id, position, group id, flags (format: [`neuron-model.md`](neuron-model.md) §5) |
| `graph.bin`             | 61 MB  | CSR graph: 10,520,431 edges, `i16` weights, `w_norm = 0.003`                                                    |
| `cells.json`            | 20 MB  | per-neuron `id`, `type`, `side`, `group`, soma position, measured flag                                          |
| `groups.json`           | 4 KB   | group names and readout roles (`wing_l/r`, `thrust`, `escape`)                                                  |
| `sensory-mappings.json` | 28 KB  | annotated Mi1/Tm3 light inputs and JO-E wind inputs by side, with references                                    |
| `manifest.json`         | 4 KB   | counts, sources and SHA-256 hashes, coordinate transform, model/selection statements                            |
| `ATTRIBUTION.md`        | —      | CC-BY 4.0 credit and a list of changes from the source release                                                  |

## 2. Provenance

Source: MaleCNS v1.0 (FlyEM / University of Cambridge / MRC LMB / Google Research), CC-BY 4.0,
<https://male-cns.janelia.org/download/>. `manifest.json` records the flat-connectome feather
files it used, with SHA-256:

| Source file                                            | Use                               |
| ------------------------------------------------------ | --------------------------------- |
| `body-annotations-male-cns-v1.0-minconf-0.5.feather`   | class, type, side, group per body |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | synapse counts per ordered pair   |
| `body-neurotransmitters-male-cns-v1.0.feather`         | dominant transmitter → sign       |

Selection and transforms (from `manifest.json` → `selection`, `model`, `coordinateTransform`):

1. Keep every body with a non-null superclass: **166,700 neurons**.
2. Keep edges with **≥ 3 synaptic contacts**: **10,520,431 edges**.
3. Order neurons with the 1,585-cell compact circuit first (`core_count`), then the rest by
   weighted degree.
4. Quantise weights: `q = round(count / w_norm)` stored as `i16`, then `w_sim = q · w_norm` at
   load.
5. Sign: ACh `+1`; GABA and Glu `−1`; unresolved or modulatory sources inject no current.
6. Scene coordinates from voxel coordinates:
   `p_scene = (p_voxel − [48000, 26000, 28000]) · 2.5e−5 · [1, −1, −1]`.
7. 139,662 neurons have measured somata. The rest stay in the simulation and are hidden in
   geometry (flag bit 16). No soma positions are fabricated.

`manifest.json` also records `directLoomingToEscape = 308` cells and `11,220` direct
looming→escape synapses, a structural sanity statistic for the LC4/LPLC2 pathway.

## 3. Identity hash: binding policies to the graph

```
dataset_hash = SHA-256(graph.bin)
             = 60cb182151d80d910a9f23a1550908fefe2233234fa7f33bb7bc0f3d70311c39   (current)
```

`BrainRuntime` computes this at load. `actor.json` and `calibration.npz` store it. The Rust
policy loader and `warm_start` refuse to run on a mismatch. Any graph change therefore
invalidates old decoders, which is the intent: a decoder is only meaningful for the neurons it
was trained on.

A second, stronger identity covers the whole model, because a sign variant changes `neurons.bin`
(where each neuron's inhibitory flag lives) while leaving `graph.bin` identical:

```
bundle_hash  = SHA-256(graph.bin ‖ neurons.bin ‖ canonical(model manifest fields))
             = edc5439e291e65233e673b5baaed069f570f5003c245e414de2aa1e41b14aff5   (current)
```

`python/fly_drone/identity.py` computes both. `dataset_hash` is kept unchanged so every accepted
actor still loads; `bundle_hash` is enforced only for alternate bundles (a manifest carrying
`wiring` or `sign_convention`), whose actors must pin it. See the harvest spec
[`superpowers/specs/2026-09-18-prior-art-harvest-design.md`](superpowers/specs/2026-09-18-prior-art-harvest-design.md)
Step 0.

## 4. Test fixture

`pipeline/out/fixture/` holds a tiny synthetic graph in the same binary format. It is produced by
fly-playground's `pipeline/gen_fixture.py` and used only by
`crates/brain-core/tests/golden_trace.rs`. It must not be used for behaviour.

## 5. Refreshing the data

```bash
# in fly-playground
yarn data:build                      # fetch feathers, filter, emit binaries (≈2 GB download)
# then, here
cp ../fly-playground/public/data/malecns/full/{neurons.bin,graph.bin,cells.json,groups.json,manifest.json} data/malecns/
yarn assay                           # causal gate on the new graph
```

After a refresh:

1. Re-run calibration and training, because the dataset hash changes.
2. Update `docs/source-provenance.json`.
3. Re-run `yarn ci`. The Rust golden trace uses the fixture, so it is unaffected. The Python
   tests load the real bundle.
