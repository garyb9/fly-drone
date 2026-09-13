# NeuroMechFly visual body

Geometry and assembly data: NeLy-EPFL / NeuroMechFly, distributed with
[FlyGym](https://github.com/NeLy-EPFL/flygym) under Apache-2.0.
The full license is included in `LICENSE-NeuroMechFly.txt`.

Source revision: `38c8ec61034cd59bc5ba0de20688d4a3c0000d60`.
Sources: `src/flygym/assets/model/neuromechfly/meshes/simplified_max2000faces`,
`rigging.yaml`, and `pose/neutral/yaw_pitch_roll.yaml` at that revision.

This is a micro-CT-derived **female** Drosophila body, used as an illustrative
visual shell for our MaleCNS simulation. It is not the MaleCNS specimen, and
the rendering is not a validated musculoskeletal simulation.

Modifications by fly-playground: quadric mesh simplification, mirrored right
parts, neutral-pose assembly, rigid-part merging, conversion to glTF, authored
pigment colors, scaling and rotation to the app coordinate frame, thin wing
membranes following the source outlines, authored veins, and cosmetic wing/leg
animation. Right joint transforms are reflected from the left for bilateral
symmetry. No site code or derivative assets from The palm or Fly Escape
are included. Far and near assets share joint pivots and original part surfaces.

Rebuild with `pipeline/build_fly_model.py`. `manifest.json` records source input
hashes, the pinned revision, output hashes, byte sizes, and triangle counts.

Please also cite the upstream publications when using this model in research:
[NeuroMechFly](https://neuromechfly.org/) links the original model and
NeuroMechFly v2 publications.
