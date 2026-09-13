# Sources and licenses

- Fly runtime and copied TypeScript references: https://github.com/garyb9/fly-playground, MIT. `source-provenance.json` records the source working tree, including uncommitted changes, and SHA-256 of copied originals. Subsequent project adaptations are tracked in this repository.
- MaleCNS v1.0: FlyEM / University of Cambridge / MRC LMB / Google Research, CC-BY 4.0. See `data/malecns/ATTRIBUTION.md` and source hashes in its manifest.
- Drone simulator: https://github.com/tau-intelligence/MuJoCo-drones-gym, pinned git submodule `ca170dc6760d422e47ef46fb3af46f8cd20410eb`. Its pyproject declares MIT; the inspected tree has no standalone license file. No upstream source is modified. Meshes in that dependency originate from MuJoCo Menagerie's Bitcraze Crazyflie model; upstream maintains their source notices.
- Fly GLB: adapted NeuroMechFly asset copied from fly-playground. See `web/public/FLY-ATTRIBUTION.md` and `web/public/LICENSE-NeuroMechFly.txt` for upstream revision and Apache-2.0 license. The scanned body is illustrative and does not claim to be the MaleCNS specimen.
- Three.js MIT; MuJoCo Apache-2.0; Gymnasium MIT; Stable-Baselines3 MIT; PyO3 Apache-2.0/MIT. Dependency licenses remain with their packages.
