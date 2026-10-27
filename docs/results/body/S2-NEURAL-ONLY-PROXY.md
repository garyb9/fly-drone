# S2 neural-only proxy staging package

This package adds an isolated diagnostic body path to **the existing fly-playground working tree**. It is not a portable backend, renderer integration, biomechanical model, or locomotion demonstration.

`src/body/neural-only.ts` and its test require the exact dependencies recorded in [s2-dependency-snapshot.json](s2-dependency-snapshot.json). The source HEAD alone is insufficient: articulation and rig sources are currently untracked; config, types, and package metadata have user modifications. Preserve that working tree and verify every SHA-256 before reproducing this result. Do not substitute HEAD copies, silently upgrade dependencies, or commit unrelated source files as part of publishing this package.

Publication consists of adding the two new source files in their intended sibling paths, plus retaining this dependency manifest with the S2 evidence. No existing sibling source file needs changing. The existing renderer is not connected to the new class. A future caller may consume `renderState()`; it returns a copy of the actual articulation state without an independent animation clock.

The isolated fixture at `/tmp/fly-neural-only-test.DYaCPU` copies the recorded dependency subset and uses the existing node_modules. Verification used Vitest with one worker and no cache, TypeScript `--noEmit`, and Prettier. Eight tests cover force accounting, zero-drive behavior, motor selectivity, decay, replay, boundary termination, and malformed/configuration-mismatched restores.

Snapshot identity contains the exact serialized bounds, proxy calibration, current physics configuration, and full rig data. Restore checks structure, finite numbers, identity, passive-only state invariants, actuator consistency, quaternion normalization, and termination consistency before replacing state. This runtime identity supplements, rather than replaces, the source dependency hashes.

Unresolved mechanics are deliberate: no contact/adhesion forces, no coxa mapping, no joint reaction forces or muscle inertia, and proxy wing lift/torque. Leg and head axes are kinematic activation-relaxation mappings. The force ledger accounts only for rigid-body wing forces, gravity, and drag. S3 must connect the canonical fly-drone neural runtime to a pinned MuJoCo body and replace these proxies with declared actuator/contact mechanics. Nothing here demonstrates walking, grooming, feeding competence, or biological fidelity.
