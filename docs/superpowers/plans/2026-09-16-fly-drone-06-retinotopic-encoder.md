# Retinotopic encoder v6 (sensing milestone) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 8-scalar learned encoder (v5) with a **retinotopic** one: a small CNN over
each eye's luma stack emits per-patch current maps into Tm4/T2 (motion/loom), Mi1/Tm3 (light) and
direct LC4/LPLC2, so the frozen connectome's own optic lobe computes motion and looming. Widen the
sensory bandwidth until the pre-registered E1 loom-selectivity gate is reachable.

**Architecture:** A new identity, `learned-v6:<hash>`, built from the already-committed, tested but
unwired modules `retinotopy.py` and `spatial_encoder.py`. `BrainRuntime` gains a per-patch
injection path (a `12x8` or `4x3` current map per population/side instead of one scalar per
population). The SAC wiring, clone warm start (v4 cues broadcast onto the spatial maps) and
`E1`/`E2` scoring are re-based on the flattened 816-current vector. The connectome and the decoder
interface (2,022 DN + VNC traces → 4 velocity commands) are unchanged. Nothing about v4 or v5
changes.

**Tech Stack:** Python 3.12, numpy 2.5, gymnasium 1.3, stable-baselines3 2.9 (SAC), torch 2.14
(CUDA on the RTX 4070), MuJoCo renderer, Rust `brain-core` + pyo3 `brain-python`.

**Specs:**

- [`docs/superpowers/specs/2026-09-16-retinotopic-sensing-v6-design.md`](../specs/2026-09-16-retinotopic-sensing-v6-design.md) — the sensing design (primary).
- [`docs/superpowers/specs/2026-09-16-wing-level-action-design.md`](../specs/2026-09-16-wing-level-action-design.md) — the action side; **separate follow-on plan**.
- [`docs/superpowers/specs/2026-09-15-training-improvements.md`](../specs/2026-09-15-training-improvements.md) — Pack 4 is subsumed here.
- [`docs/results/encoder-v5/RECOVERY-STATUS-2026-09-16.md`](../../results/encoder-v5/RECOVERY-STATUS-2026-09-16.md) and [`PHASE0-DIAGNOSTICS-2026-09-16.md`](../../results/encoder-v5/PHASE0-DIAGNOSTICS-2026-09-16.md) — why v5 stopped.

## Confirmed decisions (user, 2026-09-16)

1. **Sensing first, action later.** Wing-level action (W0–W3) becomes its own plan after M3.
2. **Clone init + alternating rounds first.** Joint encoder+decoder training (M4) only if the
   alternating schema plateaus, with E1/E2/E3 as hard anti-wire gates.
3. **Spec defaults:** `12x8` spatial maps, `4x3` direct maps; Tm4/T2/LC4/LPLC2 + Mi1/Tm3.
4. **E1 stays 0.8**, reported against the v4 baseline (0.687 policy-flown / 0.732 fixed probe). A
   miss is recorded as a fail, never relaxed.

## Global Constraints

- The connectome stays frozen: no change to wiring, weights, signs, neuron parameters or the tonic
  bias (`0.85` on DLMn/DVMn).
- The encoder's only input is its own eye frames. Currents are in `[0, 2]`, uniform within each
  patch. The deployed decoder's only input is the 2,022 DN + VNC motor traces.
- `learned-v6:<16 hex>` is a new identity; a v6 decoder pins it and cannot run against another
  encoder. v4 and v5 stay reproducible: `ENCODER_VERSION`, `learned-v5:`, the legacy room
  (`test_legacy_room_mjcf_unchanged`) and the v4/v5 replay goldens are byte-identical.
- `roam_eval.ACCEPTANCE` (A1–A7) is never edited. E1 AUC ≥ 0.8, E2 light margin ≥ 0.05 / loom
  margin ≥ 0.1. A miss is reported, never relaxed.
- Replay buffer 100 k transitions. Environment workers ≤ 6.
- Seeds never overlap: clone reuses v5's flights (600–631, 632–663); DAgger under clone 1200–1327 /
  2200–2327; validation 9000–9009; evaluation 1000–1049.
- **Ask the user before every long run**: clone fit, each SAC round, recovery DAgger, evaluation.
- Never edit `fly_drone/*.py` while a `sac-round` / `roam-*` subprocess is live.
- Never run `yarn py:build` / `yarn prep` / `yarn ci` while a simulation or training process runs.
- Commands strip ROS: `env -u PYTHONPATH .venv/bin/python …`. Format/lint:
  `.venv/bin/ruff format python tests && .venv/bin/ruff check python tests`. Rust:
  `cargo fmt --all && cargo test -p brain-core --locked`.
- Commit **and push** after each task. No agent trailer (user, 2026-09-16).

## File Structure

| File                                                                                | Responsibility                                                                                                       | Change |
| ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------ |
| `crates/brain-core/src/policy.rs`                                                   | Accept `learned-v6:` (generalise the learned prefix test)                                                            | modify |
| `python/fly_drone/brain.py`                                                         | v6 detection, per-patch roles, map injection, v6 pathway ids and silencing                                           | modify |
| `python/fly_drone/env.py`                                                           | `frame_transform` hook at both `push_frame` sites; v6 sensing order                                                   | modify |
| `python/fly_drone/sac.py`                                                           | v6 observation/step/spaces, v6 policy, spatial clone fit, export, `E1`/`E2`                                           | modify |
| `python/fly_drone/spatial_encoder.py`                                               | Channel-group flat slices and map mirror helpers                                                                     | modify |
| `python/fly_drone/spatial_clone.py`                                                 | Target-map mirror and left/right swap for v6 augmentation                                                            | modify |
| `python/fly_drone/cli.py`                                                           | v6 encoder-clone / sac-round / sac-export / sac-validate; `spatial-maps`                                             | modify |
| `tests/test_brain_v6.py`, `tests/test_sac_v6.py`, `tests/test_spatial_clone.py`     | New coverage                                                                                                         | create |
| `docs/sensory-model.md`, `docs/overview/README.md`, `docs/training.md`, `AGENTS.md` | v6 documentation                                                                                                     | modify |

Out of scope (separate plans): wing-level action (wrench), M4 joint training, live server/viewer.

---

## Part A — code (V6.0)

### Task 1: Rust accepts `learned-v6:`

**Files:**

- Modify: `crates/brain-core/src/policy.rs`

**Interfaces:**

- Produces: the learned prefix is accepted for both `learned-v5:` and `learned-v6:`; unknown
  prefixes and a bare `learned-` still fail. Python binding signatures unchanged.

- [ ] **Step 1: Write the failing test**

Append to the existing `learned_encoder_tests` module (or add one) in
`crates/brain-core/src/policy.rs`:

```rust
    #[test]
    fn accepts_learned_v5_and_v6_and_rejects_other_encoders() {
        assert!(Policy::from_json(&policy_json("learned-v5:0123456789abcdef", None)).is_ok());
        assert!(Policy::from_json(&policy_json("learned-v6:0123456789abcdef", None)).is_ok());
        assert!(Policy::from_json(&policy_json("learned-v4:0123", None)).is_err());
        assert!(Policy::from_json(&policy_json("learned-", None)).is_err());
    }
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `cargo test -p brain-core --locked accepts_learned_v5_and_v6`
Expected: FAIL on the `learned-v6:` line.

- [ ] **Step 3: Implement**

Replace the single `LEARNED_ENCODER_PREFIX` constant with a slice and adjust `from_json`:

```rust
/// Learned encoders pin their weights hash after this prefix; Python checks the exact match.
pub const LEARNED_ENCODER_PREFIXES: &[&str] = &["learned-v5:", "learned-v6:"];
```

```rust
        if p.encoder_version != ENCODER_VERSION
            && !LEARNED_ENCODER_PREFIXES
                .iter()
                .any(|prefix| p.encoder_version.starts_with(prefix))
```

- [ ] **Step 4: Run the Rust tests and rebuild if no sim is running**

```bash
cargo fmt --all && cargo test -p brain-core --locked
pgrep -f "python.*fly_drone" || yarn py:build
env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_runtime.py tests/test_env.py
```

Expected: all pass. If `pgrep` prints PIDs, stop and ask the user before rebuilding.

- [ ] **Step 5: Commit and push**

```bash
git add crates/brain-core/src/policy.rs
git commit -m "Accept learned-v6 encoder versions in Rust" && git push
```

---

### Task 2: Spatial encoder helpers (groups + mirror)

**Files:**

- Modify: `python/fly_drone/spatial_encoder.py`
- Modify: `tests/test_spatial_encoder.py`

**Interfaces:**

- Consumes: `CHANNEL_ORDER`, `channel_grid`, `flat_dim`, `flatten`, `unflatten` (already present).
- Produces (in `fly_drone.spatial_encoder`):
  - `GROUP_CHANNELS = {"light": ("mi1", "tm3"), "motion": ("tm4", "t2"), "loom": ("lc4", "lplc2")}`.
  - `group_slices() -> {group: np.ndarray of flat indices}` into `flatten()` order.
  - `mirror_maps(maps) -> dict`: each `_l` map becomes the width-flipped `_r` map and vice versa,
    matching the bilateral symmetry the clone augmentation relies on.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_spatial_encoder.py`:

```python
def test_group_slices_partition_the_flat_vector():
    from fly_drone import spatial_encoder as se

    slices = se.group_slices()
    merged = np.concatenate([slices[g] for g in slices])
    assert sorted(merged.tolist()) == list(range(se.flat_dim()))
    assert set(slices) == {"light", "motion", "loom"}
    # light = mi1+tm3 both sides; loom = lc4+lplc2 both sides
    assert len(slices["light"]) == 4 * 12 * 8
    assert len(slices["loom"]) == 4 * 4 * 3


def test_mirror_maps_swaps_and_flips_eyes():
    from fly_drone import spatial_encoder as se

    out = {
        name: np.arange(int(np.prod(se.channel_grid(name))), dtype=np.float32).reshape(
            se.channel_grid(name)
        )
        for name in se.CHANNEL_ORDER
    }
    mirrored = se.mirror_maps(out)
    for name in se.CHANNEL_ORDER:
        other = name[:-1] + ("r" if name.endswith("_l") else "l")
        np.testing.assert_array_equal(mirrored[name], out[other][:, ::-1])
```

Add `import numpy as np` if not already imported.

- [ ] **Step 2: Run them to make sure they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_spatial_encoder.py`
Expected: FAIL with `AttributeError: module 'fly_drone.spatial_encoder' has no attribute 'group_slices'`.

- [ ] **Step 3: Implement**

Add near `channel_grid` in `spatial_encoder.py`:

```python
# Channel prefixes grouped by function, for metabolic cost, E1/E2 and silencing.
GROUP_CHANNELS = {"light": ("mi1", "tm3"), "motion": ("tm4", "t2"), "loom": ("lc4", "lplc2")}


def group_slices():
    """Flat indices of each functional group into the ``flatten`` order."""
    out = {group: [] for group in GROUP_CHANNELS}
    start = 0
    for name in CHANNEL_ORDER:
        size = int(np.prod(channel_grid(name)))
        prefix = name.rsplit("_", 1)[0]
        for group, prefixes in GROUP_CHANNELS.items():
            if prefix in prefixes:
                out[group].extend(range(start, start + size))
        start += size
    return {group: np.asarray(ids, dtype=np.int64) for group, ids in out.items()}


def mirror_maps(maps):
    """Mirror a channel map dict: left and right swap, width flips (bilateral symmetry)."""
    out = {}
    for name in CHANNEL_ORDER:
        other = name[:-1] + ("r" if name.endswith("_l") else "l")
        out[name] = np.asarray(maps[other])[..., ::-1]
    return out
```

- [ ] **Step 4: Run the tests**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_spatial_encoder.py`
Expected: all pass.

- [ ] **Step 5: Commit and push**

```bash
.venv/bin/ruff format python tests && .venv/bin/ruff check python tests
git add python/fly_drone/spatial_encoder.py tests/test_spatial_encoder.py
git commit -m "Add v6 channel-group and mirror helpers for the spatial encoder" && git push
```

---

### Task 3: `BrainRuntime` v6 path

**Files:**

- Modify: `python/fly_drone/brain.py`
- Create: `tests/test_brain_v6.py`

**Interfaces:**

- Consumes: `retinotopy.build_default_maps`, `retinotopy.define_roles`, `retinotopy.apply_map`,
  `SpatialEncoder` (Task 2 helpers).
- Produces (in `fly_drone.brain`):
  - Constructor accepts a `SpatialEncoder` instance or `.pt` path; `self.encoder_version` becomes
    `learned-v6:<hash>`. `self.maps` (channel → `RetinotopicMap`), `self.roles` (channel →
    `{(bx, by): role}`), `self.current_maps` (channel → `(nx, ny)`).
  - `set_currents(vector)`: v6 clips a flat `(816,)` vector to `[0,2]` and unflattens it.
  - `encode_stack()`: v6 sets `current_maps` from `self.encoder.currents(stack)`.
  - `step()`: v6 injects every patch via `retinotopy.apply_map`.
  - `pathway_ids` on v6: `light` (mi1+tm3), `motion` (tm4+t2), `looming` (lc4+lplc2) per side, so
    `silence_inputs` and `encoder_checks` keep working.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_brain_v6.py`:

```python
import numpy as np
import pytest
from fly_drone.brain import BrainRuntime
from fly_drone.spatial_encoder import SpatialEncoder, flat_dim


@pytest.fixture(scope="module")
def v6(tmp_path_factory):
    path = tmp_path_factory.mktemp("v6") / "encoder.pt"
    enc = SpatialEncoder.fresh(seed=0)
    enc.save(path)
    return BrainRuntime(encoder=path)


def test_v6_roles_cover_every_injectable_cell_in_its_channel(v6):
    from fly_drone import spatial_encoder as se
    from fly_drone.retinotopy import build_default_maps

    maps = build_default_maps(v6.cells)
    for channel in se.CHANNEL_ORDER:
        assigned = sorted(c for ids in v6.roles[channel].values() for c in ids)
        expected = sorted(maps[channel].all_cells())
        assert assigned == expected


def test_v6_equal_currents_reach_the_right_cells(v6):
    v6.reset(3)
    vector = np.full(flat_dim(), 0.7, np.float32)
    v6.set_currents(vector)
    v6.step(1)
    assert v6.current_maps["tm4_l"].shape == (12, 8)


def test_v6_set_currents_clips_and_checks_shape(v6):
    with pytest.raises(ValueError):
        v6.set_currents(np.zeros(8, np.float32))
    out = v6.set_currents(np.full(flat_dim(), 3.0, np.float32))
    assert out.max() == 2.0


def test_v4_role_set_is_unchanged():
    brain = BrainRuntime()
    assert tuple(brain.input_ids) == ("light_l", "light_r", "looming_l", "looming_r")
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_brain_v6.py`
Expected: FAIL — the constructor does not detect the v6 encoder.

- [ ] **Step 3: Implement in `brain.py`**

At the top, import the v6 modules lazily inside the learned branch to avoid an import cycle
(`spatial_encoder` imports `brain.STACK_FRAMES`). Add a helper and branch:

```python
    def _load_v6(self, encoder):
        from .retinotopy import build_default_maps, define_roles
        from .spatial_encoder import CHANNEL_ORDER, SpatialEncoder, unflatten

        if not isinstance(encoder, SpatialEncoder):
            encoder = SpatialEncoder.load(encoder)
        self.encoder = encoder
        self.encoder_version = encoder.version
        self.maps = build_default_maps(self.cells)
        self.roles = {
            channel: define_roles(self.core, self.maps[channel], namespace=f"v6_{channel}")
            for channel in CHANNEL_ORDER
        }
        self.current_maps = {
            channel: np.zeros((self.maps[channel].nx, self.maps[channel].ny), np.float32)
            for channel in CHANNEL_ORDER
        }
        self.input_ids = {
            channel: [c for ids in self.roles[channel].values() for c in ids]
            for channel in CHANNEL_ORDER
        }
        for group in ("light", "motion", "looming"):
            ...
```

`pathway_ids` should map `light_l/light_r` to mi1+tm3 cells, `motion_l/motion_r` to tm4+t2, and
`looming_l/looming_r` to lc4+lplc2 (use `spatial_encoder.GROUP_CHANNELS`). `set_currents` detects
`self.current_maps` and calls `unflatten`. `step` injects via `apply_map` when v6. Follow the exact
pattern of the v5 branch (`brain.py:85–104`, `:179–201`) and keep the v4/v5 paths untouched.

- [ ] **Step 4: Run the v6 tests and the v4/v5 goldens**

Run:
`env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_brain_v6.py tests/test_brain_v5.py tests/test_env.py tests/test_runtime.py`
Expected: all pass.

- [ ] **Step 5: Commit and push**

```bash
.venv/bin/ruff format python tests && .venv/bin/ruff check python tests
git add python/fly_drone/brain.py tests/test_brain_v6.py
git commit -m "Wire the v6 spatial encoder into BrainRuntime" && git push
```

---

### Task 4: `ConnectomeEnv` frame-transform hook and v6 sensing order

**Files:**

- Modify: `python/fly_drone/env.py`
- Modify: `tests/test_brain_v6.py`

**Interfaces:**

- Consumes: `augment.make_transform` (already present), `BrainRuntime.learned`.
- Produces: `ConnectomeEnv(..., frame_transform=None)`; when set, it is applied to
  `plant.camera()` at both `push_frame` call sites (`env.py:153`, `env.py:385`). Default `None`
  is the identity, so v4/v5 goldens are unchanged. v6 uses the v5 sensing order.

- [ ] **Step 1: Write the failing test**

```python
def test_frame_transform_changes_the_stack_but_not_the_v4_path(tmp_path):
    import numpy as np
    from fly_drone.brain import BrainRuntime
    from fly_drone.env import ConnectomeEnv

    def darken(images):
        return (images // 2).astype(np.uint8)

    plain = ConnectomeEnv(task="free_roam", level=0, brain=BrainRuntime())
    dark = ConnectomeEnv(task="free_roam", level=0, brain=BrainRuntime(), frame_transform=darken)
    try:
        plain.reset(seed=5)
        dark.reset(seed=5)
        np.testing.assert_array_equal(plain.brain.cues, dark.brain.cues)  # v4 ignores the hook
    finally:
        plain.close()
        dark.close()
```

Also assert in a v6 env that the stack is the transformed camera frame (compare to `darken`).

- [ ] **Step 2: Run it to make sure it fails**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_brain_v6.py -k transform`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'frame_transform'`.

- [ ] **Step 3: Implement**

Add `frame_transform=None` to `__init__`, store `self.frame_transform = frame_transform`, and add:

```python
    def _frame(self):
        images = self.plant.camera()
        return images if self.frame_transform is None else self.frame_transform(images)
```

Use `self._frame()` at both `push_frame` sites (`env.py:153`, `env.py:385`). The v4 `sense` path
keeps reading `self.plant.camera()` directly, so v4 is untouched.

- [ ] **Step 4: Run the tests and the replay goldens**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_brain_v6.py tests/test_env.py`
Expected: all pass, both `replay_is_bit_identical` goldens unchanged.

- [ ] **Step 5: Commit and push**

```bash
.venv/bin/ruff format python tests && .venv/bin/ruff check python tests
git add python/fly_drone/env.py tests/test_brain_v6.py
git commit -m "Add the eye-frame transform hook and the v6 sensing order" && git push
```

---

### Task 5: SAC v6 environment and policy

**Files:**

- Modify: `python/fly_drone/sac.py`, `python/fly_drone/spatial_policy.py`
- Create: `tests/test_sac_v6.py`

**Interfaces:**

- Consumes: `SpatialEncoder`/`flat_dim`/`unflatten` (Task 2), `BrainRuntime` v6 (Task 3),
  `SpatialSACPolicy` (already present).
- Produces:
  - `learner_spaces` returns action `(flat_dim(),)` when the SAC env's encoder is v6.
  - `SacRoamEnv._obs` returns the eye stack; `step` maps `action + 1` through `unflatten` into
    `brain.set_currents`; metabolic cost uses the v6 `light`/`loom` group slices.
  - `build_sac` selects `SpatialSACPolicy` for a v6 encoder; the critic is unchanged
    (`CriticExtractor` keeps its own v5 `EyeNet`, training-only).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sac_v6.py` with a v6 encoder fixture and tests for the action space shape, the
metabolic cost at all-ones (loom group mean 2.0 → `0.01*2` etc.), actor key isolation, and a
one-step `build_sac` smoke with `SpatialSACPolicy`.

- [ ] **Step 2: Run them to make sure they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_sac_v6.py`
Expected: FAIL — the action space is still 8-D.

- [ ] **Step 3: Implement**

Thread a `spatial: bool` (or `action_dim`) through `learner_spaces`, `SacRoamEnv.__init__` and
`build_sac`, defaulting to the v5 values so existing callers are unchanged. In `SacRoamEnv.step`
add the v6 branch using `unflatten(action + 1.0)` and `group_slices()` for cost. In `build_sac` set
`policy_kwargs` to use `SpatialSACPolicy` when spatial.

**Design point (settle here):** `SpatialActor` sets `mu = nn.Identity()`, which forces the last
`net_arch["pi"]` layer to `flat_dim()` (and an `816x816` `log_std`). Benchmark parameter count and
step time; if it is heavy, use a compact latent and a shared linear head instead, and record the
choice in the commit message. Keep `log_std` clamped to `[LOG_STD_MIN, LOG_STD_MAX]`.

- [ ] **Step 4: Run the tests**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_sac_v6.py tests/test_sac.py`
Expected: all pass.

- [ ] **Step 5: Commit and push**

```bash
.venv/bin/ruff format python tests && .venv/bin/ruff check python tests
git add python/fly_drone/sac.py python/fly_drone/spatial_policy.py tests/test_sac_v6.py
git commit -m "Add the v6 SAC environment, policy and critic wiring" && git push
```

---

### Task 6: Spatial clone fit (v4 cues → v6 target maps)

**Files:**

- Modify: `python/fly_drone/sac.py`, `python/fly_drone/spatial_clone.py`
- Modify: `tests/test_spatial_clone.py`

**Interfaces:**

- Consumes: v5 clone npz (`stacks`, `cues`, `encoder_version == ENCODER_VERSION`),
  `spatial_clone.v4_cues_to_targets`, `spatial_encoder.mirror_maps` (Task 2).
- Produces: `spatial_clone.mirror_targets(targets)` and `fit_spatial_clone(paths, output,
  steps=60000, batch=256, holdout=0.1, device=None, seed=0)` writing `encoder.pt` +
  `clone.json` with per-channel and per-group (`light`/`motion`/`loom`) held-out MSE/r.

- [ ] **Step 1: Write the failing tests**

Synthetic npz → `fit_spatial_clone(..., steps=5, device="cpu")` produces a loadable
`SpatialEncoder` whose `version` starts with `learned-v6:` and a report with the group keys; a
mirrored target batch equals the mirrored prediction target.

- [ ] **Step 2: Run them to make sure they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_spatial_clone.py`
Expected: FAIL with `AttributeError: … has no attribute 'fit_spatial_clone'`.

- [ ] **Step 3: Implement**

Prediction is `tanh(flatten(net(eyes))) + 1`; the loss is per-patch MSE over all channels. Mirror a
random `MIRROR_PROB` half of each batch with `spatial_encoder.mirror_maps` on the eye stack and the
targets (the v4 clone's mirror + difference-weight lessons carry over; the target maps are
piecewise-constant per channel, so the difference term is implicit in the per-patch MSE). The Tm4/T2
targets stay at the `neutral=1.0` baseline.

- [ ] **Step 4: Run the tests**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_spatial_clone.py tests/test_encoder.py`
Expected: all pass.

- [ ] **Step 5: Commit and push**

```bash
.venv/bin/ruff format python tests && .venv/bin/ruff check python tests
git add python/fly_drone/sac.py python/fly_drone/spatial_clone.py tests/test_spatial_clone.py
git commit -m "Add the v4-cue to v6 spatial clone fit" && git push
```

---

### Task 7: v6 export, warm start and `E1`/`E2`

**Files:**

- Modify: `python/fly_drone/sac.py`, `python/fly_drone/roam_eval.py`
- Modify: `tests/test_sac_v6.py`, `tests/test_roam_eval.py`

**Interfaces:**

- Consumes: `SpatialEncoder.save/load`, existing `export_decoder` / `warm_start_decoder`.
- Produces: a v6 encoder export (`encoder.pt`, `learned-v6:`); a decoder export pinning that
  version; `encoder_checks` computing `E1` loom AUC and `E2` margins over the v6 `loom`/`light`
  group means, using the `teacher` fixed probe by default.

- [ ] **Step 1: Write the failing tests**

A synthetic encoder with loom-group currents correlated with the threat-positive labels passes E1;
a constant encoder fails. Export/load keeps the version and currents.

- [ ] **Step 2: Run them to make sure they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_sac_v6.py tests/test_roam_eval.py`
Expected: FAIL on the v6 E1 path.

- [ ] **Step 3: Implement**

Reuse `spatial_encoder.group_slices()` for the loom/light axes; keep `E1`/`E2` bars and the
`E1_policy`/`E2_policy` secondary numbers.

- [ ] **Step 4: Run the tests**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_sac_v6.py tests/test_roam_eval.py`
Expected: all pass.

- [ ] **Step 5: Commit and push**

```bash
.venv/bin/ruff format python tests && .venv/bin/ruff check python tests
git add python/fly_drone/sac.py python/fly_drone/roam_eval.py tests/test_sac_v6.py tests/test_roam_eval.py
git commit -m "Export, warm-start and score the v6 encoder" && git push
```

---

### Task 8: CLI and map inspector

**Files:**

- Modify: `python/fly_drone/cli.py`
- Modify: `tests/test_cli.py` (or the closest CLI test file)

**Interfaces:**

- Produces: `--encoder-version v6` routing for `encoder-clone`, `sac-round`, `sac-export`,
  `sac-validate`; a `spatial-maps` command wrapping `scripts/spatial_maps.py`.

- [ ] **Step 1: Write the failing tests** (parser accepts the new flag/command; v5 invocation
  unchanged).
- [ ] **Step 2: Run them to make sure they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests.**
- [ ] **Step 5: Commit and push** (`git commit -m "Expose the v6 encoder path through the CLI"`).

---

### Task 9: v6 smoke and documentation

- [ ] v6 smoke round (9k frames, warm-up 6k, 6 workers): record fps, injection cost, RAM; confirm
  `learned-v6:` round-trips and the pinned entropy regime logs (`ent_coef=0.01`,
  `log_std_mean=-2.5`).
- [ ] Update `docs/sensory-model.md` §6, `docs/overview/README.md` §7.4, `docs/training.md` §9,
  `AGENTS.md`.
- [ ] Commit: `Document the retinotopic encoder v6`.

---

## Part B — runs (V6.1 → V6.2; each gated on explicit user approval)

### Task 10: v6 clone and round-0 decoder (V6.1a)

- [ ] Fit the spatial clone on the existing 64 v5 flights (`runs/v5/clone/data.npz`,
  `data2.npz`); record held-out per-channel and mirror consistency.
- [ ] Round-0 decoder under the v6 clone: teacher-only first; add a DAgger iteration under the clone
  (β 0.5, seeds 1200–1327) if the sanity gate fails — the v5 lesson is that the decoder must fly
  under the encoder it runs with.
- [ ] **Gate:** 30-seed beacons/min ≥ 0.8 × the v4 reference (1.547), same pre-registered ratio.
  Stop and report if not.

### Task 11: one alternating V6.1 round (V6.1b)

- [ ] Encoder SAC 350k staged (stage A 100k + mid-gate, stage B 350k) then decoder SAC 150k, pinned
  entropy regime, guard active.
- [ ] **Gate:** fixed-probe `E1` > v4's 0.732, `E2` pass, ghost ≤ 0.3, guard foraging holds.
  Report against the 0.8 bar. If `E1` moves but causal dodge does not, proceed to Task 12.

### Task 12: training signal (V6.2, only if causal dodge is absent)

- [ ] Threats-without-pillars curriculum level (**needs explicit sign-off** — touches
  pre-registered `arena.LEVELS`), potential-based threat shaping, recovery DAgger on L3, n-step.
- [ ] **Gate:** A3 causal dodge (near ≥ 0.8, ghost ≤ 0.3) + `E2` pass.

---

## Risks and open items

- **Spatial current too weak** (Tm4 → LC4 ≈ 0.29 at full drive): the per-patch output gain is a
  tunable; the direct LC4/LPLC2 route guarantees a channel.
- **`E1` unattainable**: v6 keeps the 0.8 bar; a miss is a reported fail and a user decision.
- **Injection cost**: ~816 roles; inject once per camera frame and benchmark against v5.
- **`SpatialActor` parameter count**: settle `Identity` vs compact latent in Task 5.
- **Sequence**: Task 12 depends on Task 11's result; M4 joint training and wing-level action are
  separate plans.
