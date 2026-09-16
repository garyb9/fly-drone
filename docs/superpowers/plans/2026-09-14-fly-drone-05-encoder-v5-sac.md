# Encoder v5 (learned, SAC) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a small CNN, trained with SAC, turn the drone's two camera images into 8 bounded
currents into the fly's eye-input cells, so the frozen connectome can tell a thrown ball from
self-motion. A decoder trained in alternating rounds then turns DN activity into flight.

**Architecture:** `BrainRuntime` gains an optional learned encoder, with 8 input roles by cell type
and a 3-frame luma stack. With no encoder it stays byte-for-byte v4. `sac.py` wraps
`ConnectomeEnv` as a SAC environment whose learner is the encoder (decoder frozen in the loop), the
decoder (encoder frozen in the loop) or a brain-bypass control. An asymmetric SB3 policy lets the
actor see only its deployable input (eyes, or DN traces), while the critic also sees DN traces and
visible-only geometry. Decoders export to the Rust actor JSON with a new `tanh` output mode, and
encoders save as `.pt` files with a content hash that becomes the pinned encoder version.

**Tech Stack:** Python 3.12, numpy 2.5, gymnasium 1.3, stable-baselines3 2.9 (SAC), torch 2.14
(CUDA on the RTX 4070), MuJoCo renderer, Rust `brain-core` + pyo3 `brain-python`.

**Spec:** [`docs/superpowers/specs/2026-09-14-learned-encoder-sac-design.md`](../specs/2026-09-14-learned-encoder-sac-design.md) (as amended in c9b3cf2). Read it first.

## Global Constraints

- The connectome stays frozen: no change to wiring, weights, signs, neuron parameters or the tonic bias (`0.85` on DLMn/DVMn).
- The deployed encoder's only input is the two camera images (its own 3-frame luma stack). The deployed decoder's only input is the 2,022 DN + VNC motor traces.
- The critic is training-only. Its extra inputs are DN traces and geometry **only while the object is visible** (in the field of view and unoccluded, via `teacher.visible` / `env.beacon_visible`).
- Currents are in `[0, 2]`, uniform within each of 8 populations: `mi1_l, mi1_r, tm3_l, tm3_r, lc4_l, lc4_r, lplc2_l, lplc2_r`.
- Metabolic cost: `λ_loom = 0.01` on mean(LC4, LPLC2 currents), `λ_light = 0.002` on mean(Mi1, Tm3 currents), encoder learner only.
- v4 stays reproducible: `BrainRuntime()` with no encoder, the legacy room (`test_legacy_room_mjcf_unchanged`) and accepted actors must behave bit-identically. v4 artifacts keep `ENCODER_VERSION = "bright-contrast-400-splay075-noaa-loom150-v4"`.
- Learned encoder versions are `"learned-v5:" + 16 hex chars` (sha256 of the weights). An actor loads only into a runtime with the same encoder version.
- `roam_eval.ACCEPTANCE` (A1–A7) is never edited. E1–E4 are added separately, with bars E1 AUC ≥ 0.8, E2 light margin ≥ 0.05 and loom margin ≥ 0.1.
- Replay buffer 100 k transitions. Environment workers ≤ 6 (the machine has 23 GB RAM, and each brain + renderer is ≈ 0.5 GB).
- Seeds never overlap: DAgger iteration k from `200 + 1000·k` (128 flights), clone collection 600–631, evaluation 1000–1049, round validation 9000–9009.
- **Ask the user before every long run**: stage 1 DAgger, clone collection, each SAC round, the bypass run, final evaluation.
- Never run `yarn py:build`/`yarn prep`/`yarn ci` while a simulation or training process is running (rebuilding the extension crashes it).
- Commands strip ROS: `env -u PYTHONPATH .venv/bin/python -m pytest ...`. Format and lint with `.venv/bin/ruff format python tests && .venv/bin/ruff check python tests`. Rust uses `cargo fmt --all && cargo test -p brain-core --locked`, then `yarn py:build`.
- When a step says "append to" an existing test file and the snippet starts with imports, merge those imports into the file's top import block (ruff sorts them). Only the test functions go at the end.
- Commit **and push** after each task. Commit messages end with the session's `Co-Authored-By` / `Claude-Session` trailer lines.

## File Structure

| File                                                                                | Responsibility                                                                                                            | Change |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------ |
| `crates/brain-core/src/policy.rs`                                                   | Accept `learned-v5:` encoder versions; optional `output: "tanh"`                                                          | modify |
| `python/fly_drone/brain.py`                                                         | `FrameStack`, 8 roles, per-runtime `encoder_version`, `set_currents`/`push_frame`/`encode_stack`                          | modify |
| `python/fly_drone/encoder.py`                                                       | `EyeNet`, `EyesExtractor`, `LearnedEncoder` (save/load/hash), `v4_targets`, clone collect + fit                           | create |
| `python/fly_drone/env.py`                                                           | v5 sensing order (encode at step start, push frame at step end, zero currents while settling)                             | modify |
| `python/fly_drone/sac.py`                                                           | `visible_geometry`, `SacRoamEnv`, extractors, `AsymmetricSACPolicy`, decoder export/warm start, `train_round`, `validate` | create |
| `python/fly_drone/distill.py`                                                       | Pass `encoder` through screen jobs; `bypass:` controller; shared class weights                                            | modify |
| `python/fly_drone/roam_eval.py`                                                     | `ENCODER_CHECKS`, `roc_auc`, `encoder_checks` (E1/E2), `encoder` through probes and evaluation                            | modify |
| `python/fly_drone/cli.py`                                                           | `encoder-collect`, `encoder-clone`, `sac-init-decoder`, `sac-round`, `sac-validate`, `encoder-checks`, `--encoder`        | modify |
| `tests/test_brain_v5.py`                                                            | Roles, stack, v4 equivalence, version checks                                                                              | create |
| `tests/test_encoder.py`                                                             | Network shapes, save/load hash, clone job and fit                                                                         | create |
| `tests/test_sac.py`                                                                 | Geometry, env spaces and cost, actor isolation, export parity, one-round smoke                                            | create |
| `tests/test_env.py`, `tests/test_roam_eval.py`                                      | v4 replay golden hashes; AUC and E-checks                                                                                 | modify |
| `docs/sensory-model.md`, `docs/overview/README.md`, `docs/training.md`, `AGENTS.md` | v5 documentation                                                                                                          | modify |

Out of scope (a follow-up plan once a v5 actor passes A1–A6 and E1–E2): loading v5 in the live server and viewer.

---

### Task 1: Golden replay hashes for v4 (before any change)

Everything later must leave v4 untouched. First record what v4 does today.

**Files:**

- Modify: `tests/test_env.py`

**Interfaces:**

- Produces: `V4_LOOMING_REPLAY`, `V4_ROAM_REPLAY` constants; `_replay_digest(task, seed, frames)` helper.

- [ ] **Step 1: Add the helper and tests with placeholder constants**

Add `import hashlib` to the import block at the top of `tests/test_env.py`, then append:

```python
V4_LOOMING_REPLAY = "fill-me"
V4_ROAM_REPLAY = "fill-me"


def _replay_digest(task, seed, frames, **kwargs):
    env = ConnectomeEnv(task=task, **kwargs)
    try:
        obs, _ = env.reset(seed=seed)
        digest = hashlib.sha256(np.round(obs, 6).tobytes())
        for _ in range(frames):
            obs, *_ = env.step(np.array([0.3, -0.2, 0.1, 0.4]))
            digest.update(np.round(obs, 6).tobytes())
            digest.update(np.round(env.brain.cues, 6).tobytes())
        return digest.hexdigest()
    finally:
        env.close()


def test_v4_looming_replay_is_bit_identical():
    assert _replay_digest("looming", 7, 20) == V4_LOOMING_REPLAY


def test_v4_free_roam_replay_is_bit_identical():
    assert _replay_digest("free_roam", 7, 20, level=3, respawn=True) == V4_ROAM_REPLAY
```

- [ ] **Step 2: Print the digests on unchanged code and paste them in**

Run:

```bash
env -u PYTHONPATH .venv/bin/python -c "
import sys; sys.path.insert(0, 'tests')
from test_env import _replay_digest
print(_replay_digest('looming', 7, 20))
print(_replay_digest('free_roam', 7, 20, level=3, respawn=True))
print(_replay_digest('looming', 7, 20))"
```

Expected: three hex digests, with the first and third identical (determinism). Replace the two `"fill-me"` strings with the first two digests.

- [ ] **Step 3: Run the tests**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_env.py -k replay_is_bit_identical`
Expected: 2 passed.

- [ ] **Step 4: Commit and push**

```bash
.venv/bin/ruff format tests && .venv/bin/ruff check tests
git add tests/test_env.py && git commit -m "Pin v4 looming and free-roam replays before encoder v5 work" && git push
```

---

### Task 2: Rust accepts learned encoders and a tanh output layer

**Files:**

- Modify: `crates/brain-core/src/policy.rs`

**Interfaces:**

- Produces: JSON field `"output": "clip" | "tanh"` (default `clip`); `encoder_version` may be the v4 constant or start with `learned-v5:`. Python binding signatures are unchanged.

- [ ] **Step 1: Write the failing Rust tests**

Append to `crates/brain-core/src/policy.rs`:

```rust
#[cfg(test)]
mod learned_encoder_tests {
    use super::*;

    fn policy_json(encoder: &str, output: Option<&str>) -> String {
        let mut v = serde_json::json!({
            "version": 1, "encoder_version": encoder, "dataset_hash": "h",
            "feature_ids": [0], "mean": [0.0], "scale": [1.0],
            "layers": [{"weights": [[3.0], [0.5], [0.0], [-3.0]], "bias": [0.0, 0.0, 0.0, 0.0]}],
            "action_limits": [1.0, 2.0, 1.0, 1.0]
        });
        if let Some(o) = output {
            v["output"] = serde_json::json!(o);
        }
        v.to_string()
    }

    #[test]
    fn accepts_v4_and_learned_v5_and_rejects_other_encoders() {
        assert!(Policy::from_json(&policy_json(ENCODER_VERSION, None)).is_ok());
        assert!(Policy::from_json(&policy_json("learned-v5:0123456789abcdef", None)).is_ok());
        assert!(Policy::from_json(&policy_json("learned-v4:0123", None)).is_err());
        assert!(Policy::from_json(&policy_json("something-else", None)).is_err());
        assert!(Policy::from_json(&policy_json(ENCODER_VERSION, Some("relu"))).is_err());
    }

    #[test]
    fn default_output_clips_and_tanh_output_squashes() {
        let clip = Policy::from_json(&policy_json(ENCODER_VERSION, None)).unwrap();
        assert_eq!(clip.infer(&[1.0]).unwrap(), [1.0, 1.0, 0.0, -1.0]);
        let tanh = Policy::from_json(&policy_json(ENCODER_VERSION, Some("tanh"))).unwrap();
        let out = tanh.infer(&[1.0]).unwrap();
        let expected = [3.0f32.tanh(), 0.5f32.tanh() * 2.0, 0.0, -(3.0f32.tanh())];
        for (a, b) in out.iter().zip(expected) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `cargo test -p brain-core --locked learned_encoder_tests`
Expected: FAIL. The learned version is rejected, `"relu"` is accepted (unknown field ignored), and tanh is not applied.

- [ ] **Step 3: Implement**

In `policy.rs`, below `ENCODER_VERSION`:

```rust
/// Learned encoders (v5) pin their weights hash after this prefix; Python checks the exact match.
pub const LEARNED_ENCODER_PREFIX: &str = "learned-v5:";

/// Squashing of the final layer: PPO actors clip, SAC actors use tanh.
#[derive(Serialize, Deserialize, Default, Clone, Copy, PartialEq, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Output {
    #[default]
    Clip,
    Tanh,
}
```

Add to `struct Policy`, after `action_limits`:

```rust
    #[serde(default)]
    pub output: Output,
```

In `from_json`, replace `if p.encoder_version != ENCODER_VERSION` with:

```rust
        if (p.encoder_version != ENCODER_VERSION
            && !p.encoder_version.starts_with(LEARNED_ENCODER_PREFIX))
```

(keep the remaining `|| ...` conditions and add the closing parenthesis only around the version test). In `infer`, replace the final `Ok(std::array::from_fn(...))` with:

```rust
        let output = self.output;
        Ok(std::array::from_fn(|i| {
            let v = match output {
                Output::Clip => x[i].clamp(-1.0, 1.0),
                Output::Tanh => x[i].tanh(),
            };
            v * self.action_limits[i]
        }))
```

- [ ] **Step 4: Run the Rust tests, rebuild, run the Python runtime tests**

Run:

```bash
cargo fmt --all && cargo test -p brain-core --locked
pgrep -f "python.*fly_drone" || yarn py:build
env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_runtime.py tests/test_env.py
```

Expected: all pass, including both replay goldens (clip is the default, so v4 actors are unchanged). If `pgrep` prints PIDs, stop and ask the user before rebuilding.

- [ ] **Step 5: Commit and push**

```bash
git add crates/brain-core/src/policy.rs
git commit -m "Accept learned-v5 encoder versions and tanh actor outputs in Rust" && git push
```

---

### Task 3: `BrainRuntime` with 8 cell-type roles, a frame stack and encoder versions

**Files:**

- Modify: `python/fly_drone/brain.py`
- Create: `tests/test_brain_v5.py`

**Interfaces:**

- Consumes: Rust prefix `learned-v5:` (Task 2).
- Produces (in `fly_drone.brain`):
  - constants `STACK_FRAMES = 3`, `EYE_SHAPE = (48, 64)`, `LEARNED_PREFIX = "learned-v5:"`, `LEARNED_EXTERNAL = "learned-v5:external"`, `V5_CHANNELS = ("mi1_l", "mi1_r", "tm3_l", "tm3_r", "lc4_l", "lc4_r", "lplc2_l", "lplc2_r")`, `V4_TO_V5 = (0, 1, 0, 1, 2, 3, 2, 3)` (index into v4 cues `[light_l, light_r, loom_l, loom_r]`).
  - `luma_u8(images: uint8 (2,48,64,3)) -> uint8 (2,48,64)`
  - `class FrameStack`: `push(images)`, `clear()`, `array() -> uint8 (6,48,64)` (left eye oldest→newest, then right eye).
  - `BrainRuntime(seed=42, data=DATA, encoder=None)`. `encoder` is `None` (v4), `"external"` (currents set by a learner), a `.pt` path, or a `LearnedEncoder` (Task 4).
  - attributes and methods: `encoder_version: str`, `learned: bool`, `stack: FrameStack`, `set_currents(currents)`, `push_frame(images)`, `encode_stack() -> np.ndarray`, `load_policy(path, check_encoder=True)`. `silence_inputs` also accepts `light_l/light_r/looming_l/looming_r` on v5 runtimes.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_brain_v5.py`:

```python
import json

import numpy as np
import pytest
from fly_drone.brain import (
    ENCODER_VERSION,
    LEARNED_EXTERNAL,
    V4_TO_V5,
    V5_CHANNELS,
    BrainRuntime,
    FrameStack,
    luma_u8,
)


@pytest.fixture(scope="module")
def v4():
    return BrainRuntime()


@pytest.fixture(scope="module")
def v5():
    return BrainRuntime(encoder="external")


def test_luma_matches_the_rust_encoder_weights():
    images = np.zeros((2, 48, 64, 3), np.uint8)
    images[0, ..., 0] = 255
    images[1, ..., 1] = 100
    y = luma_u8(images)
    assert y.shape == (2, 48, 64) and y.dtype == np.uint8
    assert y[0, 0, 0] == round(0.2126 * 255) and y[1, 0, 0] == round(0.7152 * 100)


def test_frame_stack_fills_with_the_first_frame_then_shifts_and_clears():
    stack = FrameStack()
    assert stack.array().shape == (6, 48, 64) and not stack.array().any()
    frames = [np.full((2, 48, 64, 3), v, np.uint8) for v in (10, 20, 30, 40)]
    stack.push(frames[0])
    assert (stack.array()[[0, 1, 2]] == 10).all() and (stack.array()[3:] == 10).all()
    for f in frames[1:]:
        stack.push(f)
    assert [int(stack.array()[i, 0, 0]) for i in range(6)] == [20, 30, 40, 20, 30, 40]
    stack.clear()
    assert not stack.array().any()


def test_v5_roles_split_v4_roles_by_cell_type(v4, v5):
    assert tuple(v5.input_ids) == V5_CHANNELS
    for side in "lr":
        light = v5.input_ids[f"mi1_{side}"] + v5.input_ids[f"tm3_{side}"]
        loom = v5.input_ids[f"lc4_{side}"] + v5.input_ids[f"lplc2_{side}"]
        assert sorted(light) == sorted(v4.input_ids[f"light_{side}"])
        assert sorted(loom) == sorted(v4.input_ids[f"looming_{side}"])
    ids = [i for v in v5.input_ids.values() for i in v]
    assert len(ids) == len(set(ids))
    assert [len(v5.input_ids[c]) for c in V5_CHANNELS] == [886, 887, 1017, 1037, 71, 55, 94, 91]


def test_equal_split_currents_reproduce_the_v4_brain_exactly(v4, v5):
    rng = np.random.default_rng(3)
    v4.reset(11)
    v5.reset(11)
    for _ in range(5):
        cues = rng.uniform(0, 2, 4).astype(np.float32)
        v4.cues = cues
        v5.set_currents(cues[list(V4_TO_V5)])
        np.testing.assert_array_equal(v4.step(8), v5.step(8))


def test_versions_and_actor_encoder_check(v4, v5, tmp_path):
    assert v4.encoder_version == ENCODER_VERSION and not v4.learned
    assert v5.encoder_version == LEARNED_EXTERNAL and v5.learned
    n = len(v4.feature_ids)
    actor = {
        "version": 1,
        "encoder_version": ENCODER_VERSION,
        "dataset_hash": v4.dataset_hash,
        "feature_ids": v4.feature_ids,
        "mean": [0.0] * n,
        "scale": [1.0] * n,
        "layers": [{"weights": [[0.0] * n] * 4, "bias": [0.0] * 4}],
        "action_limits": [1.0] * 4,
    }
    path = tmp_path / "v4.json"
    path.write_text(json.dumps(actor))
    v4.load_policy(path)
    with pytest.raises(ValueError, match="encoder"):
        v5.load_policy(path)
    v5.load_policy(path, check_encoder=False)
    actor["encoder_version"] = "learned-v5:0123456789abcdef"
    path.write_text(json.dumps(actor))
    with pytest.raises(ValueError, match="encoder"):
        v4.load_policy(path)


def test_set_currents_clips_and_checks_shape_and_sense_is_v4_only(v5):
    v5.set_currents(np.array([-1, 0.5, 3, 1, 1, 1, 1, 1]))
    assert v5.cues.tolist() == [0.0, 0.5, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    with pytest.raises(ValueError):
        v5.set_currents(np.zeros(4))
    with pytest.raises(RuntimeError):
        v5.sense(np.zeros((2, 48, 64, 3), np.uint8))
    assert v5.encode_stack() is v5.cues  # external: the learner owns the currents


def test_pathway_silencing_names_work_on_v5(v5):
    v5.silence_inputs(("light_l", "looming_r"))
    v5.core.restore()
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_brain_v5.py`
Expected: FAIL with `ImportError: cannot import name 'LEARNED_EXTERNAL'`.

- [ ] **Step 3: Implement in `brain.py`**

Below `ENCODER_VERSION` add:

```python
LEARNED_PREFIX = "learned-v5:"
LEARNED_EXTERNAL = LEARNED_PREFIX + "external"
STACK_FRAMES = 3
EYE_SHAPE = (48, 64)
# Encoder v5 drives each anatomical input population with its own uniform current.
V5_CHANNELS = ("mi1_l", "mi1_r", "tm3_l", "tm3_r", "lc4_l", "lc4_r", "lplc2_l", "lplc2_r")
V5_TYPES = {"mi1": "Mi1", "tm3": "Tm3", "lc4": "LC4", "lplc2": "LPLC2"}
# v4 cues [light_l, light_r, loom_l, loom_r] copied onto the 8 v5 channels.
V4_TO_V5 = (0, 1, 0, 1, 2, 3, 2, 3)
PATHWAY_TYPES = {"light": ("mi1", "tm3"), "looming": ("lc4", "lplc2")}


def luma_u8(images):
    """Rec. 709 luma of packed RGB eyes, the same weights as the Rust v4 encoder."""
    rgb = np.asarray(images, dtype=np.float32)
    y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return np.round(y).astype(np.uint8)


class FrameStack:
    """The learned encoder's own recent frames; an empty stack repeats the first frame."""

    def __init__(self, frames=STACK_FRAMES):
        self.frames = frames
        self.buffer = None

    def clear(self):
        self.buffer = None

    def push(self, images):
        y = luma_u8(images)[:, None]
        if self.buffer is None:
            self.buffer = np.repeat(y, self.frames, axis=1)
        else:
            self.buffer = np.concatenate([self.buffer[:, 1:], y], axis=1)

    def array(self):
        if self.buffer is None:
            return np.zeros((2 * self.frames, *EYE_SHAPE), np.uint8)
        return self.buffer.reshape(2 * self.frames, *EYE_SHAPE).copy()
```

Change the constructor signature to `def __init__(self, seed=42, data=DATA, encoder=None):` and replace the block from `self.input_ids = {k: sensory[k] ...` through the `looming_` loop with:

```python
        self.encoder = encoder
        self.pathway_ids = {}
        if encoder is None:
            self.encoder_version = ENCODER_VERSION
            self.input_ids = {k: sensory[k] for k in ("light_l", "light_r")}
            for side in ("l", "r"):
                self.input_ids["looming_" + side] = self._cells(side, ("LC4", "LPLC2"))
        else:
            if isinstance(encoder, str) and encoder == "external":
                self.encoder_version = LEARNED_EXTERNAL
            else:
                from .encoder import LearnedEncoder

                if not isinstance(encoder, LearnedEncoder):
                    self.encoder = LearnedEncoder.load(encoder)
                self.encoder_version = self.encoder.version
            self.input_ids = {
                ch: self._cells(ch[-1], (V5_TYPES[ch[:-2]],)) for ch in V5_CHANNELS
            }
            for pathway, types in PATHWAY_TYPES.items():
                for side in ("l", "r"):
                    self.pathway_ids[f"{pathway}_{side}"] = sum(
                        (self.input_ids[f"{t}_{side}"] for t in types), []
                    )
```

Replace `self.cues = np.zeros(4, dtype=np.float32)` with:

```python
        self.cues = np.zeros(len(self.input_ids), dtype=np.float32)
        self.stack = FrameStack()
```

Add these methods to the class, and change `reset`, `sense`, `load_policy`, `silence_inputs` and `clear_vision_history` as shown:

```python
    def _cells(self, side, types):
        return [
            i
            for i, c in enumerate(self.cells)
            if c["side"].lower() == side and c["type"] in types
        ]

    @property
    def learned(self):
        return self.encoder is not None

    def reset(self, seed):
        self.core.reset(seed)
        self.core.bias(self.tonic, 0.85)
        self.tick = 0
        self.cues[:] = 0
        self.stack.clear()

    def sense(self, images):
        if self.learned:
            raise RuntimeError("learned encoders sense via push_frame + encode_stack")
        # (existing v4 body unchanged below)

    def set_currents(self, currents):
        currents = np.clip(np.asarray(currents, dtype=np.float32), 0.0, 2.0)
        if currents.shape != self.cues.shape:
            raise ValueError(f"expected {self.cues.shape[0]} currents")
        self.cues = currents
        return self.cues

    def push_frame(self, images):
        self.stack.push(images)

    def encode_stack(self):
        """Learned encoder: currents from the stack. External: the learner already set them."""
        if self.learned and not isinstance(self.encoder, str):
            self.set_currents(self.encoder.currents(self.stack.array()))
        return self.cues

    def load_policy(self, path, check_encoder=True):
        text = Path(path).read_text()
        version = json.loads(text)["encoder_version"]
        # A decoder reads activity shaped by one encoder; check_encoder=False only for a
        # frozen decoder inside an encoder-learning environment.
        if check_encoder and version != self.encoder_version:
            raise ValueError(
                f"actor encoder {version!r} does not match runtime {self.encoder_version!r}"
            )
        self.core.load_policy(text, self.dataset_hash, self.feature_ids)

    def silence_inputs(self, roles):
        """Silence one sensory pathway, e.g. ("looming_l", "looming_r")."""
        ids = {**self.input_ids, **self.pathway_ids}
        self.core.silence(sorted(set(sum((ids[r] for r in roles), []))), True)

    def clear_vision_history(self):
        """Forget previous frames so a respawn does not read as dark-area growth."""
        self.core.clear_vision_history()
        self.stack.clear()
```

Note: the v4 looming ids must come out in the same order as before. `_cells` is the same comprehension over `enumerate(self.cells)`, so role membership and order are unchanged.

- [ ] **Step 4: Run the new tests and the v4 goldens**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_brain_v5.py tests/test_env.py tests/test_runtime.py tests/test_distill.py`
Expected: all pass. `test_equal_split_currents_reproduce_the_v4_brain_exactly` proves the 8-way split alone changes nothing.

- [ ] **Step 5: Commit and push**

```bash
.venv/bin/ruff format python tests && .venv/bin/ruff check python tests
git add python/fly_drone/brain.py tests/test_brain_v5.py
git commit -m "Give BrainRuntime 8 cell-type input roles, a frame stack and encoder versions" && git push
```

---

### Task 4: `encoder.py`: eye network, learned encoder file, v4 clone data and fit

**Files:**

- Create: `python/fly_drone/encoder.py`
- Create: `tests/test_encoder.py`

**Interfaces:**

- Consumes: `FrameStack`, `STACK_FRAMES`, `EYE_SHAPE`, `LEARNED_PREFIX`, `V4_TO_V5`, `V5_CHANNELS`, `BrainRuntime(encoder=...)` (Task 3); `distill._roam_env`, `distill.NOISE_AXES`, `teacher.teacher_action`.
- Produces (in `fly_drone.encoder`):
  - `FEATURES = 64`; `eyes_space() -> Box uint8 (6,48,64)`; `v4_targets(cues[..., 4]) -> [..., 8]`
  - `EyeNet(frames)`: one eye (frames + side channel) → 64 features.
  - `EyesExtractor(observation_space: Dict with "eyes")`: SB3 features extractor → 128 features (shared `EyeNet`, right eye mirrored, side flag ±1).
  - `weights_hash(state) -> str` (16 hex chars).
  - `LearnedEncoder(extractor, mu)`: `.version`, `.currents(stack uint8 (6,48,64)) -> float32 (8,) in [0,2]`, `.save(path) -> version`, `LearnedEncoder.load(path)`, `LearnedEncoder.fresh(seed=0)`, `LearnedEncoder.from_actor(sb3_actor)`.
  - `_clone_job((seeds, seconds, level, noise)) -> (stacks, cues, flights)`; `collect_clone(path, flights=32, seconds=60, workers=6, seed_base=600, level=3, noise=0.2) -> dict`; `fit_clone(paths, output, steps=20000, batch=256, holdout=0.1, device=None, seed=0) -> dict` (writes `encoder.pt`, `clone.json`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_encoder.py`:

```python
import json

import numpy as np
import torch
from fly_drone import encoder
from fly_drone.brain import ENCODER_VERSION, BrainRuntime
from fly_drone.encoder import LearnedEncoder, v4_targets


def test_v4_targets_copy_light_to_mi1_tm3_and_loom_to_lc4_lplc2():
    cues = np.array([[0.1, 0.2, 0.3, 0.4]])
    assert v4_targets(cues).tolist() == [[0.1, 0.2, 0.1, 0.2, 0.3, 0.4, 0.3, 0.4]]


def test_fresh_encoder_outputs_eight_bounded_currents_and_128_features():
    enc = LearnedEncoder.fresh(seed=1)
    stack = np.random.default_rng(0).integers(0, 256, (6, 48, 64), dtype=np.uint8)
    currents = enc.currents(stack)
    assert currents.shape == (8,) and currents.dtype == np.float32
    assert (currents >= 0).all() and (currents <= 2).all()
    x = torch.zeros(2, 6, 48, 64)
    assert enc.extractor({"eyes": x}).shape == (2, 2 * encoder.FEATURES)


def test_save_load_round_trip_keeps_currents_and_version(tmp_path):
    enc = LearnedEncoder.fresh(seed=2)
    version = enc.save(tmp_path / "e.pt")
    assert version.startswith("learned-v5:") and len(version) == len("learned-v5:") + 16
    loaded = LearnedEncoder.load(tmp_path / "e.pt")
    stack = np.full((6, 48, 64), 77, np.uint8)
    np.testing.assert_array_equal(enc.currents(stack), loaded.currents(stack))
    assert loaded.version == version
    with torch.no_grad():
        loaded.mu.bias[0] += 0.1
    assert loaded.version != version


def test_brain_runtime_loads_a_learned_encoder_and_encodes_its_stack(tmp_path):
    enc = LearnedEncoder.fresh(seed=3)
    enc.save(tmp_path / "e.pt")
    brain = BrainRuntime(encoder=tmp_path / "e.pt")
    assert brain.learned and brain.encoder_version == enc.version
    brain.push_frame(np.full((2, 48, 64, 3), 90, np.uint8))
    np.testing.assert_allclose(brain.encode_stack(), enc.currents(brain.stack.array()))


def test_clone_job_records_one_stack_and_v4_cue_per_frame():
    stacks, cues, flights = encoder._clone_job(([5], 0.2, 3, 0.0))
    assert len(stacks) == len(cues) == len(flights) == 5
    assert stacks[0].shape == (6, 48, 64) and stacks[0].dtype == np.uint8
    assert cues[0].shape == (4,) and set(flights) == {5}


def test_fit_clone_writes_a_loadable_encoder_and_per_channel_report(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    path = tmp_path / "clone.npz"
    np.savez(
        path,
        stacks=rng.integers(0, 256, (n, 6, 48, 64), dtype=np.uint8),
        cues=rng.uniform(0, 2, (n, 4)).astype(np.float32),
        flight=np.repeat(np.arange(20), n // 20).astype(np.int32),
        encoder_version=ENCODER_VERSION,
    )
    report = encoder.fit_clone([path], tmp_path / "fit", steps=5, batch=16, device="cpu")
    assert set(report["held_out"]) == set(encoder.V5_CHANNELS)
    saved = json.loads((tmp_path / "fit" / "clone.json").read_text())
    assert saved["version"] == LearnedEncoder.load(tmp_path / "fit" / "encoder.pt").version
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_encoder.py`
Expected: FAIL with `ImportError: cannot import name 'encoder'`.

- [ ] **Step 3: Implement `python/fly_drone/encoder.py`**

```python
"""Learned eye encoder (v5): two cameras' 3-frame luma stacks -> 8 input currents.

The encoder sees only its own camera frames. It speaks to the frozen connectome by one
uniform, bounded current per anatomical input population (brain.V5_CHANNELS).
"""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from .brain import (
    ENCODER_VERSION,
    EYE_SHAPE,
    LEARNED_PREFIX,
    STACK_FRAMES,
    V4_TO_V5,
    V5_CHANNELS,
    FrameStack,
)

FEATURES = 64


def eyes_space(frames=STACK_FRAMES):
    return spaces.Box(0, 255, (2 * frames, *EYE_SHAPE), np.uint8)


def v4_targets(cues):
    return np.asarray(cues)[..., list(V4_TO_V5)]


class EyeNet(nn.Module):
    """One eye: frames + a side channel, 48x64 -> 24x32 -> 12x16 -> 6x8 -> 64 features."""

    def __init__(self, frames=STACK_FRAMES):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(frames + 1, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.dense = nn.Sequential(nn.Linear(32 * 6 * 8, FEATURES), nn.ReLU())

    def forward(self, x):
        return self.dense(self.conv(x))


class EyesExtractor(BaseFeaturesExtractor):
    """Both eyes through one EyeNet; the right eye is mirrored and flagged -1."""

    def __init__(self, observation_space, frames=STACK_FRAMES):
        super().__init__(observation_space, 2 * FEATURES)
        self.frames = frames
        self.eye = EyeNet(frames)

    def forward(self, observations):
        eyes = observations["eyes"]
        f = self.frames
        side = eyes.new_ones(eyes.shape[0], 1, *EYE_SHAPE)
        left = torch.cat([eyes[:, :f], side], 1)
        right = torch.cat([torch.flip(eyes[:, f:], dims=[3]), -side], 1)
        return torch.cat([self.eye(left), self.eye(right)], 1)


def weights_hash(state):
    digest = hashlib.sha256()
    for part in sorted(state):
        for key in sorted(state[part]):
            digest.update(f"{part}.{key}".encode())
            digest.update(state[part][key].detach().cpu().numpy().astype(np.float32).tobytes())
    return digest.hexdigest()[:16]


class LearnedEncoder:
    """Deployable encoder: SAC actor's eye extractor + mean head, tanh-squashed to [0, 2]."""

    def __init__(self, extractor, mu):
        self.extractor = extractor.cpu().eval()
        self.mu = mu.cpu().eval()

    @property
    def version(self):
        return LEARNED_PREFIX + weights_hash(self.state())

    def state(self):
        return {"extractor": self.extractor.state_dict(), "mu": self.mu.state_dict()}

    @classmethod
    def fresh(cls, seed=0):
        torch.manual_seed(seed)
        space = spaces.Dict({"eyes": eyes_space()})
        return cls(EyesExtractor(space), nn.Linear(2 * FEATURES, len(V5_CHANNELS)))

    @classmethod
    def from_actor(cls, actor):
        return cls(copy.deepcopy(actor.features_extractor), copy.deepcopy(actor.mu))

    @classmethod
    def load(cls, path):
        # One brain + renderer per worker process: keep torch to one thread each.
        torch.set_num_threads(1)
        state = torch.load(path, map_location="cpu", weights_only=True)
        enc = cls.fresh()
        enc.extractor.load_state_dict(state["extractor"])
        enc.mu.load_state_dict(state["mu"])
        return enc

    def currents(self, stack):
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(stack)[None], dtype=torch.float32) / 255.0
            out = torch.tanh(self.mu(self.extractor({"eyes": x}))) + 1.0
        return out[0].numpy().astype(np.float32)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state(), path)
        return self.version


def _clone_job(job):
    """Fly the teacher on v4; pair each frame's stack with the v4 cues it produced."""
    from .brain import BrainRuntime
    from .distill import NOISE_AXES, _roam_env
    from .teacher import teacher_action

    seeds, seconds, level, noise = job
    env = _roam_env(level, BrainRuntime())
    stack = FrameStack()
    stacks, cues, flights = [], [], []
    try:
        for seed in seeds:
            rng = np.random.default_rng(int(seed) + 7919)
            env.reset(seed=int(seed))
            stack.clear()
            for _ in range(int(seconds / 0.04)):
                action = teacher_action(env)[0].copy()
                action[list(NOISE_AXES)] += rng.normal(0, noise, len(NOISE_AXES))
                *_, info = env.step(np.clip(action, -1, 1))
                # plant.images holds the frame the v4 encoder read during this step.
                stack.push(env.plant.images)
                stacks.append(stack.array())
                cues.append(env.brain.cues.copy())
                flights.append(int(seed))
                if any(e["type"] == "collision" for e in info["events"]):
                    stack.clear()  # respawn cleared the v4 encoder's history too
    finally:
        env.close()
    return stacks, cues, flights


def collect_clone(path, flights=32, seconds=60, workers=6, seed_base=600, level=3, noise=0.2):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    seeds = np.arange(seed_base, seed_base + flights)
    jobs = [(c.tolist(), seconds, level, noise) for c in np.array_split(seeds, workers) if len(c)]
    stacks, cues, flight_ids = [], [], []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for s, c, f in pool.map(_clone_job, jobs):
            stacks += s
            cues += c
            flight_ids += f
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        stacks=np.asarray(stacks, dtype=np.uint8),
        cues=np.asarray(cues, dtype=np.float32),
        flight=np.asarray(flight_ids, dtype=np.int32),
        encoder_version=ENCODER_VERSION,
    )
    return {"frames": len(stacks), "flights": int(flights)}


def fit_clone(paths, output, steps=20000, batch=256, holdout=0.1, device=None, seed=0):
    """Supervised copy of v4 onto the learned encoder, so SAC starts from a working system."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    stacks, cues, flights = [], [], []
    for i, p in enumerate(paths):
        d = np.load(p)
        if str(d["encoder_version"]) != ENCODER_VERSION:
            raise ValueError(f"{p}: not collected on encoder v4")
        stacks.append(d["stacks"])
        cues.append(d["cues"])
        flights.append(d["flight"].astype(np.int64) + i * 1_000_000)
    stacks, flights = np.concatenate(stacks), np.concatenate(flights)
    targets = v4_targets(np.concatenate(cues)).astype(np.float32)
    rng = np.random.default_rng(72)
    unique = np.unique(flights)
    held = rng.choice(unique, max(1, int(len(unique) * holdout)), replace=False)
    test = np.isin(flights, held)
    train_ids, test_ids = np.flatnonzero(~test), np.flatnonzero(test)
    # Loom frames are rare: half of every batch comes from frames with loom input.
    active = train_ids[targets[train_ids, 4:].max(1) > 0.05]

    enc = LearnedEncoder.fresh(seed)
    net = nn.ModuleDict({"extractor": enc.extractor, "mu": enc.mu}).to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)

    def predict(ids):
        x = torch.as_tensor(stacks[ids], dtype=torch.float32, device=device) / 255.0
        return torch.tanh(net["mu"](net["extractor"]({"eyes": x}))) + 1.0

    for _ in range(steps):
        if len(active):
            ids = np.concatenate(
                [rng.choice(train_ids, batch // 2), rng.choice(active, batch - batch // 2)]
            )
        else:
            ids = rng.choice(train_ids, batch)
        loss = (predict(ids) - torch.as_tensor(targets[ids], device=device)).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    net.eval()
    with torch.no_grad():
        pred = np.concatenate(
            [predict(test_ids[i : i + 1024]).cpu().numpy() for i in range(0, len(test_ids), 1024)]
        )
    truth = targets[test_ids]
    report = {"frames": int(len(stacks)), "held_out_flights": int(len(held)), "held_out": {}}
    for k, name in enumerate(V5_CHANNELS):
        p, t = pred[:, k], truth[:, k]
        r = float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-9 and t.std() > 1e-9 else None
        report["held_out"][name] = {"mse": float(np.mean((p - t) ** 2)), "r": r}
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    report["version"] = LearnedEncoder(net["extractor"], net["mu"]).save(out / "encoder.pt")
    report["paths"] = [str(p) for p in paths]
    (out / "clone.json").write_text(json.dumps(report, indent=2))
    return report
```

- [ ] **Step 4: Run the tests**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_encoder.py tests/test_brain_v5.py`
Expected: all pass.

- [ ] **Step 5: Commit and push**

```bash
.venv/bin/ruff format python tests && .venv/bin/ruff check python tests
git add python/fly_drone/encoder.py tests/test_encoder.py
git commit -m "Add the learned eye encoder, its hashed file format and the v4 clone fit" && git push
```

---

### Task 5: v5 sensing order in `ConnectomeEnv`

The deployed encoder and an encoder being learned must see exactly the same frames at the
same moment. v5 therefore uses one rule everywhere. The currents applied during step `t` come
from the stack whose newest frame was rendered at the **end of step `t−1`**, after any respawn.
While settling after reset, currents are zero. v4 keeps its existing order (render at step start)
so the replay goldens stay identical.

**Files:**

- Modify: `python/fly_drone/env.py` (`reset`, `step`; the room-task body moves into `_step_room`)
- Modify: `python/fly_drone/roam_eval.py` (`_probe_setup` pushes a frame after teleporting)
- Modify: `tests/test_brain_v5.py`

**Interfaces:**

- Consumes: `BrainRuntime.learned`, `push_frame`, `encode_stack`, `set_currents` (Task 3); `LearnedEncoder` (Task 4).
- Produces: `ConnectomeEnv._sense()`, `ConnectomeEnv._step_room(action)`. Public `reset`/`step` signatures are unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_brain_v5.py`:

```python
from fly_drone.brain import luma_u8
from fly_drone.encoder import LearnedEncoder
from fly_drone.env import ConnectomeEnv


def test_learned_env_settles_on_zero_and_encodes_the_previous_end_of_step_frame(tmp_path):
    enc = LearnedEncoder.fresh(seed=4)
    enc.save(tmp_path / "e.pt")
    brain = BrainRuntime(encoder=tmp_path / "e.pt")
    env = ConnectomeEnv(task="free_roam", level=3, respawn=True, brain=brain)
    try:
        env.reset(seed=21)
        assert not brain.cues.any()
        for _ in range(3):
            expected = enc.currents(brain.stack.array())
            env.step(np.array([0.5, 0.0, 0.0, 0.2]))
            np.testing.assert_allclose(brain.cues, expected)
            newest = brain.stack.array()[[2, 5]]
            np.testing.assert_array_equal(newest, luma_u8(env.plant.images))
    finally:
        env.close()


def test_external_env_applies_the_learners_currents_unchanged(v5):
    env = ConnectomeEnv(task="free_roam", level=3, respawn=True, brain=v5)
    try:
        env.reset(seed=22)
        currents = np.linspace(0, 2, 8).astype(np.float32)
        v5.set_currents(currents)
        env.step(np.zeros(4))
        np.testing.assert_array_equal(v5.cues, currents)
    finally:
        env.close()


def test_v5_env_supports_pathway_ablations(v5):
    for ablation in ("light", "loom", "ghost"):
        env = ConnectomeEnv(task="free_roam", level=3, brain=v5, ablation=ablation)
        try:
            env.reset(seed=23)
            env.step(np.zeros(4))
        finally:
            env.brain.core.restore()
            env.close()
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_brain_v5.py -k "env"`
Expected: FAIL with `RuntimeError: learned encoders sense via push_frame + encode_stack`.

- [ ] **Step 3: Implement in `env.py`**

In `reset`, replace `self.brain.sense(self.plant.camera())` with:

```python
        if self.brain.learned:
            self.brain.push_frame(self.plant.camera())
            # Settle on zero input: identical for a deployed and a learning encoder.
            self.brain.set_currents(np.zeros_like(self.brain.cues))
        else:
            self.brain.sense(self.plant.camera())
```

Add the method:

```python
    def _sense(self):
        """v4 renders now; v5 encodes the stack rendered at the end of the previous step."""
        if self.brain.learned:
            self.brain.encode_stack()
        else:
            self.brain.sense(self.plant.camera())
```

In `step`, replace `self.brain.sense(self.plant.camera())` with `self._sense()`. Replace the
lines from `self.frames += 1` to the end of `step` so they read:

```python
        self.frames += 1
        if self.roam is not None:
            result = self._step_roam(action)
        else:
            result = self._step_room(action)
        if self.brain.learned:
            # The next step's currents come from what the eyes see now, after any respawn.
            self.brain.push_frame(self.plant.camera())
        return result

    def _step_room(self, action):
        pos = self.plant.pos[0]
        # ... the existing body from `delta = self.plant.target - pos` through its
        # `return (self.observe(), float(reward), terminated, ..., self.info())`, unchanged.
```

Move that body verbatim. Only its first line (`pos = self.plant.pos[0]`) now sits in the new method.

- [ ] **Step 4: Push a frame after probe teleports**

In `roam_eval._probe_setup`, directly after `env.brain.clear_vision_history()` add:

```python
    if env.brain.learned:
        env.brain.push_frame(env.plant.camera())
```

- [ ] **Step 5: Run the tests, including the v4 goldens**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_brain_v5.py tests/test_env.py tests/test_roam_eval.py tests/test_arena.py`
Expected: all pass. The two `replay_is_bit_identical` goldens must pass unchanged.

- [ ] **Step 6: Commit and push**

```bash
.venv/bin/ruff format python tests && .venv/bin/ruff check python tests
git add python/fly_drone/env.py python/fly_drone/roam_eval.py tests/test_brain_v5.py
git commit -m "Sense through the learned encoder's stack with one frame order for training and deployment" && git push
```

---

### Task 6: `sac.py`: visible geometry, SAC environment, asymmetric policy

**Files:**

- Create: `python/fly_drone/sac.py`
- Create: `tests/test_sac.py`

**Interfaces:**

- Consumes: `BrainRuntime(encoder=...)`, `V5_CHANNELS` (Task 3); `EyesExtractor`, `FEATURES`, `eyes_space` (Task 4); `ConnectomeEnv` with the v5 order (Task 5); `teacher.visible`.
- Produces (in `fly_drone.sac`):
  - `GEOMETRY = 8`, `LAMBDA_LOOM = 0.01`, `LAMBDA_LIGHT = 0.002`, `LEARNERS = ("encoder", "decoder", "bypass")`, `ACTOR_KEYS = {"encoder": "eyes", "decoder": "dn", "bypass": "currents"}`, `LIGHT`, `LOOM` (index lists into `V5_CHANNELS`).
  - `visible_geometry(env) -> float32 (8,)`: `[threat_visible, tx, ty, tz, beacon_visible, bx, by, bz]`, body frame, zeros when not visible.
  - `SacRoamEnv(learner, decoder=None, encoder=None, level=3, lambda_loom=LAMBDA_LOOM, lambda_light=LAMBDA_LIGHT)`: a gymnasium env with a `Dict` observation; `info["metabolic_cost"]`.
  - `KeyNormalizer(observation_space, key)`, `CriticExtractor(observation_space)`, `AsymmetricSACPolicy(..., actor_key=...)`.
  - `build_sac(learner, env, buffer_size=100_000, seed=42, device="auto", learning_starts=5_000) -> SAC`; `set_dn_stats(model, mean, scale)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sac.py`:

```python
import json

import mujoco
import numpy as np
import pytest
import torch
from fly_drone.brain import ENCODER_VERSION, BrainRuntime
from fly_drone.env import ConnectomeEnv
from fly_drone.sac import (
    GEOMETRY,
    AsymmetricSACPolicy,
    SacRoamEnv,
    visible_geometry,
)
from gymnasium import spaces


def zero_actor(path, brain):
    n = len(brain.feature_ids)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "encoder_version": ENCODER_VERSION,
                "dataset_hash": brain.dataset_hash,
                "feature_ids": brain.feature_ids,
                "mean": [0.0] * n,
                "scale": [1.0] * n,
                "layers": [{"weights": [[0.0] * n] * 4, "bias": [0.0] * 4}],
                "action_limits": [0.7, 0.5, 0.3, 0.8],
            }
        )
    )
    return path


def test_visible_geometry_reports_only_what_the_eyes_can_see():
    env = ConnectomeEnv(task="free_roam", level=0, respawn=False)
    try:
        env.reset(seed=31)
        env.plant.teleport([0.0, 0.0, 1.0], 0.0)
        env.plant.set_objects(target=[3.0, 0.5, 1.0])
        g = visible_geometry(env)
        assert g[4] == 1.0 and g[5] == pytest.approx(3.0, abs=0.05)
        assert g[6] == pytest.approx(0.5, abs=0.05)
        assert not g[:4].any()  # no threat flying
        env.plant.set_objects(target=[-3.0, 0.0, 1.0])
        assert not visible_geometry(env)[4:].any()
        assert env.launch_threat()
        assert visible_geometry(env)[0] == 1.0
        env.plant.set_ghost(True)
        assert not visible_geometry(env)[:4].any()
    finally:
        env.close()


def test_encoder_env_spaces_and_metabolic_cost(tmp_path):
    decoder = zero_actor(tmp_path / "decoder.json", BrainRuntime())
    env = SacRoamEnv("encoder", decoder=decoder, level=3)
    try:
        obs, _ = env.reset(seed=32)
        plant = env.env.plant
        bands = [
            g
            for g in range(plant.model.ngeom)
            if (mujoco.mj_id2name(plant.model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith("_band")
        ]
        greys = plant.model.geom_rgba[bands, :3]
        assert len(bands) == 4 and np.all(greys == greys[0, 0]) and 0.0 <= greys[0, 0] <= 0.15
        assert np.all(plant.model.geom_rgba[bands, 3] == 1.0)
        assert obs["eyes"].shape == (6, 48, 64) and obs["eyes"].dtype == np.uint8
        assert obs["dn"].shape == (2022,) and obs["geometry"].shape == (GEOMETRY,)
        assert env.action_space.shape == (8,)
        _, _, _, _, info = env.step(-np.ones(8))
        assert info["metabolic_cost"] == pytest.approx(0.0)
        _, _, _, _, info = env.step(np.ones(8))
        assert info["metabolic_cost"] == pytest.approx(0.01 * 2 + 0.002 * 2)
        np.testing.assert_array_equal(env.env.brain.cues, np.full(8, 2.0))
    finally:
        env.env.close()


def test_learner_arguments_are_validated(tmp_path):
    with pytest.raises(ValueError):
        SacRoamEnv("planner")
    with pytest.raises(ValueError):
        SacRoamEnv("encoder")
    with pytest.raises(ValueError):
        SacRoamEnv("decoder")


@pytest.mark.parametrize("actor_key", ["eyes", "dn"])
def test_actor_reads_only_its_key_and_critic_reads_all(actor_key):
    space = spaces.Dict(
        {
            "eyes": spaces.Box(0, 255, (6, 48, 64), np.uint8),
            "dn": spaces.Box(0, 1, (5,), np.float32),
            "geometry": spaces.Box(-np.inf, np.inf, (GEOMETRY,), np.float32),
        }
    )
    policy = AsymmetricSACPolicy(
        space,
        spaces.Box(-1, 1, (4,), np.float32),
        lambda _: 3e-4,
        actor_key=actor_key,
        net_arch={"pi": [8], "qf": [8]},
    )
    torch.manual_seed(0)
    a = {"eyes": torch.rand(2, 6, 48, 64), "dn": torch.rand(2, 5), "geometry": torch.rand(2, 8)}
    b = dict(a)
    for key in a:
        if key != actor_key:
            b[key] = torch.rand_like(a[key])
    action = torch.zeros(2, 4)
    with torch.no_grad():
        torch.testing.assert_close(policy.actor(a, deterministic=True), policy.actor(b, deterministic=True))
        assert not torch.allclose(policy.critic(a, action)[0], policy.critic(b, action)[0])
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_sac.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'fly_drone.sac'`.

- [ ] **Step 3: Implement `python/fly_drone/sac.py` (part 1)**

```python
"""SAC for encoder v5 (spec §3.3–§4): one learner at a time, the rest frozen in the loop.

Actors see only what they are deployed with: the encoder its eye stack, the decoder the
DN traces. The critic exists only during training and additionally sees DN traces and the
geometry of objects the eyes can currently see (honest-labels rule).
"""

import gymnasium as gym
import mujoco
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import MultiInputPolicy
from torch import nn

from .brain import V5_CHANNELS, BrainRuntime
from .encoder import FEATURES, EyesExtractor, eyes_space
from .env import ConnectomeEnv

GEOMETRY = 8
LAMBDA_LOOM = 0.01
LAMBDA_LIGHT = 0.002
LEARNERS = ("encoder", "decoder", "bypass")
ACTOR_KEYS = {"encoder": "eyes", "decoder": "dn", "bypass": "currents"}
LIGHT = [V5_CHANNELS.index(c) for c in ("mi1_l", "mi1_r", "tm3_l", "tm3_r")]
LOOM = [V5_CHANNELS.index(c) for c in ("lc4_l", "lc4_r", "lplc2_l", "lplc2_r")]
# Flags stay 0/1; body-frame metres are scaled into roughly [-2, 2] for the critic.
GEOMETRY_SCALE = (1.0, 0.25, 0.25, 0.25, 1.0, 0.25, 0.25, 0.25)
# Training only (spec §7): vary the wall bands' grey so the encoder cannot key on one
# texture. Evaluation (ConnectomeEnv) keeps the arena's 0.03 band.
BAND_GREY = (0.0, 0.15)


def visible_geometry(env):
    """Threat and beacon in the body frame, only while visible; zeros otherwise."""
    from .teacher import visible

    plant = env.plant
    pos = plant.pos[0]
    yaw = float(plant.rpy[0, 2])
    c, s = np.cos(yaw), np.sin(yaw)

    def body(point):
        d = np.asarray(point, dtype=float) - pos
        return [c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2]]

    out = np.zeros(GEOMETRY, np.float32)
    threat = env.roam["threat"] if env.roam is not None else None
    if threat is not None and visible(env, plant.obstacle, "obstacle"):
        out[0] = 1.0
        out[1:4] = body(plant.obstacle)
    if env.beacon_visible():
        out[4] = 1.0
        out[5:8] = body(plant.target)
    return out


def learner_spaces(learner, n_features=2022):
    """Observation (actor key + critic-only keys) and action space for one learner."""
    keys = {"geometry": spaces.Box(-np.inf, np.inf, (GEOMETRY,), np.float32)}
    action = spaces.Box(-1, 1, (4,), np.float32)
    if learner == "encoder":
        keys["eyes"] = eyes_space()
        keys["dn"] = spaces.Box(0, 1, (n_features,), np.float32)
        action = spaces.Box(-1, 1, (len(V5_CHANNELS),), np.float32)
    elif learner == "decoder":
        keys["dn"] = spaces.Box(0, 1, (n_features,), np.float32)
    else:
        keys["currents"] = spaces.Box(0, 2, (len(V5_CHANNELS),), np.float32)
    return spaces.Dict(keys), action


class SacRoamEnv(gym.Env):
    """Free roam (L3, respawn) seen by one learner: encoder, decoder or brain bypass."""

    metadata = {}

    def __init__(
        self,
        learner,
        decoder=None,
        encoder=None,
        level=3,
        lambda_loom=LAMBDA_LOOM,
        lambda_light=LAMBDA_LIGHT,
    ):
        if learner not in LEARNERS:
            raise ValueError(f"learner must be one of {LEARNERS}")
        if learner == "encoder":
            if decoder is None:
                raise ValueError("encoder learning needs a frozen decoder")
            brain = BrainRuntime(encoder="external")
            # The frozen decoder is scenery here; validate() pairs it with the learned
            # encoder under the strict version check.
            brain.load_policy(decoder, check_encoder=False)
        else:
            if encoder is None:
                raise ValueError(f"{learner} learning needs a frozen encoder")
            brain = BrainRuntime(encoder=encoder)
        self.learner = learner
        self.lambda_loom = lambda_loom
        self.lambda_light = lambda_light
        self.env = ConnectomeEnv(task="free_roam", level=level, respawn=True, brain=brain)
        n = len(brain.feature_ids)
        self.observation_space, self.action_space = learner_spaces(learner, n)
        self.features = np.zeros(n, np.float32)

    def _obs(self):
        brain, keys = self.env.brain, self.observation_space.spaces
        obs = {"geometry": visible_geometry(self.env)}
        if "eyes" in keys:
            obs["eyes"] = brain.stack.array()
        if "dn" in keys:
            obs["dn"] = self.features
        if "currents" in keys:
            obs["currents"] = brain.encoder.currents(brain.stack.array())
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        seed = int(seed if seed is not None else self.np_random.integers(0, 2**31))
        self.features, info = self.env.reset(seed=seed)
        self._randomise_bands()
        return self._obs(), info

    def _randomise_bands(self):
        plant = self.env.plant
        grey = float(self.np_random.uniform(*BAND_GREY))
        for g in range(plant.model.ngeom):
            name = mujoco.mj_id2name(plant.model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if name.endswith("_band"):
                plant.model.geom_rgba[g, :3] = grey  # alpha stays: ghost mode owns it
        # The first stacked frame was rendered before the recolour; start the stack again.
        self.env.brain.stack.clear()
        self.env.brain.push_frame(plant.camera())

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        brain = self.env.brain
        cost = 0.0
        if self.learner == "encoder":
            currents = brain.set_currents(action + 1.0)
            velocity = brain.infer(self.features) / self.env.plant.limits
            cost = self.lambda_loom * float(currents[LOOM].mean()) + self.lambda_light * float(
                currents[LIGHT].mean()
            )
        else:
            velocity = action
        self.features, reward, terminated, truncated, info = self.env.step(
            np.clip(velocity, -1, 1)
        )
        info["metabolic_cost"] = cost
        return self._obs(), float(reward - cost), terminated, truncated, info

    def close(self):
        self.env.close()


class KeyNormalizer(BaseFeaturesExtractor):
    """One observation key, standardised with mean/scale buffers (like NeuralNormalizer)."""

    def __init__(self, observation_space, key):
        n = observation_space[key].shape[0]
        super().__init__(observation_space, n)
        self.key = key
        self.register_buffer("mean", torch.zeros(n))
        self.register_buffer("scale", torch.ones(n))

    def forward(self, observations):
        return (observations[self.key] - self.mean) / self.scale


class CriticExtractor(BaseFeaturesExtractor):
    """Training-only view: every key present (eyes, DN traces, currents, visible geometry)."""

    def __init__(self, observation_space):
        keys = observation_space.spaces
        n_dn = keys["dn"].shape[0] if "dn" in keys else 0
        dim = GEOMETRY + (2 * FEATURES if "eyes" in keys else 0) + (64 if n_dn else 0)
        dim += len(V5_CHANNELS) if "currents" in keys else 0
        super().__init__(observation_space, dim)
        self.register_buffer("geometry_scale", torch.tensor(GEOMETRY_SCALE))
        self.eyes = EyesExtractor(observation_space) if "eyes" in keys else None
        self.dn_norm = KeyNormalizer(observation_space, "dn") if n_dn else None
        self.dn = nn.Sequential(nn.Linear(n_dn, 64), nn.ReLU()) if n_dn else None
        self.has_currents = "currents" in keys

    def forward(self, observations):
        parts = [observations["geometry"] * self.geometry_scale]
        if self.eyes is not None:
            parts.append(self.eyes(observations))
        if self.dn is not None:
            parts.append(self.dn(self.dn_norm(observations)))
        if self.has_currents:
            parts.append(observations["currents"])
        return torch.cat(parts, 1)


class AsymmetricSACPolicy(MultiInputPolicy):
    """SAC policy whose actor reads one deployable key while the critic reads every key."""

    def __init__(self, *args, actor_key="eyes", **kwargs):
        self.actor_key = actor_key
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor=None):
        if self.actor_key == "eyes":
            extractor = EyesExtractor(self.observation_space)
        else:
            extractor = KeyNormalizer(self.observation_space, self.actor_key)
        return super().make_actor(extractor)

    def make_critic(self, features_extractor=None):
        return super().make_critic(CriticExtractor(self.observation_space))

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data["actor_key"] = self.actor_key
        return data


def build_sac(learner, env, buffer_size=100_000, seed=42, device="auto", learning_starts=5_000):
    from stable_baselines3 import SAC

    return SAC(
        AsymmetricSACPolicy,
        env,
        buffer_size=buffer_size,
        batch_size=256,
        learning_starts=learning_starts,
        train_freq=1,
        gradient_steps=2,
        gamma=0.99,
        learning_rate=3e-4,
        seed=seed,
        device=device,
        verbose=1,
        policy_kwargs={
            "actor_key": ACTOR_KEYS[learner],
            # The encoder's head is linear on the eye features (LearnedEncoder format);
            # the decoder's hidden layers are tanh (Rust actor format).
            "net_arch": {"pi": [] if learner == "encoder" else [64, 64], "qf": [256, 256]},
            "activation_fn": nn.Tanh,
        },
    )


def set_dn_stats(model, mean, scale):
    """Standardise DN traces the same way in the actor, the critic and the target critic."""
    mean = torch.as_tensor(np.asarray(mean), dtype=torch.float32)
    scale = torch.as_tensor(np.asarray(scale), dtype=torch.float32)
    for net in (model.actor, model.critic, model.critic_target):
        for module in net.modules():
            if isinstance(module, KeyNormalizer) and module.key == "dn":
                module.mean.copy_(mean)
                module.scale.copy_(scale)
```

- [ ] **Step 4: Run the tests**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_sac.py`
Expected: 5 passed.

- [ ] **Step 5: Commit and push**

```bash
.venv/bin/ruff format python tests && .venv/bin/ruff check python tests
git add python/fly_drone/sac.py tests/test_sac.py
git commit -m "Add the SAC free-roam environment and an actor-critic split by deployable input" && git push
```

---

### Task 7: SAC decoder export (tanh) and warm start from DAgger data

**Files:**

- Modify: `python/fly_drone/distill.py` (extract `class_weights`)
- Modify: `python/fly_drone/sac.py`
- Modify: `tests/test_sac.py`

**Interfaces:**

- Consumes: `learner_spaces`, `build_sac`, `set_dn_stats`, `KeyNormalizer` (Task 6); `distill._load`; Rust `"output": "tanh"` (Task 2); `BrainRuntime.load_policy` strict check (Task 3).
- Produces:
  - `distill.class_weights(drive, mask) -> np.ndarray` (per-drive weights, mean 1 over present drives)
  - `sac.SpacesOnlyEnv(learner, n_features=2022)`
  - `sac.export_decoder(model, brain, path, limits) -> float` (max parity error, raises above 1e-4)
  - `sac.warm_start_decoder(model, paths, dataset_hash, steps=4000, holdout=0.1, seed=72) -> dict`
  - `sac.init_decoder(paths, encoder, output, steps=4000, device="auto") -> dict`, which writes `output/decoder.zip`, `output/decoder.json` and `output/warm-start.json`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sac.py`:

```python
import hashlib

from fly_drone.arena import ArenaSpec
from fly_drone.brain import DATA
from fly_drone.encoder import LearnedEncoder
from fly_drone.sac import SpacesOnlyEnv, build_sac, export_decoder, set_dn_stats, warm_start_decoder
from fly_drone.teacher import DRIVES


def test_export_decoder_matches_rust_with_tanh_output_and_pins_the_encoder(tmp_path):
    LearnedEncoder.fresh(seed=5).save(tmp_path / "e.pt")
    brain = BrainRuntime(encoder=tmp_path / "e.pt")
    model = build_sac("decoder", SpacesOnlyEnv("decoder"), buffer_size=1, device="cpu")
    rng = np.random.default_rng(0)
    set_dn_stats(model, rng.uniform(0, 0.1, 2022), rng.uniform(0.5, 1.5, 2022))
    error = export_decoder(model, brain, tmp_path / "d.json", ArenaSpec().limits)
    assert error <= 1e-4
    data = json.loads((tmp_path / "d.json").read_text())
    assert data["output"] == "tanh" and data["encoder_version"] == brain.encoder_version
    assert [len(layer["bias"]) for layer in data["layers"]] == [64, 64, 4]
    with pytest.raises(ValueError, match="encoder"):
        BrainRuntime().load_policy(tmp_path / "d.json")


def test_warm_start_decoder_learns_labels_and_shares_normalisation(tmp_path):
    rng = np.random.default_rng(1)
    n = 400
    x = rng.uniform(0, 1, (n, 2022)).astype(np.float32)
    path = tmp_path / "dagger.npz"
    np.savez(
        path,
        x=x.astype(np.float16),
        y=(x[:, :4] * 1.6 - 0.8).astype(np.float32),
        drive=rng.integers(0, len(DRIVES), n).astype(np.int8),
        flight=np.repeat(np.arange(20), n // 20).astype(np.int32),
        dataset_hash=hashlib.sha256((DATA / "graph.bin").read_bytes()).hexdigest(),
        encoder_version=ENCODER_VERSION,
        student="None",
        beta=1.0,
    )
    model = build_sac("decoder", SpacesOnlyEnv("decoder"), buffer_size=1, device="cpu")
    digest = hashlib.sha256((DATA / "graph.bin").read_bytes()).hexdigest()
    report = warm_start_decoder(model, [path], digest, steps=300)
    assert report["held_out_flights"] == 2 and set(report["drives"]) == set(DRIVES)
    assert report["loss_last"] < report["loss_first"]
    actor_norm = model.actor.features_extractor
    assert float(actor_norm.scale.min()) >= 0.003
    assert torch.equal(model.critic.features_extractor.dn_norm.mean, actor_norm.mean)
    assert torch.equal(model.critic_target.features_extractor.dn_norm.scale, actor_norm.scale)
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_sac.py -k "export_decoder or warm_start"`
Expected: FAIL with `ImportError: cannot import name 'SpacesOnlyEnv'`.

- [ ] **Step 3: Extract `class_weights` in `distill.py`**

Add above `fit`:

```python
def class_weights(drive, mask):
    """Threat frames are rare: weight every drive present in `mask` equally (mean 1)."""
    counts = np.bincount(drive[mask], minlength=len(DRIVES)).astype(float)
    weight = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
    return weight / weight[counts > 0].mean()
```

In `fit`, replace the four lines from `counts = np.bincount(...)` through `class_weight /= ...` with
`class_weight = class_weights(drive, ~test)`. Run `pytest -q tests/test_distill.py`. It must still pass.

- [ ] **Step 4: Implement in `sac.py`**

```python
class SpacesOnlyEnv(gym.Env):
    """Just the spaces SAC needs to build a model; no brain or renderer."""

    metadata = {}

    def __init__(self, learner, n_features=2022):
        self.observation_space, self.action_space = learner_spaces(learner, n_features)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = {k: np.zeros(s.shape, s.dtype) for k, s in self.observation_space.spaces.items()}
        return obs, {}

    def step(self, action):
        return self.reset()[0], 0.0, True, False, {}


def export_decoder(model, brain, path, limits):
    """SAC decoder actor -> Rust actor JSON (tanh hidden, tanh output) with a parity check."""
    import json
    from pathlib import Path

    actor = model.actor
    layers = []
    for layer in list(actor.latent_pi) + [actor.mu]:
        if isinstance(layer, nn.Linear):
            layers.append(
                {
                    "weights": layer.weight.detach().cpu().tolist(),
                    "bias": layer.bias.detach().cpu().tolist(),
                }
            )
        elif not isinstance(layer, nn.Tanh):
            raise ValueError("only tanh MLP export supported")
    norm = actor.features_extractor
    limits = np.asarray(limits, dtype=float)
    payload = {
        "version": 1,
        "encoder_version": brain.encoder_version,
        "dataset_hash": brain.dataset_hash,
        "feature_ids": brain.feature_ids,
        "mean": norm.mean.cpu().tolist(),
        "scale": norm.scale.cpu().tolist(),
        "layers": layers,
        "action_limits": limits.tolist(),
        "output": "tanh",
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    brain.load_policy(path)
    rng = np.random.default_rng(123)
    error = 0.0
    for x in rng.uniform(0, 1, (32, len(brain.feature_ids))).astype(np.float32):
        obs = {"dn": x, "geometry": np.zeros(GEOMETRY, np.float32)}
        expected = model.predict(obs, deterministic=True)[0] * limits
        error = max(error, float(np.max(np.abs(expected - brain.infer(x)))))
    if error > 1e-4:
        raise RuntimeError(f"Rust policy parity failed: {error}")
    return error


def warm_start_decoder(model, paths, dataset_hash, steps=4000, holdout=0.1, seed=72):
    """Behaviour-clone stage-1 DAgger labels onto the SAC decoder's tanh head."""
    from .distill import _load, class_weights
    from .teacher import DRIVES

    x, y, drive, flight = _load(paths, dataset_hash)
    rng = np.random.default_rng(seed)
    unique = np.unique(flight)
    held = rng.choice(unique, max(1, int(len(unique) * holdout)), replace=False)
    test = np.isin(flight, held)
    train = np.flatnonzero(~test)
    set_dn_stats(model, x[train].mean(0), np.maximum(x[train].std(0), 0.003))
    actor, device = model.actor, model.device
    weight = class_weights(drive, ~test)[drive].astype(np.float32)
    # tanh never reaches +-1: stop labels just short of saturation.
    target = np.clip(y, -0.97, 0.97).astype(np.float32)
    params = list(actor.latent_pi.parameters()) + list(actor.mu.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    torch.manual_seed(seed)

    def predict(ids):
        obs = {"dn": torch.as_tensor(x[ids], device=device)}
        return torch.tanh(actor.mu(actor.latent_pi(actor.features_extractor(obs))))

    losses = []
    for _ in range(steps):
        ids = rng.choice(train, 256)
        w = torch.as_tensor(weight[ids, None], device=device)
        err = (predict(ids) - torch.as_tensor(target[ids], device=device)).square()
        loss = (w * err).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    report = {
        "samples": int(len(x)),
        "held_out_flights": int(len(held)),
        "loss_first": losses[0],
        "loss_last": float(np.mean(losses[-20:])),
        "drives": {name: {} for name in DRIVES},
    }
    with torch.no_grad():
        # Start SAC exploration narrow around the cloned behaviour.
        actor.log_std.weight.zero_()
        actor.log_std.bias.fill_(-2.5)
        for split, mask in (("train", ~test), ("held_out", test)):
            ids = np.flatnonzero(mask)
            errs = [
                (predict(ids[i : i + 4096]) - torch.as_tensor(target[ids[i : i + 4096]], device=device))
                .square()
                .mean(1)
                .cpu()
                .numpy()
                for i in range(0, len(ids), 4096)
            ]
            err = np.concatenate(errs) if errs else np.zeros(0)
            for k, name in enumerate(DRIVES):
                sel = drive[mask] == k
                report["drives"][name][f"{split}_mse"] = float(err[sel].mean()) if sel.any() else None
    return report


def init_decoder(paths, encoder, output, steps=4000, device="auto"):
    """Round-0 decoder: DAgger labels on a tanh head, exported for the given encoder."""
    import json
    from pathlib import Path

    from .arena import ArenaSpec

    brain = BrainRuntime(encoder=encoder)
    # buffer_size=1: this model is only cloned and saved; rounds reload it with a real buffer.
    model = build_sac(
        "decoder", SpacesOnlyEnv("decoder", len(brain.feature_ids)), buffer_size=1, device=device
    )
    report = warm_start_decoder(model, paths, brain.dataset_hash, steps)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "decoder")
    report["export_max_error"] = export_decoder(model, brain, out / "decoder.json", ArenaSpec().limits)
    report["encoder_version"] = brain.encoder_version
    report["paths"] = [str(p) for p in paths]
    (out / "warm-start.json").write_text(json.dumps(report, indent=2))
    return report
```

- [ ] **Step 5: Run the tests**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_sac.py tests/test_distill.py`
Expected: all pass.

- [ ] **Step 6: Commit and push**

```bash
.venv/bin/ruff format python tests && .venv/bin/ruff check python tests
git add python/fly_drone/distill.py python/fly_drone/sac.py tests/test_sac.py
git commit -m "Export SAC decoders to Rust with tanh parity and warm-start them from DAgger labels" && git push
```

---

### Task 8: Evaluation with a learned encoder; E1–E3 checks

**Files:**

- Modify: `python/fly_drone/distill.py` (`_screen_job`, `screen`)
- Modify: `python/fly_drone/roam_eval.py` (`_probe_job`, `skill_probes`, `evaluate_free_roam`, new E-checks)
- Modify: `tests/test_roam_eval.py`

**Interfaces:**

- Consumes: `BrainRuntime(encoder=...)`, `V4_TO_V5` (Task 3); `LearnedEncoder` (Task 4); `sac.LIGHT`, `sac.LOOM` (Task 6); `teacher.visible`.
- Produces:
  - Screen job tuple `(controller, seeds, seconds, level, ablation[, encoder])` and probe job tuple `(controller, probe, seeds, vision, seconds[, encoder])`. The encoder is used only for `policy:` and `bypass:` controllers. Baselines always fly v4.
  - Controller `bypass:<sac.zip>`: a SAC model whose actor reads `{"currents", "geometry"}` (geometry zeros at evaluation).
  - `distill.screen(..., encoder=None)`, `roam_eval.skill_probes(..., encoder=None)`, `roam_eval.evaluate_free_roam(..., encoder=None)`
  - `roam_eval.THREAT_POSITIVE_RANGE = 3.0`, `THREAT_NEGATIVE_RANGE = 6.0`, `ENCODER_CHECKS = {"E1_min_loom_auc": 0.8, "E2_min_light_margin": 0.05, "E2_min_loom_margin": 0.1}`
  - `roam_eval.roc_auc(scores, labels) -> float | None`
  - `roam_eval.encoder_scores(currents, threat, beacon) -> {"E1": ..., "E2": ..., "thresholds": ...}` (threat labels: 1 positive, 0 negative, −1 excluded)
  - `roam_eval._checks_job((policy, encoder, seeds, seconds, level)) -> (currents, threat, beacon)`
  - `roam_eval.encoder_checks(policy, output, encoder=None, episodes=50, seconds=120, workers=6, seed_base=1000, level=3) -> dict`
  - `roam_eval.bypass_comparison(full_summary, bypass_summary) -> dict` (E3, reported without a bar)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_roam_eval.py`:

```python
import json

import pytest
from fly_drone import distill
from fly_drone.brain import BrainRuntime
from fly_drone.encoder import LearnedEncoder
from fly_drone.roam_eval import (
    _checks_job,
    bypass_comparison,
    encoder_scores,
    roc_auc,
)
from test_sac import zero_actor


def test_roc_auc_counts_ties_as_half_and_needs_both_classes():
    assert roc_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)
    assert roc_auc([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert roc_auc([0.5] * 4, [0, 1, 0, 1]) == 0.5
    assert roc_auc([0.1, 0.2], [1, 1]) is None


def _synthetic(loom_selective):
    rng = np.random.default_rng(0)
    n = 2000
    threat = rng.choice([-1, 0, 1], n, p=[0.1, 0.8, 0.1])
    beacon = rng.random(n) < 0.4
    currents = rng.uniform(0, 0.2, (n, 8)).astype(np.float32)
    currents[:, :4] += np.where(beacon, 1.0, 0.0)[:, None]
    if loom_selective:
        currents[:, 4:] += np.where(threat == 1, 1.5, 0.0)[:, None]
    else:
        currents[:, 4:] += rng.uniform(0, 1.5, (n, 1))
    return currents, threat, beacon


def test_a_selective_encoder_passes_e1_and_e2():
    report = encoder_scores(*_synthetic(True))
    assert report["E1"]["passed"] and report["E2"]["passed"]
    assert report["E1"]["positives"] > 0 and report["E1"]["negatives"] > 0


def test_loom_that_ignores_threats_fails_e1_and_e2():
    report = encoder_scores(*_synthetic(False))
    assert not report["E1"]["passed"] and not report["E2"]["passed"]


def test_checks_job_labels_frames_and_maps_v4_cues_to_eight_channels(tmp_path):
    actor = zero_actor(tmp_path / "a.json", BrainRuntime())
    currents, threat, beacon = _checks_job((str(actor), None, [3], 0.4, 3))
    assert len(currents) == len(threat) == len(beacon) == 10
    assert currents[0].shape == (8,) and set(threat) <= {-1, 0, 1}


def test_screen_job_flies_a_v5_policy_with_its_encoder_and_baselines_on_v4(tmp_path):
    enc = LearnedEncoder.fresh(seed=6)
    enc.save(tmp_path / "e.pt")
    path = zero_actor(tmp_path / "a.json", BrainRuntime())
    actor = json.loads(path.read_text())
    actor["encoder_version"] = enc.version
    path.write_text(json.dumps(actor))
    job = (f"policy:{path}", [3], 0.4, 3, "loom", str(tmp_path / "e.pt"))
    _, _, runs = distill._screen_job(job)
    assert len(runs) == 1
    _, _, runs = distill._screen_job(("cue_script", [3], 0.4, 3, "none", None))
    assert len(runs) == 1


def test_bypass_comparison_flags_only_a_bypass_better_on_all_three():
    full = {"beacons_per_min": 1.0, "collisions_per_min": 0.4, "near_dodge_rate": 0.8}
    worse = {"beacons_per_min": 1.2, "collisions_per_min": 0.6, "near_dodge_rate": 0.9}
    better = {"beacons_per_min": 1.2, "collisions_per_min": 0.3, "near_dodge_rate": 0.9}
    assert not bypass_comparison(full, worse)["bypass_better"]
    assert bypass_comparison(full, better)["bypass_better"]
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_roam_eval.py`
Expected: FAIL with `ImportError: cannot import name '_checks_job'`.

- [ ] **Step 3: Thread the encoder through screens (`distill.py`)**

In `_screen_job`, replace the first lines through the policy load with:

```python
    controller, seeds, seconds, level, ablation, *rest = job
    encoder = rest[0] if rest else None
    brain = BrainRuntime(encoder=encoder)
    bypass = None
    if controller.startswith("policy:"):
        brain.load_policy(controller.split(":", 1)[1])
    elif controller.startswith("bypass:"):
        from stable_baselines3 import SAC

        bypass = SAC.load(controller.split(":", 1)[1], device="cpu")
```

In the frame loop, add a branch before the final `else`:

```python
                elif bypass is not None:
                    # E3 control: a decoder that reads the encoder's currents, not the brain.
                    bypass_obs = {
                        "currents": brain.encoder.currents(brain.stack.array()),
                        "geometry": np.zeros(8, np.float32),
                    }
                    action = bypass.predict(bypass_obs, deterministic=True)[0]
```

Give `screen` a parameter `encoder=None`. Build its jobs as:

```python
    learned = ("policy:", "bypass:")
    jobs = [
        (c, chunk.tolist(), seconds, level, a, encoder if c.startswith(learned) else None)
        for c, a in combos
        for chunk in np.array_split(all_seeds, min(per, seeds))
        if len(chunk)
    ]
```

and add `"encoder": str(encoder) if encoder else None,` to the report dict.

- [ ] **Step 4: Thread the encoder through probes and evaluation, and add E-checks (`roam_eval.py`)**

In `_probe_job`:

```python
    controller, probe, seeds, vision, seconds, *rest = job
    encoder = rest[0] if rest and controller.startswith("policy:") else None
    brain = BrainRuntime(encoder=encoder)
```

Give `skill_probes` a parameter `encoder=None`, and make its job tuple `(controller, probe, chunk.tolist(), vision, seconds, encoder)`.
Give `evaluate_free_roam` a parameter `encoder=None`. Pass `encoder=encoder` to `screen(...)` and to
`skill_probes(key, episodes, workers, seed_base=seed_base, encoder=encoder)`, and set
`report["encoder"] = str(encoder) if encoder else None`.

Append the E-checks:

```python
# Encoder v5 checks, pre-registered 2026-09-14 before any v5 result (spec §5). Additive:
# they never change ACCEPTANCE.
THREAT_POSITIVE_RANGE = 3.0
THREAT_NEGATIVE_RANGE = 6.0
ENCODER_CHECKS = {
    "E1_min_loom_auc": 0.8,
    "E2_min_light_margin": 0.05,
    "E2_min_loom_margin": 0.1,
}


def roc_auc(scores, labels):
    """Mann-Whitney AUC with tied scores ranked at their average; None without both classes."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    pos, neg = int(labels.sum()), int((~labels).sum())
    if not pos or not neg:
        return None
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2.0)[inverse]
    return float((ranks[labels].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def encoder_scores(currents, threat, beacon):
    from .sac import LIGHT, LOOM

    currents = np.asarray(currents, dtype=float)
    threat = np.asarray(threat)
    keep = threat >= 0
    light = currents[keep][:, LIGHT].mean(1)
    loom = currents[keep][:, LOOM].max(1)
    is_threat = threat[keep] == 1
    seen = np.asarray(beacon, dtype=bool)[keep]
    auc = {
        "loom_threat": roc_auc(loom, is_threat),
        "loom_beacon": roc_auc(loom, seen),
        "light_beacon": roc_auc(light, seen),
        "light_threat": roc_auc(light, is_threat),
    }
    t = ENCODER_CHECKS
    complete = None not in auc.values()
    light_margin = auc["light_beacon"] - auc["light_threat"] if complete else None
    loom_margin = auc["loom_threat"] - auc["loom_beacon"] if complete else None
    return {
        "E1": {
            "loom_auc": auc["loom_threat"],
            "positives": int(is_threat.sum()),
            "negatives": int((~is_threat).sum()),
            "passed": bool(
                auc["loom_threat"] is not None and auc["loom_threat"] >= t["E1_min_loom_auc"]
            ),
        },
        "E2": {
            "auc": auc,
            "light_margin": light_margin,
            "loom_margin": loom_margin,
            "passed": bool(
                complete
                and light_margin >= t["E2_min_light_margin"]
                and loom_margin >= t["E2_min_loom_margin"]
            ),
        },
        "thresholds": ENCODER_CHECKS,
    }


def _checks_job(job):
    """Fly a policy intact; per frame, the currents applied and labels of the state they saw."""
    from .brain import V4_TO_V5, BrainRuntime
    from .distill import _roam_env
    from .teacher import visible

    policy, encoder, seeds, seconds, level = job
    brain = BrainRuntime(encoder=encoder)
    brain.load_policy(policy)
    env = _roam_env(level, brain)
    currents, threat, beacon = [], [], []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=int(seed))
            previous_gap = None
            for _ in range(int(seconds / 0.04)):
                flying = env.roam["threat"] is not None
                gap = float(np.linalg.norm(env.plant.obstacle - env.plant.pos[0]))
                if not flying or gap > THREAT_NEGATIVE_RANGE:
                    label = 0
                elif (
                    gap < THREAT_POSITIVE_RANGE
                    and previous_gap is not None
                    and gap < previous_gap
                    and visible(env, env.plant.obstacle, "obstacle")
                ):
                    label = 1
                else:
                    label = -1
                previous_gap = gap if flying else None
                seen = env.beacon_visible()
                obs, *_ = env.step(brain.infer(obs) / env.plant.limits)
                applied = brain.cues if brain.learned else brain.cues[list(V4_TO_V5)]
                currents.append(np.asarray(applied, dtype=np.float32).copy())
                threat.append(label)
                beacon.append(bool(seen))
    finally:
        env.close()
    return currents, threat, beacon


def encoder_checks(
    policy, output, encoder=None, episodes=50, seconds=120, workers=6, seed_base=1000, level=3
):
    """E1-E2 on held-out seeds, intact brain. encoder=None measures the v4 baseline."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    seeds = np.arange(seed_base, seed_base + episodes)
    policy = str(Path(policy).resolve())
    encoder = str(Path(encoder).resolve()) if encoder else None
    jobs = [
        (policy, encoder, c.tolist(), seconds, level)
        for c in np.array_split(seeds, min(workers, episodes))
        if len(c)
    ]
    currents, threat, beacon = [], [], []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for c, t, b in pool.map(_checks_job, jobs):
            currents += c
            threat += t
            beacon += b
    report = {
        "policy": policy,
        "encoder": encoder,
        "seeds": [int(seeds[0]), int(seeds[-1])],
        "seconds": seconds,
        "frames": len(currents),
        **encoder_scores(currents, threat, beacon),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2))
    return report


def bypass_comparison(full, bypass):
    """E3, reported without a bar: can a decoder reading the 8 currents do better than the brain?"""
    keys = ("beacons_per_min", "collisions_per_min", "near_dodge_rate")
    pick = {name: {k: s.get(k) for k in keys} for name, s in (("full", full), ("bypass", bypass))}
    f, b = pick["full"], pick["bypass"]
    better = (
        None not in (*f.values(), *b.values())
        and b["beacons_per_min"] >= f["beacons_per_min"]
        and b["collisions_per_min"] <= f["collisions_per_min"]
        and b["near_dodge_rate"] >= f["near_dodge_rate"]
    )
    return {
        **pick,
        "bypass_better": bool(better),
        "note": "If bypass_better, stop and report to the user before any stage 4 conclusions.",
    }
```

- [ ] **Step 5: Run the tests**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_roam_eval.py tests/test_distill.py tests/test_brain_v5.py`
Expected: all pass. Existing probe and screen tests still pass with 5-tuple jobs.

- [ ] **Step 6: Commit and push**

```bash
.venv/bin/ruff format python tests && .venv/bin/ruff check python tests
git add python/fly_drone/distill.py python/fly_drone/roam_eval.py tests/test_roam_eval.py
git commit -m "Evaluate free roam with a learned encoder and pre-register encoder checks E1-E3" && git push
```

---

### Task 9: SAC rounds, round validation and CLI

**Files:**

- Modify: `python/fly_drone/sac.py`
- Modify: `python/fly_drone/cli.py`
- Modify: `tests/test_sac.py`

**Interfaces:**

- Consumes: `SacRoamEnv`, `build_sac`, `set_dn_stats`, `export_decoder`, `init_decoder` (Tasks 6–7); `LearnedEncoder` (Task 4); `distill.screen(..., encoder=)`, `roam_eval.encoder_checks` (Task 8); `encoder.collect_clone`, `encoder.fit_clone` (Task 4).
- Produces:
  - `sac.MetabolicLogger` (SB3 callback, records `rollout/metabolic_cost`)
  - `sac.train_round(learner, output, frames, decoder=None, encoder=None, init=None, workers=6, seed=42, buffer_size=100_000, device="auto", level=3, learning_starts=5_000) -> dict`. It writes `output/<learner>.zip`, `output/round.json`, `output/checkpoints/`, and additionally `encoder.pt` (encoder learner) or `decoder.json` (decoder learner). `init` is a `.zip` from the previous round or, for the first encoder round only, the clone `encoder.pt`.
  - `sac.validate(decoder, encoder, output, seeds=10, seed_base=9000, seconds=60, workers=6, level=3) -> dict` with keys `near_dodge_rate`, `balanced_dodge_rate`, `ghost_near_dodge_rate`, `beacons_per_min`, `collisions_per_min`, `E1`, `E2`
  - CLI: `encoder-collect`, `encoder-clone`, `sac-init-decoder`, `sac-round`, `sac-validate`, `encoder-checks`, and `--encoder` on `evaluate` and `roam-screen`

- [ ] **Step 1: Write the failing smoke test**

Append to `tests/test_sac.py`:

```python
from fly_drone import distill
from fly_drone.sac import train_round, validate


def test_rounds_for_every_learner_resume_export_and_validate(tmp_path):
    decoder0 = zero_actor(tmp_path / "decoder0.json", BrainRuntime())
    LearnedEncoder.fresh(seed=7).save(tmp_path / "clone.pt")
    small = {"workers": 1, "buffer_size": 200, "device": "cpu", "learning_starts": 20}

    enc = train_round("encoder", tmp_path / "enc1", 40, decoder=decoder0, init=tmp_path / "clone.pt", **small)
    assert enc["encoder_version"] == LearnedEncoder.load(tmp_path / "enc1" / "encoder.pt").version
    assert enc["frames"] >= 40

    again = train_round("encoder", tmp_path / "enc2", 30, decoder=decoder0, init=tmp_path / "enc1" / "encoder.zip", **small)
    assert again["encoder_version"] != enc["encoder_version"]

    encoder_pt = tmp_path / "enc1" / "encoder.pt"
    dec = train_round("decoder", tmp_path / "dec1", 40, encoder=encoder_pt, **small)
    assert dec["export_max_error"] <= 1e-4
    BrainRuntime(encoder=encoder_pt).load_policy(tmp_path / "dec1" / "decoder.json")

    train_round("bypass", tmp_path / "byp", 40, encoder=encoder_pt, **small)
    job = (f"bypass:{tmp_path / 'byp' / 'bypass.zip'}", [3], 0.4, 3, "none", str(encoder_pt))
    assert len(distill._screen_job(job)[2]) == 1

    summary = validate(tmp_path / "dec1" / "decoder.json", encoder_pt, tmp_path / "val.json", seeds=1, seconds=0.4, workers=1)
    assert {"near_dodge_rate", "beacons_per_min", "E1", "E2"} <= set(summary)
    assert json.loads((tmp_path / "val.json").read_text())["encoder"] == str(encoder_pt.resolve())
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_sac.py -k rounds`
Expected: FAIL with `ImportError: cannot import name 'train_round'`.

- [ ] **Step 3: Implement in `sac.py`**

Add `import json` and `from pathlib import Path` at the top, and `from stable_baselines3.common.callbacks import BaseCallback`. Then:

```python
class MetabolicLogger(BaseCallback):
    def _on_step(self):
        costs = [info.get("metabolic_cost", 0.0) for info in self.locals["infos"]]
        self.logger.record_mean("rollout/metabolic_cost", float(np.mean(costs)))
        return True


def _factory(learner, rank, seed, decoder, encoder, level):
    def make():
        from stable_baselines3.common.monitor import Monitor

        torch.set_num_threads(1)
        env = Monitor(SacRoamEnv(learner, decoder=decoder, encoder=encoder, level=level))
        env.reset(seed=seed + rank)
        return env

    return make


def train_round(
    learner,
    output,
    frames,
    decoder=None,
    encoder=None,
    init=None,
    workers=6,
    seed=42,
    buffer_size=100_000,
    device="auto",
    level=3,
    learning_starts=5_000,
):
    """One SAC round for one learner; the other half is frozen inside the environment."""
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from .arena import ArenaSpec
    from .encoder import LearnedEncoder

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(int(workers), 6))
    decoder = str(Path(decoder).resolve()) if decoder else None
    encoder = str(Path(encoder).resolve()) if encoder else None
    factories = [_factory(learner, r, seed, decoder, encoder, level) for r in range(workers)]
    # Each env owns a full brain and renderer; spawn keeps EGL state per process.
    vec = SubprocVecEnv(factories, start_method="spawn") if workers > 1 else DummyVecEnv(factories)
    try:
        if init is not None and str(init).endswith(".zip"):
            # Fresh replay buffer each round: the frozen partner changed, old transitions are stale.
            model = SAC.load(
                init, env=vec, device=device, buffer_size=buffer_size, learning_starts=learning_starts
            )
        else:
            model = build_sac(learner, vec, buffer_size, seed, device, learning_starts)
            if init is not None:
                if learner != "encoder":
                    raise ValueError("a .pt init is the v4 clone for the first encoder round")
                state = torch.load(init, map_location="cpu", weights_only=True)
                model.actor.features_extractor.load_state_dict(state["extractor"])
                model.actor.mu.load_state_dict(state["mu"])
                with torch.no_grad():
                    model.actor.log_std.weight.zero_()
                    model.actor.log_std.bias.fill_(-2.5)
        if learner == "encoder":
            stats = json.loads(Path(decoder).read_text())
            set_dn_stats(model, stats["mean"], stats["scale"])
        callbacks = CallbackList(
            [
                CheckpointCallback(
                    save_freq=max(1, 50_000 // workers),
                    save_path=str(out / "checkpoints"),
                    name_prefix=learner,
                ),
                MetabolicLogger(),
            ]
        )
        model.learn(total_timesteps=frames, callback=callbacks, reset_num_timesteps=True)
        model.save(out / learner)
        report = {
            "learner": learner,
            "frames": int(model.num_timesteps),
            "decoder": decoder,
            "encoder": encoder,
            "init": str(init) if init is not None else None,
            "workers": workers,
            "seed": seed,
            "buffer_size": buffer_size,
        }
        if learner == "encoder":
            report["encoder_version"] = LearnedEncoder.from_actor(model.actor).save(out / "encoder.pt")
        elif learner == "decoder":
            brain = BrainRuntime(encoder=encoder)
            report["export_max_error"] = export_decoder(
                model, brain, out / "decoder.json", ArenaSpec().limits
            )
            report["encoder_version"] = brain.encoder_version
        (out / "round.json").write_text(json.dumps(report, indent=2))
        return report
    finally:
        vec.close()


def validate(decoder, encoder, output, seeds=10, seed_base=9000, seconds=60, workers=6, level=3):
    """End-of-round check on validation seeds: near-dodge (intact, ghost), foraging, E1-E2."""
    from .distill import screen
    from .roam_eval import encoder_checks

    output = Path(output)
    decoder = str(Path(decoder).resolve())
    encoder = str(Path(encoder).resolve())
    key = f"policy:{decoder}"
    report = screen(
        None,
        output.with_suffix(".screen.json"),
        seeds,
        seconds,
        level,
        workers,
        seed_base=seed_base,
        combos=[(key, "none"), (key, "ghost")],
        encoder=encoder,
    )
    checks = encoder_checks(
        decoder, output.with_suffix(".checks.json"), encoder, seeds, seconds, workers, seed_base, level
    )
    none, ghost = report["results"][f"{key}|none"], report["results"][f"{key}|ghost"]
    summary = {
        "decoder": decoder,
        "encoder": encoder,
        "seeds": [seed_base, seed_base + seeds - 1],
        "near_dodge_rate": none["near_dodge_rate"],
        "balanced_dodge_rate": none["balanced_dodge_rate"],
        "ghost_near_dodge_rate": ghost["near_dodge_rate"],
        "beacons_per_min": none["beacons_per_min"],
        "collisions_per_min": none["collisions_per_min"],
        "E1": checks["E1"],
        "E2": checks["E2"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    return summary
```

- [ ] **Step 4: Add the CLI commands (`cli.py`)**

After the `roam-screen` parser block add:

```python
    p.add_argument("--encoder", help="learned encoder .pt for policy/bypass controllers")
    p = sub.add_parser("encoder-collect")
    p.add_argument("--output", required=True)
    p.add_argument("--flights", type=int, default=32)
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed-base", type=int, default=600)
    p.add_argument("--level", type=int, default=3)
    p = sub.add_parser("encoder-clone")
    p.add_argument("data", nargs="+")
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=20000)
    p = sub.add_parser("sac-init-decoder")
    p.add_argument("data", nargs="+", help="stage-1 DAgger .npz files")
    p.add_argument("--encoder", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=4000)
    p = sub.add_parser("sac-round")
    p.add_argument("learner", choices=["encoder", "decoder", "bypass"])
    p.add_argument("--output", required=True)
    p.add_argument("--frames", type=int, required=True)
    p.add_argument("--decoder", help="frozen decoder actor.json (encoder learner)")
    p.add_argument("--encoder", help="frozen encoder .pt (decoder/bypass learner)")
    p.add_argument("--init", help="previous round .zip, or the clone encoder.pt")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p = sub.add_parser("sac-validate")
    p.add_argument("--decoder", required=True)
    p.add_argument("--encoder", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--seed-base", type=int, default=9000)
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--workers", type=int, default=6)
    p = sub.add_parser("encoder-checks")
    p.add_argument("--policy", required=True)
    p.add_argument("--encoder", help="omit for the v4 baseline")
    p.add_argument("--output", required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seconds", type=float, default=120)
    p.add_argument("--workers", type=int, default=6)
```

To the `evaluate` parser add `p.add_argument("--encoder", help="learned encoder .pt (free roam)")`.
Pass `encoder=args.encoder` to `distill.screen(...)` in the `roam-screen` branch and to
`evaluate_free_roam(...)` in the free-roam evaluate branch. In the `roam-screen` branch, map a
`.zip` controller to the E3 bypass:

```python
            controllers = [
                c
                if c in ("teacher", "cue_script", "random")
                else f"bypass:{Path(c).resolve()}"
                if c.endswith(".zip")
                else f"policy:{Path(c).resolve()}"
                for c in args.controllers
            ]
```

Directly after `args = parser.parse_args()` add:

```python
    if args.command in (
        "encoder-collect",
        "encoder-clone",
        "sac-init-decoder",
        "sac-round",
        "sac-validate",
        "encoder-checks",
    ):
        from . import encoder, roam_eval, sac

        if args.command == "encoder-collect":
            result = encoder.collect_clone(
                args.output, args.flights, args.seconds, args.workers, args.seed_base, args.level
            )
        elif args.command == "encoder-clone":
            result = encoder.fit_clone(args.data, args.output, args.steps)
        elif args.command == "sac-init-decoder":
            result = sac.init_decoder(args.data, args.encoder, args.output, args.steps)
        elif args.command == "sac-round":
            result = sac.train_round(
                args.learner,
                args.output,
                args.frames,
                decoder=args.decoder,
                encoder=args.encoder,
                init=args.init,
                workers=args.workers,
                seed=args.seed,
                buffer_size=args.buffer_size,
            )
        elif args.command == "sac-validate":
            result = sac.validate(
                args.decoder, args.encoder, args.output, args.seeds, args.seed_base, args.seconds, args.workers
            )
        else:
            result = roam_eval.encoder_checks(
                args.policy, args.output, args.encoder, args.episodes, args.seconds, args.workers
            )
            result = {k: result[k] for k in ("frames", "E1", "E2")}
        print(json.dumps(result, indent=2))
        return
```

- [ ] **Step 5: Run the tests and the CLI help**

Run:

```bash
env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_sac.py
env -u PYTHONPATH .venv/bin/fly-drone sac-round --help && env -u PYTHONPATH .venv/bin/fly-drone evaluate --help | grep encoder
env -u PYTHONPATH .venv/bin/python -m pytest -q
```

Expected: all pass, and both help texts list the new options.

- [ ] **Step 6: Commit and push**

```bash
.venv/bin/ruff format python tests && .venv/bin/ruff check python tests
git add python/fly_drone/sac.py python/fly_drone/cli.py tests/test_sac.py
git commit -m "Run alternating SAC rounds with validation, and expose encoder v5 commands" && git push
```

---

### Task 10: Document encoder v5 (mechanism, not results)

**Files:**

- Modify: `docs/sensory-model.md` (new §6), `docs/overview/README.md` (§3, §7, §7.3), `docs/training.md` (new §9), `docs/overview/architecture.html`, `AGENTS.md`

- [ ] **Step 1: `docs/sensory-model.md`: add `## 6. Encoder v5 (learned, free roam only)`**

Cover, with the numbers from the spec and Tasks 3–6:

- Input: each eye's Rec. 709 luma as `uint8`, the last 3 frames, a 6 × 48 × 64 stack; an empty stack repeats the first frame; respawn and probe teleports clear it.
- Network: a shared `EyeNet` (conv 16/5×5/s2 → 32/3×3/s2 → 32/3×3/s2 → dense 64), with the right eye mirrored and flagged −1. A linear head gives 8 values, and `tanh + 1` maps them to currents in `[0, 2]`.
- Channel table: `mi1_l/r` 886/887, `tm3_l/r` 1,017/1,037, `lc4_l/r` 71/55, `lplc2_l/r` 94/91. Uniform current per population. Silencing names `light_*` and `looming_*` still work.
- Sensing order: currents for step `t` come from the stack ending with the frame rendered at the end of step `t−1`; zero currents while settling after reset. v4 keeps its order.
- Identity: version `learned-v5:` + the first 16 hex characters of the weights' sha256. Actors load only with the same encoder, and legacy actors stay on v4.
- Metabolic cost `0.01 · mean(loom) + 0.002 · mean(light)` and why it is split.
- Checks E1/E2: definitions and bars copied from `roam_eval.ENCODER_CHECKS` and spec §5.

- [ ] **Step 2: `docs/overview/README.md`**

- §3: one paragraph saying free roam uses the learned v5 encoder while legacy tasks use v4, linking `../sensory-model.md#6`.
- §7: add `### 7.4 Encoder v5: alternating SAC rounds`. Cover the stage list (DAgger on L2 → v4 clone → round-0 decoder → 3 × (encoder SAC 350 k frames, decoder SAC 150 k frames) → evaluation), the asymmetric critic with visible-only geometry, and the SAC objective:
  `J(π) = E[Σ γᵗ (r_t + α H(π(·|o_t)))]`, `Q` targets `y = r + γ (min_j Q̄_j(s′, a′) − α log π(a′|o′))`, with `γ = 0.99`, batch 256, 2 gradient steps per env step, buffer 100 k.
- §7.3: replace "Could the encoder be learned too?" with the evidence (385 false loom escapes in 8 min; loom level 0.49 without a threat vs 0.44 with one) and the decision to learn it, linking the spec.

- [ ] **Step 3: `docs/training.md`: add `## 9. Encoder v5: SAC rounds`**

Copy the exact command sequence from Tasks 11–15 below (paths under `runs/v5/`), the stop rule, the training-only wall band grey randomisation (`BAND_GREY = (0.0, 0.15)`), the ≤ 6 workers limit and the memory note (≈ 5.3 GB buffer + ≈ 0.9 GB per worker).

- [ ] **Step 4: `docs/overview/architecture.html` and `AGENTS.md`**

- Run `grep -n "encoder\|cue" docs/overview/architecture.html`. In the sensing box of each diagram that shows the encoder, add a second label line: `free roam: learned v5 CNN → 8 population currents`.
- In `AGENTS.md`, replace the "The brain stays the fly's" row text with:
  `Wiring, weights, signs and neuron parameters are frozen. Learning lives in the decoder and, for free roam, the v5 sensory encoder (camera images only, 8 uniform population currents in [0, 2]; see docs/sensory-model.md §6). Legacy actors stay on encoder v4.`

- [ ] **Step 5: Format and commit**

```bash
node_modules/.bin/prettier --write docs/sensory-model.md docs/overview/README.md docs/training.md AGENTS.md
git add docs AGENTS.md && git commit -m "Document the learned v5 encoder and its SAC training rounds" && git push
```

---

## Phase B: runs (ask the user before each task; never in parallel with a rebuild)

Start every long run detached, as `setsid nohup <command> > runs/v5/<name>.log 2>&1 &`. The harness has
falsely killed plain background jobs for "low memory". Watch the log and check `free -g` a few
minutes in. Report the numbers to the user at the end of each task, including failures.
Runs are git-ignored. Copy only final reports to `docs/results/`.

### Task 11: Stage 1: DAgger decoder on L2 with v4 (≈ 2 h)

- [ ] **Step 1: Ask the user to start stage 1.**
- [ ] **Step 2: Iteration 0 (teacher only)**

```bash
env -u PYTHONPATH .venv/bin/fly-drone roam-collect --output runs/v5/dagger/it0.npz --flights 128 --seconds 60 --levels 2 --workers 6 --seed-base 200
env -u PYTHONPATH .venv/bin/fly-drone roam-fit runs/v5/dagger/it0.npz --output runs/v5/dagger/it0
env -u PYTHONPATH .venv/bin/fly-drone roam-screen runs/v5/dagger/it0/warm-actor.json teacher random --level 2 --seeds 10 --seed-base 9000 --workers 6 --output runs/v5/dagger/it0-screen.json
```

- [ ] **Step 3: Iterations 1–3 (student in the loop)**

For `k, beta` in `(1, 0.5), (2, 0.25), (3, 0.0)`:

```bash
env -u PYTHONPATH .venv/bin/fly-drone roam-collect --output runs/v5/dagger/it$k.npz --student runs/v5/dagger/it$((k-1))/warm-actor.json --beta $beta --flights 128 --seconds 60 --levels 2 --workers 6 --seed-base $((200 + 1000*k))
env -u PYTHONPATH .venv/bin/fly-drone roam-fit runs/v5/dagger/it*.npz --output runs/v5/dagger/it$k
env -u PYTHONPATH .venv/bin/fly-drone roam-screen runs/v5/dagger/it$k/warm-actor.json --level 2 --seeds 10 --seed-base 9000 --workers 6 --output runs/v5/dagger/it$k-screen.json
```

- [ ] **Step 4: Choose and report.** Pick the iteration with the highest beacons/min on the validation seeds, breaking ties by fewer collisions/min. Write `runs/v5/dagger/choice.json` with `{"iteration": k, "actor": ..., "beacons_per_min": ..., "collisions_per_min": ..., "teacher_beacons_per_min": ...}`. Report it to the user.

### Task 12: Stage 2a: v4 clone, round-0 decoder, v4 E1 baseline (≈ 1.5 h)

- [ ] **Step 1: Ask the user to start stage 2a.**
- [ ] **Step 2: Collect and fit the clone**

```bash
env -u PYTHONPATH .venv/bin/fly-drone encoder-collect --output runs/v5/clone/data.npz --flights 32 --seconds 60 --workers 6 --seed-base 600
env -u PYTHONPATH .venv/bin/fly-drone encoder-collect --output runs/v5/clone/data2.npz --flights 32 --seconds 60 --workers 6 --seed-base 632
env -u PYTHONPATH .venv/bin/fly-drone encoder-clone runs/v5/clone/data.npz runs/v5/clone/data2.npz --output runs/v5/clone --steps 60000
```

Why (amended by Task 12d after the 30-seed Task 12 gate failure of 2026-09-15: 1.27 vs 1.93 beacons/min on L2 seeds 9000–9029, with avoidance intact but |yaw bias| 0.106 vs 0.077 and 8 vs 3 zero-beacon seeds): the clone steered asymmetrically, so the fit now mirrors a random half of every batch (eyes swapped and flipped, `_l`/`_r` targets swapped; v4 is exactly mirror-symmetric) and trains on a second collection (seeds 632–663); `clone.json` adds `held_out_mirror`.

Why (amended by Task 12c after the second Task 12 gate failure of 2026-09-15): the clone copied each light channel at r 0.99 but the left−right light difference only at r 0.85, the side signal the brain steers to beacons with; the fit now also matches pathway differences, oversamples light-side frames, and runs 60k steps.

Report the held-out `mse` and `r` per channel, and `r`, `rmse`, `gain` per pathway difference (`held_out_differences`), and `r` per pathway from `held_out_mirror`, from `runs/v5/clone/clone.json`.

- [ ] **Step 3: Round-0 decoder on the clone, from data collected under the clone**

```bash
env -u PYTHONPATH .venv/bin/fly-drone roam-collect --output runs/v5/round0-data/it0.npz --flights 128 --seconds 60 --levels 2 --workers 6 --seed-base 200 --encoder runs/v5/clone/encoder.pt
env -u PYTHONPATH .venv/bin/fly-drone sac-init-decoder runs/v5/round0-data/it0.npz --encoder runs/v5/clone/encoder.pt --output runs/v5/round0
```

Why (amended by Task 12b after the Task 12 gate failure of 2026-09-15): the stage-1 DAgger files were flown with v4, and the clone gives different DN/motor traces for the same flights, so the decoder must be fitted on traces recorded under the clone (`sac-init-decoder` now rejects v4 files for a learned encoder); the warm start fits pre-tanh so saturated labels are not underfit.

Expected: `export_max_error ≤ 1e-4` in `runs/v5/round0/warm-start.json`.

- [ ] **Step 4: Sanity gate before any SAC (engineering check, not acceptance)**

```bash
env -u PYTHONPATH .venv/bin/fly-drone roam-screen runs/v5/round0/decoder.json --encoder runs/v5/clone/encoder.pt --level 2 --seeds 10 --seed-base 9000 --workers 6 --output runs/v5/round0/l2-screen.json
```

Compare with the chosen DAgger actor's screen from Task 11 (same seeds). If beacons/min is below 0.8× that actor's, **stop and report**: the clone or the tanh head lost the skill, and SAC would start from a broken system.

- [ ] **Step 5: Validation baseline and the v4 E1 reference**

```bash
env -u PYTHONPATH .venv/bin/fly-drone sac-validate --decoder runs/v5/round0/decoder.json --encoder runs/v5/clone/encoder.pt --output runs/v5/round0/validation.json
env -u PYTHONPATH .venv/bin/fly-drone encoder-checks --policy runs/v5/dagger/it<chosen>/warm-actor.json --output runs/v5/e1-v4-baseline.json
```

(`encoder-checks` with no `--encoder` measures v4 on the 50 evaluation seeds. This is the "v4 baseline reported alongside" E1.)

### Task 13: Rounds 1–3: alternating encoder and decoder SAC (≈ 1 h per round)

For `k = 1, 2, 3`, with `PREV_DEC = runs/v5/round0/decoder.json` and `PREV_DEC_ZIP = runs/v5/round0/decoder.zip` for k = 1, otherwise `runs/v5/round$((k-1))/decoder/decoder.json` and `.../decoder.zip`. Set `ENC_INIT = runs/v5/clone/encoder.pt` for k = 1, otherwise `runs/v5/round$((k-1))/encoder/encoder.zip`.

- [ ] **Step 1: Ask the user to start round k.**
- [ ] **Step 2: Encoder SAC (decoder frozen)**

```bash
env -u PYTHONPATH .venv/bin/fly-drone sac-round encoder --output runs/v5/round$k/encoder --frames 350000 --decoder $PREV_DEC --init $ENC_INIT --workers 6 > runs/v5/round$k-encoder.log 2>&1
```

- [ ] **Step 3: Decoder SAC (encoder frozen)**

```bash
env -u PYTHONPATH .venv/bin/fly-drone sac-round decoder --output runs/v5/round$k/decoder --frames 150000 --encoder runs/v5/round$k/encoder/encoder.pt --init $PREV_DEC_ZIP --workers 6 > runs/v5/round$k-decoder.log 2>&1
```

- [ ] **Step 4: Validate**

```bash
env -u PYTHONPATH .venv/bin/fly-drone sac-validate --decoder runs/v5/round$k/decoder/decoder.json --encoder runs/v5/round$k/encoder/encoder.pt --output runs/v5/round$k/validation.json
```

- [ ] **Step 5: Apply the stop rule and report.** Show the user a table of rounds 0..k: near-dodge, balanced, ghost near-dodge, beacons/min, collisions/min, E1 AUC, mean metabolic cost (last `rollout/metabolic_cost` in the encoder log). **Eligibility (amended 2026-09-16):** a round counts only if `roam_eval.round_eligible` passes — beacons/min ≥ ½ of round 0's, ghost near-dodge ≤ `A3_max_ghost_dodge_rate` (0.3), and E2 passes. If round k is ineligible, or is eligible but does not beat the best eligible earlier round's `near_dodge_rate`, **stop and report** rather than starting round k+1. A near-dodge gain that costs foraging or is not causal is not progress.

- [ ] **Step 6: Choose the final pair.** Take the best **eligible** round (`roam_eval.pick_best_round`) by validation near-dodge rate (ties go to more beacons/min); round 0 is the fallback and is always eligible, so a run where every SAC round is degenerate ends on the round-0 pair:

```bash
mkdir -p runs/v5/final && cp runs/v5/round<best>/encoder/encoder.pt runs/v5/round<best>/decoder/decoder.json runs/v5/final/
```

### Task 14: E3: brain-bypass control (≈ 1.5 h)

- [ ] **Step 1: Ask the user to start the bypass run.**
- [ ] **Step 2: Train a decoder that reads the 8 currents directly** (the budget of three decoder rounds)

```bash
env -u PYTHONPATH .venv/bin/fly-drone sac-round bypass --output runs/v5/bypass --frames 450000 --encoder runs/v5/final/encoder.pt --workers 6 > runs/v5/bypass.log 2>&1
```

- [ ] **Step 3: Screen both on the evaluation seeds and compare**

```bash
env -u PYTHONPATH .venv/bin/fly-drone roam-screen runs/v5/final/decoder.json runs/v5/bypass/bypass.zip --encoder runs/v5/final/encoder.pt --ablations none ghost --seeds 50 --seconds 120 --level 3 --seed-base 1000 --workers 6 --output runs/v5/e3-screen.json
env -u PYTHONPATH .venv/bin/python -c "
import json; from pathlib import Path; from fly_drone.roam_eval import bypass_comparison
r = json.loads(Path('runs/v5/e3-screen.json').read_text())['results']
full = next(v for k, v in r.items() if k.startswith('policy:') and k.endswith('|none'))
byp = next(v for k, v in r.items() if k.startswith('bypass:') and k.endswith('|none'))
out = bypass_comparison(full, byp); Path('runs/v5/e3.json').write_text(json.dumps(out, indent=2)); print(out)"
```

If `bypass_better` is true, **stop and report to the user** before Task 15 conclusions.

### Task 15: Stage 4: evaluation, E1–E4, results (≈ 3–3.5 h at 6 workers)

The spec estimated ~1 h. At ≤ 6 workers, 10 conditions × 50 seeds × 120 s plus A5 probes take ≈ 3 h. Tell the user this number when asking.

- [ ] **Step 1: Ask the user to start the final evaluation.**
- [ ] **Step 2: Pre-registered acceptance A1–A7 and encoder checks E1–E2**

```bash
env -u PYTHONPATH .venv/bin/fly-drone evaluate --task free_roam --policy runs/v5/final/decoder.json --encoder runs/v5/final/encoder.pt --episodes 50 --workers 6 --output runs/v5/evaluation.json > runs/v5/evaluation.log 2>&1
env -u PYTHONPATH .venv/bin/fly-drone encoder-checks --policy runs/v5/final/decoder.json --encoder runs/v5/final/encoder.pt --output runs/v5/e1-e2.json
```

- [ ] **Step 3: E4 legacy reproducibility**

```bash
env -u PYTHONPATH .venv/bin/python -m pytest -q tests/test_arena.py::test_legacy_room_mjcf_unchanged tests/test_env.py -k "legacy or replay_is_bit_identical"
```

- [ ] **Step 4: Record results honestly**

Copy `runs/v5/evaluation.json`, `e1-e2.json`, `e1-v4-baseline.json`, `e3.json` and every round's `validation.json` to `docs/results/encoder-v5/`. In `docs/validation.md`, add a section "Free roam with encoder v5" that tables A1–A7 (pass/fail with values), E1 (v5 vs v4 AUC), E2 margins, E3 comparison and E4. Use the thresholds straight from `ACCEPTANCE` and `ENCODER_CHECKS`, never rounded in the pass direction. Do **not** add the actor to `docs/results/accepted-policies.json` unless A1–A6, E1, E2 and E4 all pass and the user agrees.

- [ ] **Step 5: Commit, push and report**

```bash
node_modules/.bin/prettier --write docs/validation.md
git add docs/results/encoder-v5 docs/validation.md && git commit -m "Record encoder v5 free-roam evaluation and encoder checks" && git push
```

Report every A and E line to the user, failures included. If anything failed, name which pathway or stage the evidence points at, and don't propose changing any threshold.
