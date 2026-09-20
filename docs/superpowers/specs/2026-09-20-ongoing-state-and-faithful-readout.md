# Ongoing state and a faithful readout — design

**Status:** proposed 2026-09-20. Supersedes the "add more senses" branch of P3. Documentation
first; no source under `python/fly_drone/` or `crates/` changes until this spec lands.

**Why now.** The frozen connectome drives the body through a declared, teacher-free codec and threat
escape is real and causal — but the drone idles most of the time. Three rounds of sensory-side
engineering (v5 encoder, v6 retinotopic, P3 optic-flow relay) were spent on this and all three came
back null-to-negative. [`../../results/liveness/FINDING-2026-09-20.md`](../../results/liveness/FINDING-2026-09-20.md)
concluded "the next lever is control/excitability, not more sensory channels". This spec carries
that one step further: the idling has a measured arithmetic cause, and it is not a mystery.

**Parent spec:** [`2026-09-19-body-agnostic-fidelity-cyborg-design.md`](2026-09-19-body-agnostic-fidelity-cyborg-design.md).
This is its **P4**, and it comes before P4-transfer and P5-evaluation in that spec's §9.

---

## 1. The governing principle this spec adds

> **Liveness is generated in the brain and only read by the bridge.**

The project already caught and reversed this exact failure once. [`../../free-roam.md:85`](../../free-roam.md)
records that a constant explore drive in the bridge (`[0.8, 0, 0, 0.25]`, cloned onto every decoder)
was "the source of 'the drone flies like it has a life of its own'", and it was removed as
engineered behaviour. Making the drone livelier is now the explicit goal, which makes that shortcut
the single most tempting move available. It is ruled out.

Concretely: a constant in the codec is forbidden. Tonic drive **on identified neurons**, background
excitability and membrane noise are permitted — they are state, not decisions — and each is a
declared, versioned, additive identity with a silencing gate, exactly like C1 feedback and C2
dynamics. The difference is not cosmetic: a bridge constant produces motion no ablation can remove,
while neural tonic drive is silenceable and therefore falsifiable.

This clause belongs in [`../../overview/README.md`](../../overview/README.md) §1 alongside the
existing contract clauses.

---

## 2. Fact 1 — the codec cannot command a speed that counts as moving

`Adapter.command` ([`../../../python/fly_drone/adapter.py:242-262`](../../../python/fly_drone/adapter.py))
sets forward speed as

```
vx = g_fwd · max(0, power − rest_power)
```

with `g_fwd = 0.69203`, `rest_power = 0.439525` (committed `../../results/adapter/adapter.json`) and
`plant.LIMITS[0] = 0.4 m/s`. A frame counts as stuck below **0.05 m/s** (`distill.py:384`,
`env.py:677`). Crossing those constants with the measured per-stimulus `power` deltas in the
committed `../../results/liveness/readout-audit.json`:

| stimulus           |    Δpower |  commanded vx | ≥ 0.05 m/s? |
| ------------------ | --------: | ------------: | ----------- |
| `light_l`          |     0.036 |     0.010 m/s | no          |
| `light_r`          |     0.074 |     0.021 m/s | no          |
| `light_both`       |     0.109 |     0.030 m/s | no          |
| `loom_l`           |     0.153 |     0.042 m/s | no          |
| **`loom_r`**       | **0.197** | **0.055 m/s** | **yes**     |
| `relay:cruise_yaw` |     0.016 |     0.004 m/s | no          |

**Only a full one-sided loom clears the bar.** A bright light filling both eyes commands 3 cm/s and
is scored as stationary. Even if every flight-power motoneuron saturated (`power` → 1.0), the
ceiling is 0.155 m/s — 39 % of the body's range.

This is precisely the reported behaviour: vivid escape, idling otherwise. The brain is not failing
to decide. It is deciding, and being handed a throttle that barely opens.

Three compounding causes, all inside the codec:

1. **Half the channel is discarded.** `max(0, ·)` (`adapter.py:256`) rectifies away every excursion
   below rest, so the brain can command "faster than rest" but never "slower".
2. **The calibration battery fights itself.** `STIMULI` (`adapter.py:48-56`) declares `vx = 0` for
   the two loom rows, yet loom produces the _largest_ `power` excursion in the battery. The
   least-squares `gain()` (`adapter.py:143-147`) is therefore dragged down by the rows that move the
   signal most. `g_fwd` is small **because** loom raises power and loom is declared to mean stop.
3. **Steering reads the wrong cells.** `_laterality()` (`adapter.py:105-107`) takes a
   left-minus-right mean over all 2,022 descending/VNC traces — so small it needs `g_yaw = 210.7`
   to reach a unit turn. Meanwhile `brain.py:180-198` already derives `steer_l`/`steer_r` from the
   fly's actual wing steering motoneurons (`b1/b2/b3 MN`, `i1/i2 MN`, `iii1/iii3 MN`, `hg1-4 MN`),
   those readouts move measurably in the audit (0.05–0.20), **and the codec never reads them**.
   `wing_l`/`wing_r` (30 cells each) and `thrust` (16) are likewise defined and unused;
   `yaw_torque` is an empty group.

---

## 3. Fact 2 — the brain has no ongoing activity to read

`BrainRuntime` applies a tonic bias of `0.85` to exactly one population: the DLMn/DVMn flight-power
motoneurons (`brain.py:202-203`). **Every one of the other 166,698 neurons has `bias = 0.0` and
`v_threshold = 1.0`**, and with `leak ≈ 0.7788` a cell needs sustained drive above `0.221` to fire
at all. Measured: `mean_feature_activity = 0.0048` at rest; 24 of 2,022 traces above 0.05.

A brain with no ongoing activity can only be reactive. That is the behaviour profile observed.

The gap is sharpest at the steering motoneurons. In a flying fly, b1/hg/i/iii MNs fire essentially
once per wingbeat, continuously; steering is phase and amplitude modulation of an ongoing train, not
recruitment from silence. The project already adopted that hypothesis — for the _power_ muscles
only. Applying it to the power MNs and not the steering MNs is why the model has thrust-at-rest,
no steering-at-rest, and a `steer_l − steer_r` with no dynamic range.

Two supporting facts from the core:

- `refrac_ticks = round(2.0 ms / 5.0 ms) = 0` (`crates/brain-core/src/core/lif.rs:40-44`): there is
  **no refractory period at all**. Cells are silent or fire every tick, with no graded rate code in
  between. Recorded here as a known fidelity gap; changing `dt` changes everything downstream, so
  it is documented, not acted on.
- The one dynamics experiment ever run varied only `noise_sigma` (0.02 → 0.00) and made idling
  _worse_ (`slow_fraction` 0.627 → 0.673). Background fluctuation is load-bearing for mobility.

---

## 4. Fact 3 — 9,189 neurons are wired in and transmit nothing

Measured directly from the canonical `graph.bin` CSR (out-edges present, every out-weight exactly
zero):

| transmitter   |   cells | with out-edges | total out-weight |
| ------------- | ------: | -------------: | ---------------: |
| histamine     |   7,891 |          6,179 |            **0** |
| unclear       |   2,999 |          2,489 |            **0** |
| dopamine      |     392 |            391 |            **0** |
| octopamine    |     101 |             82 |            **0** |
| serotonin     |      48 |             48 |            **0** |
| acetylcholine | 103,720 |        103,310 |       603.0 mean |
| glutamate     |  29,302 |         29,085 |       589.7 mean |
| gaba          |  22,069 |         22,045 |       996.7 mean |

**9,189 neurons (5.5 % of the network) and 332,316 edges (3.2 %) carry no current.** They integrate
input and spike; they transmit nothing. This is the declared modelling hypothesis recorded in
`manifest.json` ("unresolved/modulatory source current zero") and `docs/neuron-model.md:68-73`,
quantified for the first time here. Three consequences the project has not yet drawn:

1. **The histaminergic photoreceptor relay is electrically dead.** 6,179 histaminergic cells
   transmit nothing, so photoreceptor→lamina carries no signal. The parent spec's §7 P3 proposed to
   "validate/activate photoreceptor→lamina rather than only pooled v4 cues"; the relay that shipped
   injects directly into T4/T5, _bypassing_ the dead layer rather than restoring it. That was the
   right call for a declared front-end, but it means the relay's null result says nothing about the
   real relay.
2. **The s2/s3 sign-convention results need re-reading.** `make_sign_bundle.py` retags the
   inhibitory bit in `neurons.bin` only; `graph.bin` is byte-identical across canonical and
   `malecns-sign-s2` (verified with `cmp`). Retagging the sign of a neuron whose out-weights are all
   zero is a no-op, so the s2 convention (which adds histamine to the inhibitory set) **could not
   have had any effect through histamine**. "S2 is indistinguishable on the probe"
   (`external-prior-art.md:196`) has a mechanical explanation and is not evidence about the sign
   hypothesis.
3. **Neuromodulatory tone cannot be switched on inside this repo.** Biasing an octopaminergic cell
   makes it spike into zero-weight edges. Restoring modulatory transmission means rebuilding
   `graph.bin` in the upstream fly-playground pipeline. **Out of scope here; a separate, later
   plan.** Recorded so the option is not silently forgotten.

---

## 5. Fact 4 — the measuring stick cannot see the complaint

`liveness.LIVENESS` (`python/fly_drone/liveness.py:16-23`) sets L1 `slow_fraction ≤ 0.25` and L2
`coverage ≥ 0.15`, chosen with the RL teacher's numbers (0.009 / 26.2 %) as context. Two problems:

- **It re-imports the abandoned target.** Anchoring "alive" to an RL teacher reintroduces the thing
  the v5/v6 programme was closed for.
- **It cannot measure the actual complaint.** A brick drifting at 0.06 m/s on a constant heading
  passes L1 and L2. A fly-like flight of straight segments punctuated by body saccades and genuine
  pauses may fail both. "Idling" is a statement about the _structure_ of motion, and nothing
  currently measures structure.

The fix is to anchor the bar on published _Drosophila_ free-flight statistics and demote the teacher
to a context row. The bar only gets stricter: L1–L5 stay, L6–L8 are added.

---

## 6. Facts an implementer needs before touching code

These were verified on 2026-09-20. They are the non-obvious ones; get them wrong and the work
silently does nothing.

**F1. Per-frame series are not recorded anywhere.** The rollout loop in
`python/fly_drone/distill.py:353-397` keeps only aggregates per episode (`slow_fraction`,
`mean_yaw_command`, `visited_cells`). `yaw_commands` is a local list, discarded after the mean.
**Motion-structure statistics therefore require adding per-frame recording** — see A1.

**F2. Position and heading are not in `info`.** `env.step` returns `info["speed"]` (horizontal
speed, `env.py:677`), `info["visited_cells"]`, `info["collisions"]`, `info["collision_kinds"]`,
`info["threats"]`, `info["beacons_collected"]`. Position and yaw must be read from the plant:
`env.plant.pos[0]` is `(x, y, z)` in metres and `env.plant.rpy[0, 2]` is body yaw in radians.

**F3. Frame rate is 25 Hz.** `FRAME_SECONDS = 0.04` (`env.py:12`). One `env.step` = 8 brain ticks
at 200 Hz = 8 plant sub-steps at 200 Hz. `frames = int(seconds / 0.04)` in the rollout loop.

**F4. `summarise(runs)` averages across seeds** (`distill.py:433-447`) with the pattern
`float(np.mean([r["<key>"] for r in runs]))`. Every new per-episode scalar must be added there or
it will not reach the liveness report.

**F5. `liveness(results, policy, controls)`** (`python/fly_drone/liveness.py:27`) reads summaries
keyed `f"{controller}|{ablation}"` and uses `_per_seed(summary, metric)` +
`paired_bootstrap(a, b)` from `roam_eval` for confidence intervals. New criteria follow the same
shape: a dict per criterion with its value, its bar, and `passed`, folded into the final
`out["passed"] = all(...)`.

**F6. Adding a manifest key to `identity.MODEL_KEYS` does not change the canonical hash.**
`canonical_model` builds `{k: manifest[k] for k in MODEL_KEYS if k in manifest}`
(`identity.py:41-43`), so a key absent from the canonical manifest contributes nothing. A new
`"tonic"` marker must be added to **both** `MODEL_KEYS` and `ALTERNATE_KEYS` (`identity.py:18-37`)
for the C3a bundle to be a properly pinned alternate identity. Verify the canonical `bundle_hash`
stays `edc5439e…` afterwards.

**F7. The tonic bias lives in Python, not in a bundle file.** `brain.py:202-203` and `:268` call
`self.core.bias(self.tonic, 0.85)` directly. `Brain.bias(ids, value)` is exposed through PyO3 and
clamps to `[0, 2]` (`crates/brain-core/src/core/sim.rs:153-162`). C3a therefore declares its tonic
map in the manifest and has `BrainRuntime` apply it — no new binary format is needed.

**F8. The adapter is loaded once per process and cached.** `adapter.load_default` memoises in the
module-global `_DEFAULT` (`adapter.py:265-278`). A `--adapter` flag must thread a path through
`declared_command(..., path=...)`, which already accepts one (`adapter.py:281-289`); do not rely on
mutating the cache.

**F9. `Adapter.load` self-certifies.** It recomputes the identity hash from `params` +
`dataset_hash` (+ `visual`) and refuses a file whose stored `adapter_version` disagrees
(`adapter.py:202-208`). Any new field that changes behaviour must be folded into `_version`
(`adapter.py:164-170`), exactly as `visual` is, or two different codecs will share one identity.

---

## 7. Workstream A — codec v2 and a fly-anchored bar

Bridge only. **No contract change**: learning is allowed to live in the bridge. The committed
`docs/results/adapter/adapter.json` and `adapter-check.json` stay byte-identical throughout.

### A0 — record real fly free-flight statistics (blocking)

**Why it blocks:** A1's thresholds must be _derived_ from published numbers. Inventing them, or
reusing the teacher's numbers, reproduces the failure this spec is correcting (§5).

Add a subsection to `docs/references.md` with, for each statistic, the value, the species and
condition (free flight vs. tethered vs. walking; arena size), and the citation:

- body-saccade rate (Hz) and amplitude (rad) in free flight
- inter-saccade interval distribution (median, spread)
- translational speed distribution (median, p90)
- moving-bout and pause-bout duration distributions
- turn-angle distribution between bouts
- mean-squared-displacement exponent, if reported

Candidate sources **to verify, never to cite unread**: Tammero & Dickinson 2002 (free-flight
saccades); Censi et al. 2013 (saccade initiation); Muijres et al. 2014 (escape manoeuvres);
van Breugel & Dickinson 2012 (visual search); Berman et al. 2014 (behavioural space). If a
statistic has no defensible published value, **say so and omit that criterion** rather than invent
a bar. It is legitimate for A0 to return "only three of the six are citable"; L6–L8 then cover only
those three.

Record the derived bars in one place: a `FLY_REFERENCE` dict in `motion_stats.py` carrying value,
units, and citation key per statistic, so every threshold traces to a line in `references.md`.

### A1 — `python/fly_drone/motion_stats.py` + L6–L8

**A1.1 Record per-frame series.** In the rollout loop (`distill.py:353-397`), alongside the existing
`yaw_commands.append(...)`, accumulate three lists: `speeds.append(info["speed"])`,
`positions.append(tuple(env.plant.pos[0][:2]))`, `headings.append(float(env.plant.rpy[0, 2]))`.
Per F2 these come from the plant, not `info`. Keep them local.

**A1.2 Reduce before storing.** Do **not** put the raw series in the run dict —
`adapter-check.json` is already 1.07 MB and 50 seeds × 3,000 frames would balloon it. Call
`motion_stats.episode_stats(speeds, headings, positions, dt=0.04)` inside the loop and merge the
returned scalars into the run dict.

**A1.3 The statistics.** Pure functions over 1-D arrays, no simulation, no imports from `env`:

```python
MOVING_THRESHOLD = 0.05   # m/s; identical to distill.py:384 so bouts and slow_fraction agree

def episode_stats(speeds, headings, positions, dt=0.04) -> dict
```

- **Bouts.** `moving = speeds >= MOVING_THRESHOLD`; run-length-encode into alternating move/pause
  bouts. Emit `mean_move_bout_s`, `median_move_bout_s`, `mean_pause_bout_s`, `bouts_per_min`,
  `bout_length_cv`. _This is the metric that separates the three failure modes:_ a brick has one
  endless pause bout, a bridge-constant cruise has one endless move bout, a fly alternates.
- **Saccades.** `omega = np.diff(np.unwrap(headings)) / dt` (rad/s). A saccade is a local maximum of
  `|omega|` above `SACCADE_THRESHOLD` with a minimum separation of `SACCADE_REFRACTORY` seconds
  (both from `FLY_REFERENCE`). Emit `saccade_rate_hz`, `mean_saccade_amplitude_rad`, `median_isi_s`,
  `isi_cv`. Amplitude is the integral of `omega` across the peak's contiguous above-threshold span.
- **Speed shape.** `median_speed`, `p90_speed`, `speed_cv`.
- **Exploration structure.** `msd_exponent`: least-squares slope of `log MSD(tau)` against
  `log tau` for `tau` in `[0.2 s, 5.0 s]`, where
  `MSD(tau) = mean_t ||pos[t+tau] - pos[t]||^2`. Interpretation: ~0 stuck, ~1 diffusive,
  ~2 ballistic. Return `None` when the episode has fewer than 2·(5/dt) frames.

Every function must tolerate degenerate input (all-stationary, single frame, constant heading) and
return `None` rather than raise or emit `nan`. `None` propagates to "criterion not evaluable", never
to "passed".

**A1.4 Aggregate.** Add each new scalar to `summarise` (F4) with the existing
`float(np.mean([...]))` pattern, skipping `None`s.

**A1.5 Criteria L6–L8** in `liveness.LIVENESS`, each a two-sided window derived from
`FLY_REFERENCE` in A0:

| id     | criterion             | statistic                                                | shape of bar                                                                                              |
| ------ | --------------------- | -------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **L6** | intermittency         | `bouts_per_min`, `mean_move_bout_s`, `mean_pause_bout_s` | each inside the cited fly window — catches the brick _and_ the endless-cruise                             |
| **L7** | saccadic turning      | `saccade_rate_hz`, `mean_saccade_amplitude_rad`          | inside the cited window — catches constant-yaw drift                                                      |
| **L8** | exploration structure | `msd_exponent`                                           | inside the cited window (super-diffusive, not ballistic) — catches a straight-line drifter that passes L2 |

Keep L1–L5 exactly as they are. **The bar only gets stricter; nothing pre-registered is weakened.**
Demote the teacher to a context row: it may appear in the report, never in a `passed`. Extend
`out["passed"]` to `all(l1..l8)`, treating a `None` criterion as not-evaluable and reporting it
explicitly rather than silently passing.

**A1.6** `liveness_check` in `roam_eval.py` needs no new combos — L6–L8 read the same intact
summary L1/L2 already read.

### A2 — `scripts/command_audit.py` (the free falsifier)

Mirror `scripts/readout_audit.py` in shape and tone. For each declared stimulus in
`readout-audit.json` and each codec (`adapter.json`, `adapter-v2.json`), reconstruct the readout
vector implied by the audit's deltas, call `Adapter.command`, and emit the commanded
`[vx, vy, vz, yaw]` in **physical units** (multiply by `plant.LIMITS = [0.4, 0.4, 0.2, 0.8]`)
against the 0.05 m/s threshold. Write `docs/results/liveness/command-audit.json` and print the §2
table.

**Run this before spending a single seed.** If codec v2 does not lift the non-loom stimuli above
0.05 m/s here, it has already failed and no simulation is needed.

### A3 — codec v2

New identity `declared-v2`, written to `docs/results/adapter/adapter-v2.json`. Canonical
`adapter.json` untouched. Keep `_latch`, `ESCAPE_SCALE`, `SIDE_FRACTION`, `g_climb` and the escape
path **exactly as they are** — that pathway works; do not renovate it.

Four changes, each separately ablatable:

**A3.1 Steering from the steering motoneurons.** Replace `_laterality` (the 2,022-cell population
mean, needing `g_yaw = 210.7`) with the readouts already derived at `brain.py:180-198`:

```
steer_raw = (mean(steer_l) - mean(steer_r)) - rest_steer_diff
steer     = clip(steer_raw / steer_span, -1, 1) * (1 - 2*loom)
```

Read them through the same possibly-ablated `features` vector the codec is handed, using
`feature_positions(brain)` — **never through `brain.read()`**, which would be a side channel around
the ablation and would silently invalidate every causal control.

**A3.2 Two-sided forward drive.** Drop the `max(0, ·)` at `adapter.py:256`:

```
drive = clip((power - rest_power) / power_span, -1, 1)
vx    = g_fwd * drive
```

so the brain can command slowing as well as speeding.

**A3.3 Span normalisation.** `steer_span` and `power_span` are the maximum absolute excursion from
rest measured across the calibration battery, stored in `params`. This is what lets a 0.05 readout
excursion become a usable command instead of rounding to nothing.

**A3.4 Stop the battery fighting itself.** Fit `g_fwd` only on rows whose declared `vx` target is
non-zero, so the loom rows (target `vx = 0`, largest `power` excursion) no longer drag the gain
down. Minimal change: pass a row mask to the existing `gain()` helper. Alternative, if the mask
proves awkward: re-declare the loom rows' `vx` target to a defensible non-zero value. Either way,
**record which was chosen and why** — this is the step most likely to be quietly fudged.

**A3.5 A calibration-time falsifier.** Extend the existing sanity check at `adapter.py:159-160`:
after fitting, assert that the brightest **non-loom** battery stimulus commands at least
0.05 m/s (`g_fwd * drive * 0.4 >= 0.05`). If it does not, raise with the computed value. A codec
that cannot command motion must fail loudly at calibration rather than ship another idler.

**A3.6 Identity.** Fold a `codec: "v2"` field into `_version` (F9) exactly as `visual` is folded,
so v1 and v2 can never be confused. `Adapter.command` branches on it; both versions stay loadable.

**A3.7 CLI.** `--adapter {v1,v2}` on `liveness-check` and `adapter-check`, threading a path into
`declared_command(..., path=...)` (F8). Default stays v1 until v2 wins a gate.

### A4 — tests

`tests/test_motion_stats.py`:

- synthetic square-wave speed → known bout count and durations
- synthetic impulse train in heading → known saccade rate and amplitude
- pure straight-line motion → `msd_exponent ≈ 2`; frozen position → stuck-case `None`/~0
- all-stationary, one-frame and constant-heading inputs return `None`, never `nan` or an exception
- every `FLY_REFERENCE` entry carries a citation key

`tests/test_adapter_v2.py`:

- `adapter.json` byte-identical after a v2 calibration (the guard already used for
  `adapter-relay.json`)
- v1 and v2 identities differ; each refuses to load under the other's `adapter_version`
- forward drive is two-sided: `power < rest_power` yields `vx < 0`
- steering responds to a `steer_l`/`steer_r` asymmetry injected into the **features vector**, and
  goes to zero when that vector is zeroed (proves no side channel)
- the A3.5 calibration falsifier raises on a synthetic brain that cannot reach 0.05 m/s
- source assertion: no `teacher` import anywhere in the behaviour path

---

## 8. Workstream B — C3 ongoing state

**C3 is a contract change**, declared and versioned exactly like C1 (feedback) and C2 (dynamics):
new `bundle_hash`, manifest marker, opt-in flag, silencing gate. The canonical bundle and the
accepted actors never load it. **Requires explicit user sign-off before the bundle is built.**

### B0 — the neuromodulator question (done)

Answered in §4 by `scripts/transmission_audit.py`. Modulatory transmission is zeroed in `graph.bin`;
neuromodulatory tone needs an upstream fly-playground rebuild and is **out of scope**. Do not spend
time here.

### B1 — C3a, tonic drive on the flight motoneuron set

**Biological anchor.** The fly's wing steering motoneurons (`b1/b2/b3 MN`, `i1/i2 MN`,
`iii1/iii3 MN`, `hg1-4 MN`) fire about once per wingbeat continuously in flight; steering is phase
and amplitude modulation of that ongoing train, not recruitment from silence. The project already
adopted exactly this hypothesis for the power muscles (`b = 0.85` on DLMn/DVMn). C3a applies it
consistently. **This is not a new hypothesis — it is the existing one, stopped halfway.**

**B1.1 Marker.** Add a `tonic` key to the alternate manifest:

```json
"tonic": {
  "version": "declared-tonic-v1",
  "source": "declared; flight-state tonic drive on identified motoneurons; no activity fit",
  "roles": {"power_l": 0.85, "power_r": 0.85, "steer_l": <b>, "steer_r": <b>}
}
```

Add `"tonic"` to **both** `identity.MODEL_KEYS` and `identity.ALTERNATE_KEYS` (F6). Assert in a test
that the canonical `bundle_hash` is unchanged afterwards.

**B1.2 Runtime.** `BrainRuntime` reads the marker and applies the map in place of the hard-coded
pair at `brain.py:202-203` and `:268`; absent the marker, behaviour is byte-identical to today.
Note the `[0, 2]` clamp in `sim.rs:153-162` (F7).

**B1.3 Choosing `b`.** Not by behaviour. Use the closed form already worked in
`docs/neuron-model.md` §2: a constant drive `b` free-runs a cell when `b/(1-lambda) > v_th`, i.e.
`b > 0.221`, and `b = 0.85` gives one spike every second tick (activity trace `a = 0.5`). Pick `b`
for the steering MNs so their resting activity trace sits near **0.5** — mid-rate, so synaptic input
can push it both up and down. Verify with `scripts/readout_audit.py`, which already measures resting
readouts. **Record the arithmetic, not a tuned number.**

**B1.4 Build script.** A sibling of `scripts/make_feedback_bundle.py` (symlink the shared immutable
files, write the marker) producing `data/malecns-tonic/`. Add it to `.gitignore` next to the other
generated bundles.

**B1.5 Gate.** `silence_tonic` ablation on `BrainRuntime` (restore `b = 0` for the added roles),
registered in `env.PATHWAYS`-style dispatch at `env.py:207-214`, plus a `tonic-check` command
mirroring `roam_eval.feedback_check` (`roam_eval.py:434`). Silencing the added tonic must change
behaviour, or C3a is rejected.

### B2 — C3b, excitability re-derived against a non-behavioural target

The machinery is already in the working tree: `dynamics.py:EXCITABLE_PARAMS`, `excitable_arrays`,
`scripts/make_dynamics_bundle.py --prior excitable`. **Keep it; replace the constant.**

`threshold_scale = 0.5` is hand-picked, with no anchor, no falsifier and no test. Adopting it
because behaviour improved would be fitting the brain to the task through the back door — the exact
drift the parent spec's §1 exists to prevent.

**B2.1** Choose the scale so the **resting** network matches a stated, cited baseline-activity
statistic — e.g. the fraction of the population spontaneously active, or a median resting rate —
measured with `scripts/readout_audit.py`'s rest run. Today's baseline for comparison:
`mean_feature_activity = 0.0048`, 24 of 2,022 traces above 0.05.

**B2.2** Treat `noise_sigma` the same way, given the recorded 0.627 → 0.673 result.

**B2.3 Pre-register the falsifier before the value is chosen**, and write the target statistic and
its citation into the manifest marker. **The target is a resting statistic; a behavioural score must
never appear in the derivation.** If no defensible published baseline exists, say so and drop B2
rather than keep 0.5.

### B3 — tests

Bundle identity and additivity; canonical bundle byte-unchanged and canonical `bundle_hash`
unchanged after the `MODEL_KEYS` edit; `silence_tonic` restores the pre-C3a resting state; an
assertion that the derived scale traces to the recorded statistic rather than a literal.

---

## 9. The 2×2 gate

Conditions: `{codec v1, codec v2} × {canonical bundle, C3 bundle}`, each at ablation
`none` / `sensory` / `ghost`, plus the `random` and `cue_script` baselines at `none`.

| #   | step                                                        | cost              | gate                                                               |
| --- | ----------------------------------------------------------- | ----------------- | ------------------------------------------------------------------ |
| 1   | `scripts/command_audit.py` on all four cells                | free, offline     | v2 must lift non-loom stimuli above 0.05 m/s, or stop here         |
| 2   | **ask** → 15-seed smoke across the 2×2                      | ~2 h at 6 workers | direction only, non-pre-registered                                 |
| 3   | **ask** → full pre-registered gate on the winning cell only | ~3 h              | `liveness-check` (L1–L8) and `adapter-check` (A1–A7 **untouched**) |
| 4   | adopt or record the negative                                | —                 | causal pass only                                                   |

**Primary risk, and the guard that already exists.** Tonic drive plus noise could make the drone
wander convincingly without seeing anything — liveness by luck. L3 (silencing sensors must provably
reduce motion), L4 (ghost / random / cue_script must each fail the mobility test) and L5
(`mean_abs_yaw_bias <= 0.25`, no loss-of-control) are exactly the guards for this, and they are
already implemented and pre-registered. **That these gates exist is why this lever is safe to pull.**
If a bundle passes L1/L2/L6–L8 but fails L3 or L4, it is **rejected and recorded as a negative**.
No exceptions, no threshold edits, no "but it looks alive in the viewer".

**Falsifiers, pre-registered.**

- Codec v2 on the canonical bundle does not improve motion structure → the brain state is the
  binding constraint; say so.
- The C3 bundle under codec v1 does not improve it → ongoing state is not the constraint.
- Neither cell improves it → the declared-adapter route has a deeper limit. That is a real finding
  about connectome/model sufficiency. **Report it; do not retreat to a teacher.**

---

## 10. Execution order

Each step ends with tests, lint, commit and push (AGENTS.md). Steps 1–6 involve **no long run**.

1. **A0** — fly statistics into `references.md` and `FLY_REFERENCE`. _Blocking for A1.5._
2. **A1** — `motion_stats.py`, per-frame recording, `summarise`, L6–L8, tests.
3. **A2** — `command_audit.py`; run it on v1 and record the baseline table.
4. **A3** — codec v2 behind `--adapter v2`; calibrate; run `command_audit.py` again.
   **Decision point:** if v2 does not clear 0.05 m/s offline, stop and report.
5. **B1** — C3a tonic bundle, `silence_tonic`, `tonic-check`, tests. _Needs user sign-off._
6. **B2** — C3b excitability re-derived, or dropped with a reason recorded.
7. **Ask** → the 15-seed smoke (§9 step 2).
8. **Ask** → the full gate on the winner (§9 step 3).
9. FINDING under `docs/results/liveness/`, spec status updated, whichever way it goes.

**Verify at every step:**

```bash
env -u PYTHONPATH .venv/bin/python -m pytest -q
.venv/bin/ruff format python tests && .venv/bin/ruff check python tests
```

**Offline diagnostics (free, no arena):**

```bash
env -u PYTHONPATH .venv/bin/python scripts/transmission_audit.py
env -u PYTHONPATH .venv/bin/python scripts/readout_audit.py
env -u PYTHONPATH .venv/bin/python scripts/command_audit.py
```

**Gated runs — ask first, every time:**

```bash
env -u PYTHONPATH .venv/bin/fly-drone liveness-check --adapter v2 \
    --episodes 15 --seconds 120 --level 3 --workers 6 --seed-base 2000 \
    --output docs/results/liveness/liveness-check-v2-smoke.json
```

---

## 11. Invariants

- `roam_eval.ACCEPTANCE`, `CONDITIONS` and A1–A7 are never touched.
- The canonical `data/malecns` bundle and the accepted actors stay byte-identical;
  `test_legacy_room_mjcf_unchanged` still passes.
- Committed `adapter.json` / `adapter-check.json` stay byte-identical.
- No teacher in the behaviour path; the teacher is at most an unweighted context row.
- One brain, one bridge; no scenario switching, no per-condition codec.
- Liveness is generated in the brain and only read by the bridge (§1).
- ≤ 6 workers. Ask before every multi-seed run.

## 12. Corrections this spec makes to the record

1. **P2 never tested its own hypothesis.** The "connectome-constrained dynamics" gate varied only
   `noise_sigma`; leak and threshold were numerically identical to canonical. It is recorded as a
   negative, which risks retiring a lever that was never pulled. Corrected in the parent spec.
2. **The s2/s3 sign results are not evidence about the sign hypothesis** where histamine is
   concerned — see §4.2.
3. **The `threshold_scale = 0.5` in the working tree is a tuning knob, not a prior** — re-derived
   in B2.
4. **Honest caveat on the one working skill.** `escape` is 2 giant-fibre cells reached by a declared
   308-cell / 11,220-synapse direct looming→escape shortcut, and the manifest states "escape is a
   modeled body impulse". The behaviour is causal and real; the claim should carry that caveat.

## 13. Artifacts

`motion_stats.py`; `command_audit.py`; `transmission_audit.py` (landed); `adapter-v2.json`; the
C3a/C3b bundles and their gates; `tests/test_motion_stats.py`, `tests/test_adapter_v2.py`;
`references.md` additions; a FINDING per gate under `docs/results/liveness/`. Patches to the parent
spec, `plans/2026-09-19-fly-drone-10-liveness-relay.md`, `overview/README.md` §1 and §10, and
`neuron-model.md`.
