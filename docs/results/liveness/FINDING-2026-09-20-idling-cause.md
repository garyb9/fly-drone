# Why the connectome-driven body idles — a measured cause

**Status:** recorded 2026-09-20. Offline analysis only: no arena, no seeds, no long run. Every
number below comes from committed artifacts or from a direct read of the canonical bundle, and each
is reproducible with the commands at the end.

The preceding finding ([`FINDING-2026-09-20.md`](FINDING-2026-09-20.md)) concluded that "the next
lever is control/excitability, not more sensory channels". This one carries that to the arithmetic.
**The idling is not a mystery.** It has three measured causes, two of which were never suspected.

## 1. The codec cannot command a speed that counts as moving

`Adapter.command` (`python/fly_drone/adapter.py:242-262`) sets `vx = g_fwd · max(0, power − rest_power)`.
With the committed `g_fwd = 0.69203`, `rest_power = 0.439525` (`../adapter/adapter.json`) and
`plant.LIMITS[0] = 0.4 m/s`, against the 0.05 m/s stuck threshold (`distill.py:384`, `env.py:677`),
crossed with the measured `power` deltas in `readout-audit.json`:

| stimulus           |    Δpower |  commanded vx | ≥ 0.05 m/s? |
| ------------------ | --------: | ------------: | ----------- |
| `light_l`          |     0.036 |     0.010 m/s | no          |
| `light_r`          |     0.074 |     0.021 m/s | no          |
| `light_both`       |     0.109 |     0.030 m/s | no          |
| `loom_l`           |     0.153 |     0.042 m/s | no          |
| **`loom_r`**       | **0.197** | **0.055 m/s** | **yes**     |
| `relay:cruise_yaw` |     0.016 |     0.004 m/s | no          |

**Only a full one-sided loom clears the bar.** A bright light filling both eyes commands 3 cm/s and
is scored stationary. Saturating every flight-power motoneuron would reach 0.155 m/s — 39 % of the
body's range. The observed profile (vivid escape, idle otherwise) is what this table predicts.

Three compounding causes inside the codec:

1. `max(0, ·)` (`adapter.py:256`) discards every excursion below rest: the brain can command faster
   than rest, never slower.
2. The calibration battery fights itself. `STIMULI` (`adapter.py:48-56`) declares `vx = 0` for the
   loom rows, yet loom produces the battery's largest `power` excursion, so the least-squares
   `gain()` (`adapter.py:143-147`) is dragged down by the rows that move the signal most. `g_fwd` is
   small _because_ loom raises power and loom is declared to mean stop.
3. Steering reads a left-minus-right mean over all 2,022 traces (`adapter.py:105-107`), needing
   `g_yaw = 210.7` for a unit turn — while `steer_l`/`steer_r`, derived at `brain.py:180-198` from
   the fly's actual wing steering motoneurons and moving 0.05–0.20 in the audit, are never read.

## 2. Exactly one neuron population has ongoing activity

`brain.py:202-203` biases the DLMn/DVMn flight-power motoneurons at `b = 0.85`. The other **166,698
neurons sit at `bias = 0` under `v_threshold = 1.0`**, needing sustained drive above 0.221 to fire
(`leak ≈ 0.7788`). At rest `mean_feature_activity = 0.0048`; 24 of 2,022 traces exceed 0.05. A brain
with no ongoing activity can only be reactive.

The steering motoneurons are the sharpest instance: in a flying fly they fire about once per
wingbeat continuously and steering modulates that ongoing train, but here they start from silence,
so `steer_l − steer_r` has no dynamic range. The tonic-bias hypothesis was applied to the power
muscles and not to the steering muscles.

Related: `refrac_ticks = round(2.0 / 5.0) = 0` (`crates/brain-core/src/core/lif.rs:40-44`) — there
is no refractory period, so cells are silent or fire every tick, with no graded rate code between.

## 3. 9,189 neurons are wired in and transmit nothing

Read directly from the canonical `graph.bin` CSR (`scripts/transmission_audit.py`,
`transmission-audit.json`):

| transmitter   |   cells | with out-edges | silent out | mean Σ\|w\| |
| ------------- | ------: | -------------: | ---------: | ----------: |
| acetylcholine | 103,720 |        103,310 |          0 |       603.0 |
| glutamate     |  29,302 |         29,085 |          0 |       589.7 |
| gaba          |  22,069 |         22,045 |          0 |       996.7 |
| histamine     |   7,891 |          6,179 |  **6,179** |         0.0 |
| unclear       |   2,999 |          2,489 |  **2,489** |         0.0 |
| dopamine      |     392 |            391 |    **391** |         0.0 |
| octopamine    |     101 |             82 |     **82** |         0.0 |
| serotonin     |      48 |             48 |     **48** |         0.0 |

**9,189 neurons (5.5 %) and 332,316 edges (3.2 %) carry no current.** This is the declared
hypothesis in `manifest.json` ("unresolved/modulatory source current zero"), quantified for the
first time. Three consequences:

1. **The histaminergic photoreceptor→lamina relay is electrically dead.** The P3 optic-flow relay
   injects into T4/T5 _past_ that layer rather than through it — correct for a declared front-end,
   but it means the relay's null result says nothing about the real relay.
2. **The s2/s3 sign results are not evidence about the sign hypothesis for histamine.**
   `make_sign_bundle.py` retags only the inhibitory bit in `neurons.bin`; `graph.bin` is
   byte-identical between canonical and `malecns-sign-s2` (verified with `cmp`). Retagging a
   zero-weight neuron's sign is a no-op, so s2 could not have acted through histamine. "S2 is
   indistinguishable on the probe" (`external-prior-art.md:196`) has a mechanical explanation.
3. **Neuromodulatory tone cannot be switched on in this repo.** Biasing an octopaminergic cell makes
   it spike into zero-weight edges. Restoring modulatory transmission means rebuilding `graph.bin`
   upstream in fly-playground — deferred to a separate plan, recorded so it is not forgotten.

## What this changes

The next work is specified in
[`../../superpowers/specs/2026-09-20-ongoing-state-and-faithful-readout.md`](../../superpowers/specs/2026-09-20-ongoing-state-and-faithful-readout.md):
a faithful codec (workstream A, bridge only) and ongoing state as contract change C3 (workstream B),
gated 2×2 against the existing causal controls. `ACCEPTANCE` A1–A7 are untouched. The liveness bar's
L1/L2 are teacher-anchored and blind to motion structure, and are being re-anchored on published fly
free-flight statistics.

**No behaviour claim is made here.** This finding is arithmetic and data inspection; whether fixing
either cause makes the drone livelier is the pre-registered question the 2×2 gate answers.

## Reproduce

```bash
env -u PYTHONPATH .venv/bin/python scripts/transmission_audit.py
env -u PYTHONPATH .venv/bin/python scripts/readout_audit.py
cmp data/malecns/graph.bin data/malecns-sign-s2/graph.bin   # identical
```

The §1 table is `readout-audit.json` crossed with `../adapter/adapter.json`; `scripts/command_audit.py`
(workstream A2) will emit it directly.
