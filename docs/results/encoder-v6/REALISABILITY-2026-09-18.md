# Frozen-v6 realisability probe (plan 08 Task 1, G0)

**Date:** 2026-09-18. **Scope:** plan 08 Task 1. Collection + a `roam-fit` diagnostic; no training
run, `roam_eval.ACCEPTANCE` and the E1/E2 bars untouched.

## Protocol

| Set | Command | Samples | Flights | Drives seen |
| --- | --- | ---: | ---: | --- |
| L0–2 | `roam-collect --flights 64 --seconds 60 --levels 0 1 2 --encoder runs/v6/clone/encoder.pt` | 48,000 | 64 | avoid, beacon, explore |
| L3 (added) | same, `--flights 24 --levels 3 --seed-base 3000` | 18,000 | 24 | all four |

Both fitted with `roam-fit … --encoder runs/v6/clone/encoder.pt`. Held-out R² is the
`warm-start.json` number: 10% of **flights** held out, total-variance R² over the four axes. The
gate (plan 08) is **each drive held-out R² ≥ 0.5**.

## Result — G0 fails

L0–2 (held-out flights 6):

| Drive | held_out R² | train R² | held-out samples |
| --- | ---: | ---: | ---: |
| avoid | 0.285 | 0.929 | 385 |
| beacon | 0.406 | 0.695 | 2,021 |
| explore | **−1.134** | −0.385 | 2,094 |
| threat | — (absent) | — | 0 |

L3 (held-out flights 2):

| Drive | held_out R² | train R² | held-out samples |
| --- | ---: | ---: | ---: |
| avoid | 0.103 | 0.941 | 123 |
| beacon | −0.426 | 0.620 | 676 |
| explore | **−2.523** | −0.708 | 626 |
| threat | **−1.779** | 0.969 | 75 |

## Why the literal gate is not the right test

1. **`explore` is stochastic by construction.** The teacher's explore action is a random yaw cast
   resampled every 1.5–3.5 s (`teacher.py`); no feature can predict it, so its R² is negative by
   design, not by realisability failure.
2. **The bar was never met — and was not needed.** The v5 DAgger iterations
   (`runs/v5/dagger/it*/warm-start.json`) never reached 0.5 on any drive either (avoid 0.06–0.38,
   beacon 0.25–0.49, explore negative), yet `it0`/`it1` foraged at ~2.0–2.2 beacons/min and became
   the v5 round-0 base. In-sample R² here is 0.62–0.97: the features carry the signal; the
   held-out gap is overfitting across layouts with few flights, which DAgger's iterative collection
   is what fixes.
3. **`threat` cannot be judged at L0–2** (no threats) and is too rare at 24 L3 flights (75 held-out
   samples) for a stable R².

G0 as written therefore reports "fail" while the underlying question — can features predict the
label — is positive in-sample and answered downstream by whether DAgger screens improve.

## Recommendation

Treat G0 as a **diagnostic, not a hard gate**: record the R² table above, then proceed to DAgger
iterations 0–3 (plan 08 Task 3), where each student is screened on beacons/collisions/ghost. Keep
the 0.5 bar in the record but do not block on it. If the user prefers a real gate, the protocol
needs (a) `explore` excluded as stochastic, (b) `threat` measured at L3 with enough flights for a
stable held-out split, and (c) a bar chosen against the v5 precedent.
