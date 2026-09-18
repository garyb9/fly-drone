#!/usr/bin/env bash
# Threat-focused DAgger iteration (plan 08 Task 3 follow-up).
#
# DAgger 0-3 left the threat drive at ~1% of frames, so no causal dodge emerged
# (docs/results/encoder-v6/DAGGER-2026-09-18.md). This collects level 3 only, from the
# it2 student, so the threat drive is well represented, then refits on the foraging data
# plus the threat set and screens at level 3.
#
# Run:  setsid nohup env -u PYTHONPATH docs/results/encoder-v6/scripts/dagger-v6-threat.sh \
#         > runs/roam/dagger-v6-threat-driver.log 2>&1 &
set -uo pipefail

ENC=runs/v6/clone/encoder.pt
OUT=runs/roam
W=6

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "collect 128 L3 flights from it2 (beta 0.25)"
if ! env -u PYTHONPATH .venv/bin/fly-drone roam-collect \
      --output "$OUT/v6-d3t.npz" --flights 128 --seconds 60 --levels 3 \
      --student "$OUT/v6-it2/warm-actor.json" --beta 0.25 \
      --encoder "$ENC" --workers "$W" --seed-base 8000 >> "$OUT/dagger-v6-threat.log" 2>&1; then
  log "collect FAILED"; exit 1
fi

log "refit on d0 d1 d2 d3t"
if ! env -u PYTHONPATH .venv/bin/fly-drone roam-fit \
      "$OUT/v6-d0.npz" "$OUT/v6-d1.npz" "$OUT/v6-d2.npz" "$OUT/v6-d3t.npz" \
      --encoder "$ENC" --output "$OUT/v6-it3t" >> "$OUT/dagger-v6-threat.log" 2>&1; then
  log "fit FAILED"; exit 1
fi

log "screen at L3 (none+ghost)"
if ! env -u PYTHONPATH .venv/bin/fly-drone roam-screen \
      "$OUT/v6-it3t/warm-actor.json" teacher random \
      --seeds 10 --seconds 60 --level 3 --ablations none ghost \
      --encoder "$ENC" --workers "$W" \
      --output "$OUT/v6-it3t-screen.json" >> "$OUT/dagger-v6-threat.log" 2>&1; then
  log "screen FAILED"; exit 1
fi

log "threat-focused DAgger iteration complete"
