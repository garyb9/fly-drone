#!/usr/bin/env bash
# Decoder-only SAC on the frozen v6 pair, anchored to the threat-trained DAgger student (it3t).
# Plan 08 Task 4. The encoder and connectome are frozen; only the 4-axis decoder learns.
#
# Run:  setsid nohup env -u PYTHONPATH docs/results/encoder-v6/scripts/sac-decoder-v6.sh \
#         > runs/roam/sac-decoder-v6-driver.log 2>&1 &
set -uo pipefail

ENC=runs/v6/clone/encoder.pt
OUT=runs/roam/v6-sac1
LOG=runs/roam/sac-decoder-v6.log

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "init decoder (SAC spaces) on d0 d1 d2 d3t"
if ! env -u PYTHONPATH .venv/bin/fly-drone sac-init-decoder \
      runs/roam/v6-d0.npz runs/roam/v6-d1.npz runs/roam/v6-d2.npz runs/roam/v6-d3t.npz \
      --encoder "$ENC" --output "$OUT/init" >> "$LOG" 2>&1; then
  log "init FAILED"; exit 1
fi

log "decoder SAC 150k, anchor=it3t"
if ! env -u PYTHONPATH .venv/bin/fly-drone sac-round decoder --encoder "$ENC" \
      --init "$OUT/init/decoder.zip" --anchor runs/roam/v6-it3t/warm-actor.json \
      --frames 150000 --workers 6 --output "$OUT" >> "$LOG" 2>&1; then
  log "round FAILED"; exit 1
fi

log "validate (L3, 10 seeds x 60 s, none+ghost)"
if ! env -u PYTHONPATH .venv/bin/fly-drone sac-validate --decoder "$OUT/decoder.json" \
      --encoder "$ENC" --seeds 10 --seconds 60 --workers 6 --output "$OUT/validation.json" \
      >> "$LOG" 2>&1; then
  log "validate FAILED"; exit 1
fi

log "round guard vs round 0"
env -u PYTHONPATH .venv/bin/python - >> "$LOG" 2>&1 <<'PY'
import json
from fly_drone.roam_eval import round_eligible
v = json.load(open("runs/roam/v6-sac1/validation.json"))
b = json.load(open("runs/v6/round0/validation.json"))
print("eligible", round_eligible(v, b))
print("beacons", v.get("beacons_per_min"), "ghost", v.get("ghost_near_dodge_rate"),
      "E2", (v.get("E2") or {}).get("passed"))
PY
log "done"
