#!/usr/bin/env bash
# Decoder-only SAC on the frozen v6 pair with the new dense threat shaping, retrain after the
# sac1 collapse (docs/results/encoder-v6/SAC-ROUND1-2026-09-18.md). Starts from the round-eligible
# forager it2 (beacons 1.6/min) and anchors to it, so the shaping teaches threat avoidance without
# the policy abandoning foraging. Encoder/connectome frozen; only the decoder learns.
#
# Run:  setsid nohup env -u PYTHONPATH docs/results/encoder-v6/scripts/sac-decoder-v6-shaped.sh \
#         > runs/roam/sac-decoder-v6-shaped-driver.log 2>&1 &
set -uo pipefail

ENC=runs/v6/clone/encoder.pt
OUT=runs/roam/v6-sac2
ANCHOR=runs/roam/v6-it2/warm-actor.json
LOG=runs/roam/sac-decoder-v6-shaped.log

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "init decoder (SAC spaces) on d0 d1 d2 (the it2 data)"
if ! env -u PYTHONPATH .venv/bin/fly-drone sac-init-decoder \
      runs/roam/v6-d0.npz runs/roam/v6-d1.npz runs/roam/v6-d2.npz \
      --encoder "$ENC" --output "$OUT/init" >> "$LOG" 2>&1; then
  log "init FAILED"; exit 1
fi

log "decoder SAC 150k, level 3, shaped reward, anchor=it2"
if ! env -u PYTHONPATH .venv/bin/fly-drone sac-round decoder --encoder "$ENC" \
      --init "$OUT/init/decoder.zip" --anchor "$ANCHOR" \
      --frames 150000 --workers 6 --level 3 --output "$OUT" >> "$LOG" 2>&1; then
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
v = json.load(open("runs/roam/v6-sac2/validation.json"))
b = json.load(open("runs/v6/round0/validation.json"))
print("eligible", round_eligible(v, b))
print("beacons", v.get("beacons_per_min"), "ghost", v.get("ghost_near_dodge_rate"),
      "E2", (v.get("E2") or {}).get("passed"))
PY
log "done"
