#!/usr/bin/env bash
# Task 12 (Stage 2a) driver: waits for encoder-collect, then clone, round-0 decoder, sanity gate, validation, v4 E1.
set -euo pipefail
cd /home/gb/projects/fly-drone
FD="env -u PYTHONPATH .venv/bin/fly-drone"
log() { echo "[$(date '+%F %T')] $*"; }

COLLECT_PID=${1:?collect pid}
while kill -0 "$COLLECT_PID" 2>/dev/null; do sleep 15; done
test -f runs/v5/clone/data.npz || { log "FAIL: data.npz missing"; exit 1; }
log "collect done"

$FD encoder-clone runs/v5/clone/data.npz --output runs/v5/clone --steps 20000
log "clone done"

$FD sac-init-decoder runs/v5/dagger/it*.npz --encoder runs/v5/clone/encoder.pt --output runs/v5/round0
log "round0 decoder done"

$FD roam-screen runs/v5/round0/decoder.json --encoder runs/v5/clone/encoder.pt --level 2 --seeds 10 \
    --seed-base 9000 --workers 6 --output runs/v5/round0/l2-screen.json
bpm=$(.venv/bin/python -c "import json;r=json.load(open('runs/v5/round0/l2-screen.json'))['results'];print(next(v['beacons_per_min'] for k,v in r.items() if k.startswith('policy:')))")
log "sanity screen beacons/min=$bpm (gate 1.76)"
if .venv/bin/python -c "import sys;sys.exit(0 if $bpm >= 0.8*2.2 else 1)"; then
    log "GATE PASS"
else
    log "GATE FAIL: stopping before validation"; exit 2
fi

$FD sac-validate --decoder runs/v5/round0/decoder.json --encoder runs/v5/clone/encoder.pt \
    --output runs/v5/round0/validation.json
log "validation done"

$FD encoder-checks --policy runs/v5/dagger/it0/warm-actor.json --output runs/v5/e1-v4-baseline.json
log "stage 2a done"
