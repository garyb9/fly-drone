#!/usr/bin/env bash
# Task 10 (v6 plan): spatial clone -> teacher flights under the clone -> round-0 decoder -> 30-seed gate.
set -euo pipefail
cd /home/gb/projects/fly-drone
FD="env -u PYTHONPATH .venv/bin/fly-drone"
PY="env -u PYTHONPATH .venv/bin/python"
log() { echo "[$(date '+%F %T')] $*"; }
bpm() { $PY -c "import json,sys;r=json.load(open(sys.argv[1]))['results'];print(next(v['beacons_per_min'] for k,v in r.items() if k.startswith('policy:')))" "$1"; }

log "clone fit start"
$FD encoder-clone runs/v5/clone/data.npz runs/v5/clone/data2.npz \
    --output runs/v6/clone --spatial

log "teacher collection under the clone start"
$FD roam-collect --output runs/v6/round0-data/it0.npz --flights 128 --seconds 60 \
    --levels 2 --seed-base 200 --workers 6 --encoder runs/v6/clone/encoder.pt

log "round-0 decoder fit"
$FD sac-init-decoder runs/v6/round0-data/it0.npz \
    --encoder runs/v6/clone/encoder.pt --output runs/v6/round0 --steps 4000

log "gate30 screen (level 2, seeds 9000-9029)"
$FD roam-screen runs/v6/round0/decoder.json --encoder runs/v6/clone/encoder.pt \
    --level 2 --seeds 30 --seed-base 9000 --workers 6 \
    --output runs/v6/round0/gate30-round0.json
got=$(bpm runs/v6/round0/gate30-round0.json)
ref=$(bpm runs/v5/round0/gate30-v4-it0.json)
log "gate30: round0=$got v4_ref=$ref gate=0.8x=$($PY -c "print(0.8*$ref)")"
if $PY -c "import sys;sys.exit(0 if $got >= 0.8*$ref else 1)"; then
    log "GATE PASS"
else
    log "GATE FAIL: stopping before validation"; exit 2
fi
log "Task 10 done"
