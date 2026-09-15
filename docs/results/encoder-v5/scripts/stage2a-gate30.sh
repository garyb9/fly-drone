#!/usr/bin/env bash
# Task 12 gate re-screen on 30 seeds (user decision 2026-09-15): same 0.8x ratio rule, larger sample for BOTH systems.
set -euo pipefail
cd /home/gb/projects/fly-drone
FD="env -u PYTHONPATH .venv/bin/fly-drone"
PY="env -u PYTHONPATH .venv/bin/python"
log() { echo "[$(date '+%F %T')] $*"; }
bpm() { $PY -c "import json,sys;r=json.load(open(sys.argv[1]))['results'];print(next(v['beacons_per_min'] for k,v in r.items() if k.startswith('policy:')))" "$1"; }

$FD roam-screen runs/v5/dagger/it0/warm-actor.json --level 2 --seeds 30 --seed-base 9000 --workers 6 \
    --output runs/v5/round0/gate30-v4-it0.json
ref=$(bpm runs/v5/round0/gate30-v4-it0.json)
log "gate30 v4 it0 beacons/min=$ref"

$FD roam-screen runs/v5/round0/decoder.json --encoder runs/v5/clone/encoder.pt --level 2 --seeds 30 --seed-base 9000 \
    --workers 6 --output runs/v5/round0/gate30-round0.json
got=$(bpm runs/v5/round0/gate30-round0.json)
log "gate30 round0 beacons/min=$got (gate 0.8 x $ref)"
if $PY -c "import sys;sys.exit(0 if $got >= 0.8*$ref else 1)"; then
    log "GATE PASS"
else
    log "GATE FAIL: stopping before validation"; exit 2
fi

$FD sac-validate --decoder runs/v5/round0/decoder.json --encoder runs/v5/clone/encoder.pt \
    --output runs/v5/round0/validation.json
log "validation done"

$FD encoder-checks --policy runs/v5/dagger/it0/warm-actor.json --output runs/v5/e1-v4-baseline.json
log "stage 2a done"
