#!/usr/bin/env bash
# Task 12 re-run #3 after Task 12c: difference-aware clone (60k steps), round-0 data under it, refit, gate.
set -euo pipefail
cd /home/gb/projects/fly-drone
FD="env -u PYTHONPATH .venv/bin/fly-drone"
PY="env -u PYTHONPATH .venv/bin/python"
log() { echo "[$(date '+%F %T')] $*"; }

# Keep the record of both failed gates.
if [ ! -d runs/v5/clone-v1 ]; then
    mkdir -p runs/v5/clone-v1
    mv runs/v5/clone/encoder.pt runs/v5/clone/clone.json runs/v5/clone-v1/
fi
[ -d runs/v5/round0 ] && [ ! -d runs/v5/round0-v2-clonedata ] && mv runs/v5/round0 runs/v5/round0-v2-clonedata
[ -d runs/v5/round0-data ] && [ ! -d runs/v5/round0-data-v1 ] && mv runs/v5/round0-data runs/v5/round0-data-v1

$FD encoder-clone runs/v5/clone/data.npz --output runs/v5/clone --steps 60000
log "clone done: $($PY -c "import json;c=json.load(open('runs/v5/clone/clone.json'));print(c['version'], {k:round(v['r'],3) for k,v in c['held_out_differences'].items() if v['r'] is not None})")"

$FD roam-collect --output runs/v5/round0-data/it0.npz --flights 128 --seconds 60 --levels 2 --workers 6 \
    --seed-base 200 --encoder runs/v5/clone/encoder.pt
log "clone-driven collect done"

$FD sac-init-decoder runs/v5/round0-data/it0.npz --encoder runs/v5/clone/encoder.pt --output runs/v5/round0
log "round0 decoder done"

$FD roam-screen runs/v5/round0/decoder.json --encoder runs/v5/clone/encoder.pt --level 2 --seeds 10 \
    --seed-base 9000 --workers 6 --output runs/v5/round0/l2-screen.json
bpm=$($PY -c "import json;r=json.load(open('runs/v5/round0/l2-screen.json'))['results'];print(next(v['beacons_per_min'] for k,v in r.items() if k.startswith('policy:')))")
log "sanity screen beacons/min=$bpm (gate 1.76)"
if $PY -c "import sys;sys.exit(0 if $bpm >= 0.8*2.2 else 1)"; then
    log "GATE PASS"
else
    log "GATE FAIL: stopping before validation"; exit 2
fi

$FD sac-validate --decoder runs/v5/round0/decoder.json --encoder runs/v5/clone/encoder.pt \
    --output runs/v5/round0/validation.json
log "validation done"

$FD encoder-checks --policy runs/v5/dagger/it0/warm-actor.json --output runs/v5/e1-v4-baseline.json
log "stage 2a done"
