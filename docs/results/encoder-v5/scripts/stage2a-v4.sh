#!/usr/bin/env bash
# Task 12 re-run #4 after Task 12d: mirror-symmetric clone on 64 flights (60k steps), round-0 data under it,
# refit, 30-seed gate against the measured v4 reference (runs/v5/round0-v3/gate30-v4-it0.json).
set -euo pipefail
cd /home/gb/projects/fly-drone
FD="env -u PYTHONPATH .venv/bin/fly-drone"
PY="env -u PYTHONPATH .venv/bin/python"
log() { echo "[$(date '+%F %T')] $*"; }
bpm() { $PY -c "import json,sys;r=json.load(open(sys.argv[1]))['results'];print(next(v['beacons_per_min'] for k,v in r.items() if k.startswith('policy:')))" "$1"; }

test -f runs/v5/clone/data2.npz || { log "FAIL: data2.npz missing"; exit 1; }

# Keep the record of the difference-aware attempt.
if [ ! -d runs/v5/clone-v2-diffaware ]; then
    mkdir -p runs/v5/clone-v2-diffaware
    mv runs/v5/clone/encoder.pt runs/v5/clone/clone.json runs/v5/clone-v2-diffaware/
fi
[ -d runs/v5/round0 ] && [ ! -d runs/v5/round0-v3 ] && mv runs/v5/round0 runs/v5/round0-v3
[ -d runs/v5/round0-data ] && [ ! -d runs/v5/round0-data-v2 ] && mv runs/v5/round0-data runs/v5/round0-data-v2
REF_JSON=runs/v5/round0-v3/gate30-v4-it0.json
ref=$(bpm $REF_JSON)
log "v4 30-seed reference beacons/min=$ref ($REF_JSON)"

$FD encoder-clone runs/v5/clone/data.npz runs/v5/clone/data2.npz --output runs/v5/clone --steps 60000
log "clone done: $($PY -c "import json;c=json.load(open('runs/v5/clone/clone.json'));print(c['version'], 'diff r', {k:round(v['r'],3) for k,v in c['held_out_differences'].items() if v['r'] is not None}, 'mirror r', {k:(round(v['r'],3) if isinstance(v,dict) and v.get('r') is not None else v) for k,v in c.get('held_out_mirror',{}).items()})")"

$FD roam-collect --output runs/v5/round0-data/it0.npz --flights 128 --seconds 60 --levels 2 --workers 6 \
    --seed-base 200 --encoder runs/v5/clone/encoder.pt
log "clone-driven collect done"

$FD sac-init-decoder runs/v5/round0-data/it0.npz --encoder runs/v5/clone/encoder.pt --output runs/v5/round0
log "round0 decoder done"

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
