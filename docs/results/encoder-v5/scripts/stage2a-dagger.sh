#!/usr/bin/env bash
# Task 12 after the clone strategy was exhausted (user decision 2026-09-15): DAgger under the frozen mirror clone,
# 30-seed gate after each iteration (max 2). Pass -> best round 0; no pass -> best round 0 with a recorded override.
# Either way: install runs/v5/round0, validate, v4 E1 baseline, "stage 2a done" (smoke13 chains on it).
set -euo pipefail
cd /home/gb/projects/fly-drone
FD="env -u PYTHONPATH .venv/bin/fly-drone"
PY="env -u PYTHONPATH .venv/bin/python"
ENC=runs/v5/clone/encoder.pt
REF_JSON=runs/v5/round0-v3/gate30-v4-it0.json
log() { echo "[$(date '+%F %T')] $*"; }
stat30() { $PY -c "import json,sys;r=json.load(open(sys.argv[1]))['results'];v=next(v for k,v in r.items() if k.startswith('policy:'));print(v['beacons_per_min'], v['collisions_per_min'])" "$1"; }

# Archive the teacher-only mirror-clone round 0 (already screened on 30 seeds: 1.2).
if [ -d runs/v5/round0 ] && [ ! -d runs/v5/round0-mirror-it0 ]; then mv runs/v5/round0 runs/v5/round0-mirror-it0; fi
ref=$(stat30 $REF_JSON | cut -d' ' -f1)
gate=$($PY -c "print(0.8*$ref)")
log "v4 30-seed reference $ref -> gate $gate"

CANDS="runs/v5/round0-mirror-it0"
DATA="runs/v5/round0-data/it0.npz"
prev=runs/v5/round0-mirror-it0
passed=0
for k in 1 2; do
    beta=$([ "$k" -eq 1 ] && echo 0.5 || echo 0.25)
    $FD roam-collect --output runs/v5/round0-data/it$k.npz --student $prev/decoder.json --beta $beta --flights 128 \
        --seconds 60 --levels 2 --workers 6 --seed-base $((200 + 1000 * k)) --encoder $ENC
    DATA="$DATA runs/v5/round0-data/it$k.npz"
    log "dagger it$k collect done (beta $beta)"
    out=runs/v5/round0-dagger$k
    $FD sac-init-decoder $DATA --encoder $ENC --output $out
    log "dagger it$k refit done: $($PY -c "import json;w=json.load(open('$out/warm-start.json'));print('export', w['export_max_error'], {k:v.get('held_out_mse') for k,v in w['drives'].items()})")"
    $FD roam-screen $out/decoder.json --encoder $ENC --level 2 --seeds 30 --seed-base 9000 --workers 6 \
        --output $out/gate30-round0.json
    read bpm cpm < <(stat30 $out/gate30-round0.json)
    log "dagger it$k gate30 beacons/min=$bpm collisions/min=$cpm (gate $gate)"
    CANDS="$CANDS $out"
    prev=$out
    if $PY -c "import sys;sys.exit(0 if $bpm >= $gate else 1)"; then
        passed=1
        log "GATE PASS at dagger it$k"
        break
    fi
    log "GATE FAIL at dagger it$k"
done

best=$($PY - $gate $passed $CANDS <<'EOF'
import json, sys
gate, passed, cands = float(sys.argv[1]), bool(int(sys.argv[2])), sys.argv[3:]
rows = []
for c in cands:
    r = json.load(open(f"{c}/gate30-round0.json"))["results"]
    v = next(v for k, v in r.items() if k.startswith("policy:"))
    rows.append({"dir": c, "beacons_per_min": v["beacons_per_min"], "collisions_per_min": v["collisions_per_min"]})
best = max(rows, key=lambda x: (x["beacons_per_min"], -x["collisions_per_min"]))
json.dump({"gate": gate, "passed": best["beacons_per_min"] >= gate, "override": best["beacons_per_min"] < gate,
           "best": best["dir"], "candidates": rows}, open("runs/v5/gate-choice.json", "w"), indent=2)
print(best["dir"])
EOF
)
cp -r "$best" runs/v5/round0
cp runs/v5/gate-choice.json runs/v5/round0/gate.json
log "installed round0 from $best ($(cat runs/v5/round0/gate.json | tr -d '\n' | cut -c1-160))"

$FD sac-validate --decoder runs/v5/round0/decoder.json --encoder $ENC --output runs/v5/round0/validation.json
log "validation done"

$FD encoder-checks --policy runs/v5/dagger/it0/warm-actor.json --output runs/v5/e1-v4-baseline.json
log "stage 2a done"
