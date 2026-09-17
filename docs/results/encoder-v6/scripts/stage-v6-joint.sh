#!/usr/bin/env bash
# M4 (plan 07): one gated joint encoder+decoder SAC round, then the gates.
# Joint = both heads move in one loop; no frozen partner. Warm-started from the round-0 pair:
#   --init    runs/v6/clone/encoder.pt      (encoder head + anchor reference)
#   --decoder runs/v6/round0/decoder.json   (velocity head + dn stats)
# Gates (all must hold): G1 guard, G2 union E1 > round-0 union, G3 E2, G4 bypass not better.
# Any failure -> revert to runs/v6/round0 and record it. Ask the user before the long run.
set -euo pipefail
cd /home/gb/projects/fly-drone
FD="env -u PYTHONPATH .venv/bin/fly-drone"
PY="env -u PYTHONPATH .venv/bin/python"
log() { echo "[$(date '+%F %T')] $*"; }

FRAMES=${FRAMES:-300000}
OUT=${OUT:-runs/v6/joint/round1}
CLONE=runs/v6/clone/encoder.pt
ROUND0_DEC=runs/v6/round0/decoder.json
ROUND0_VAL=runs/v6/round0/validation.json

if [ ! -f "$CLONE" ] || [ ! -f "$ROUND0_DEC" ] || [ ! -f "$ROUND0_VAL" ]; then
    log "missing round-0 artifacts; cannot warm-start the joint round"; exit 1
fi

log "M4 joint SAC ($FRAMES frames)"
$FD sac-round joint --output "$OUT" --frames "$FRAMES" \
    --init "$CLONE" --decoder "$ROUND0_DEC" --workers 6 > runs/v6/joint-round1.log 2>&1

log "M4 joint validate"
$FD sac-validate --decoder "$OUT/decoder.json" --encoder "$OUT/encoder.pt" \
    --output "$OUT/validation.json"

log "M4 G1 guard + G2/G3 (validation) + liveness"
$PY scripts/encoder_liveness.py "$OUT/encoder.pt"
$PY -c "
import json
from fly_drone.roam_eval import pick_best_round, round_eligible, round_gate_report
vals = [json.load(open('$ROUND0_VAL')), json.load(open('$OUT/validation.json'))]
for r in round_gate_report(vals): print('M4 table', r)
print('M4 eligible', int(round_eligible(vals[1], vals[0])), 'best', pick_best_round(vals))"

log "M4 fixed-probe E1/E2 (union reported)"
$FD encoder-checks --policy "$OUT/decoder.json" --encoder "$OUT/encoder.pt" \
    --output "$OUT/e1e2-fixed.json" --episodes 50 --seconds 120 --workers 6 --controller teacher
$PY -c "
import json
r = json.load(open('$OUT/e1e2-fixed.json'))
e1 = r['E1']
print('M4 E1 loom', e1['loom_auc'], 'motion', e1.get('motion_auc'), 'union', e1.get('union_auc'), '(bar 0.8)')
print('M4 E2 pass', r['E2']['passed'], 'margins', r['E2']['light_margin'], r['E2']['loom_margin'])"

log "M4 E1/E2 done (G2/G3) at $OUT/e1e2-fixed.json"

# G4 (E3 brain-bypass) needs v6 bypass support, which does not exist yet: the bypass learner reads
# the 8 v5 currents and `distill.screen` rejects a v6 encoder. Tracked as plan-07 Task 6b; run it
# before accepting the pair. Until then, do not claim E3.
log "M4 NOTE: G4 (E3 bypass) not run - v6 bypass support missing (plan 07 Task 6b)"
log "M4 done; report G1-G3 and compare against runs/v6/round0"
