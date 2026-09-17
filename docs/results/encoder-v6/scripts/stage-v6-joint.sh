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

log "M4 G4 E3 bypass (decoder reading the same 816 currents, brain bypassed; ${FRAMES} frames)"
$FD sac-round bypass --output "${OUT}-bypass" --frames "$FRAMES" --encoder "$OUT/encoder.pt" \
    --workers 6 > runs/v6/joint-bypass.log 2>&1
$FD roam-screen "$OUT/decoder.json" "${OUT}-bypass/bypass.zip" --encoder "$OUT/encoder.pt" \
    --ablations none ghost --seeds 50 --seconds 120 --level 3 --seed-base 1000 --workers 6 \
    --output "$OUT/e3-screen.json"
$PY -c "
import json
from pathlib import Path
from fly_drone.roam_eval import bypass_comparison
r = json.loads(Path('$OUT/e3-screen.json').read_text())['results']
key = lambda pre: next(v for k, v in r.items() if k.startswith(pre) and k.endswith('|none'))
out = bypass_comparison(key('policy:'), key('bypass:'))
Path('$OUT/e3.json').write_text(json.dumps(out, indent=2))
print('M4 bypass_better', out['bypass_better'])"

log "M4 done; report G1-G4 and compare against runs/v6/round0"
