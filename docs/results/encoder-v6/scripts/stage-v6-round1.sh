#!/usr/bin/env bash
# v6 Task 11: one alternating encoder+decoder SAC round from the neutral round-0 pair,
# staged (Phase 1.5 schema) with the probe mid-gate and the round selection guard, then the
# decisive fixed-probe E1. Round 0 = neutral-Tm4/T2 clone + round-0 decoder (gate PASS 2.00).
# seed_motion was tried and its gate FAILED (1.50) so it is not used here.
set -euo pipefail
cd /home/gb/projects/fly-drone
FD="env -u PYTHONPATH .venv/bin/fly-drone"
PY="env -u PYTHONPATH .venv/bin/python"
log() { echo "[$(date '+%F %T')] $*"; }

ENC_MID_FRAMES=${ENC_MID_FRAMES:-100000}
ENC_FRAMES=${ENC_FRAMES:-350000}
DEC_FRAMES=${DEC_FRAMES:-150000}
ENC_PROBE_FLOOR=${ENC_PROBE_FLOOR:-0.02}
ROUND0_VAL=runs/v6/round0/validation.json

if [ ! -f "$ROUND0_VAL" ]; then
    log "round-0 validation.json missing; run sac-validate on runs/v6/round0 first"; exit 1
fi

mid_gate() {
    local value
    value=$(grep -oE 'action_state_std[[:space:]]*\|[[:space:]]*[0-9.eE+-]+' "$1" | tail -1 | grep -oE '[0-9.eE+-]+$')
    if [ -z "$value" ]; then
        log "mid gate: no action_state_std in $1; refusing to continue"; return 1
    fi
    $PY -c "import sys; v=float('$value'); print(f'mid action_state_std {v:.3f} (floor $ENC_PROBE_FLOOR)'); sys.exit(0 if v >= $ENC_PROBE_FLOOR else 1)"
}

log "T11 round 1 encoder SAC stage A (to $ENC_MID_FRAMES, keeping the snapshot)"
$FD sac-round encoder --output runs/v6/round1/encoder --frames $ENC_MID_FRAMES \
    --decoder runs/v6/round0/decoder.json --init runs/v6/clone/encoder.pt --spatial \
    --workers 6 --keep-resume > runs/v6/round1-encoder-a.log 2>&1

log "T11 round 1 mid-round probe gate"
if ! mid_gate runs/v6/round1-encoder-a.log; then
    rm -rf runs/v6/round1/encoder/resume
    log "T11 STOP: encoder is blind by $ENC_MID_FRAMES; keeping round 0"; exit 2
fi

log "T11 round 1 encoder SAC stage B (to $ENC_FRAMES)"
$FD sac-round encoder --output runs/v6/round1/encoder --frames $ENC_FRAMES \
    --decoder runs/v6/round0/decoder.json --init runs/v6/clone/encoder.pt --resume --spatial \
    --workers 6 > runs/v6/round1-encoder.log 2>&1

log "T11 round 1 decoder SAC"
$FD sac-round decoder --output runs/v6/round1/decoder --frames $DEC_FRAMES \
    --encoder runs/v6/round1/encoder/encoder.pt --init runs/v6/round0/decoder.zip \
    --workers 6 > runs/v6/round1-decoder.log 2>&1

log "T11 round 1 validate"
$FD sac-validate --decoder runs/v6/round1/decoder/decoder.json \
    --encoder runs/v6/round1/encoder/encoder.pt --output runs/v6/round1/validation.json

log "T11 round 1 selection guard"
$PY -c "
import json
from fly_drone.roam_eval import pick_best_round, round_eligible, round_gate_report
vals = [json.load(open('runs/v6/round0/validation.json')), json.load(open('runs/v6/round1/validation.json'))]
for r in round_gate_report(vals): print('T11 table', r)
print('T11 eligible_round1', int(round_eligible(vals[1], vals[0])), 'best_eligible', pick_best_round(vals))"

log "T11 round 1 fixed-probe E1/E2 (the decisive v6 test)"
$FD encoder-checks --policy runs/v6/round1/decoder/decoder.json \
    --encoder runs/v6/round1/encoder/encoder.pt --output runs/v6/e1e2-fixed-round1.json \
    --episodes 50 --seconds 120 --workers 6 --controller teacher
$PY -c "
import json
r = json.load(open('runs/v6/e1e2-fixed-round1.json'))
print('T11 E1 loom_auc', r['E1']['loom_auc'], '(round0 neutral 0.720, v4 fixed 0.732, bar 0.8)')
print('T11 E2 pass', r['E2']['passed'])"
log "T11 done"
