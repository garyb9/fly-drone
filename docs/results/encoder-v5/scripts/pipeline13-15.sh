#!/usr/bin/env bash
# Tasks 13-15 compute pipeline (user approved autonomous run through Task 15, 2026-09-15).
# T13: alternating encoder/decoder SAC rounds with the plan's stop rule; T14: bypass + E3; T15: evaluation + E1/E2 + E4.
# Phase 1.5 (2026-09-16, after the round-1 encoder collapse): the encoder round is staged. The
# first 100k frames stop at a checkpoint (keep-resume), a mid-round validation catches a blind
# encoder within ~100k frames instead of a full 350k round, and only then does stage B continue.
# The mid gate reuses ROUND_GATE's foraging threshold; no new threshold is introduced.
set -euo pipefail
cd /home/gb/projects/fly-drone
FD="env -u PYTHONPATH .venv/bin/fly-drone"
PY="env -u PYTHONPATH .venv/bin/python"
log() { echo "[$(date '+%F %T')] $*"; }

# Round 0 is a candidate for the final pair (clone encoder + round-0 decoder).
declare -A ENC DEC
ENC[0]=runs/v5/clone/encoder.pt
DEC[0]=runs/v5/round0/decoder.json
START=${START_ROUND:-1}
ENC_MID_FRAMES=${ENC_MID_FRAMES:-100000}
ENC_FRAMES=${ENC_FRAMES:-350000}
DEC_FRAMES=${DEC_FRAMES:-150000}
# Set SKIP_STAGE_A=1 to reuse a stage-A snapshot (e.g. after fixing a later step); the staged
# encoder round then starts from the snapshot already on disk.
SKIP_STAGE_A=${SKIP_STAGE_A:-0}
# NO_STOP=1 keeps the ladder going through every round even when one is ineligible or does not
# improve (the user asked to push rounds 2-3 after round 1 collected no beacons). The guard still
# records eligibility and step 6 still selects only eligible rounds.
NO_STOP=${NO_STOP:-0}

# Rounds before START were run in an earlier invocation; point the final-pair arrays at their
# on-disk outputs so step 6 can still choose them (e.g. START_ROUND=2 after a manual round 1).
for j in $(seq 1 $((START - 1))); do
    ENC[$j]=runs/v5/round$j/encoder/encoder.pt
    DEC[$j]=runs/v5/round$j/decoder/decoder.json
done

# Round selection guard (2026-09-16): a round must keep foraging, dodge on sight (ghost) and keep
# E2 semantics to count. Prints "<eligible> <best_eligible_index>" for rounds 0..k.
guard() {
    $PY -c "
import json, sys
from fly_drone.roam_eval import pick_best_round, round_eligible
k = int(sys.argv[1])
vals = [json.load(open('runs/v5/round0/validation.json' if i == 0 else f'runs/v5/round{i}/validation.json')) for i in range(k + 1)]
print(int(round_eligible(vals[k], vals[0])), pick_best_round(vals))" "$1"
}

# Mid-round degeneracy gate: the phase-1c probe logged during training. A blind encoder's
# deployed output stops varying with the observation (the round-1 collapse); the warm start sits
# near 0.22. This is a degeneracy check, not an acceptance bar: it only rejects an essentially
# constant action, so a healthy but drifting encoder cannot fail it. A decoder pairing cannot be
# used here because a decoder JSON pins the encoder version it was trained against.
ENC_PROBE_FLOOR=${ENC_PROBE_FLOOR:-0.02}
mid_gate() {
    local value
    value=$(grep -oE 'action_state_std[[:space:]]*\|[[:space:]]*[0-9.eE+-]+' "$1" | tail -1 | grep -oE '[0-9.eE+-]+$')
    if [ -z "$value" ]; then
        log "mid gate: no action_state_std in $1; refusing to continue"
        return 1
    fi
    $PY -c "import sys; v=float('$value'); print(f'mid action_state_std {v:.3f} (floor $ENC_PROBE_FLOOR)'); sys.exit(0 if v >= $ENC_PROBE_FLOOR else 1)"
}

last=0
for k in 1 2 3; do
    if [ "$k" -eq 1 ]; then
        PREV_DEC=runs/v5/round0/decoder.json; PREV_DEC_ZIP=runs/v5/round0/decoder.zip; ENC_INIT=runs/v5/clone/encoder.pt
    else
        PREV_DEC=runs/v5/round$((k - 1))/decoder/decoder.json; PREV_DEC_ZIP=runs/v5/round$((k - 1))/decoder/decoder.zip
        ENC_INIT=runs/v5/round$((k - 1))/encoder/encoder.zip
    fi
    if [ "$k" -ge "$START" ]; then
        if [ "$SKIP_STAGE_A" = "1" ] && [ -f runs/v5/round$k/encoder/resume/model.zip ]; then
            log "T13 round $k encoder stage A skipped (reusing runs/v5/round$k/encoder/resume)"
        else
            log "T13 round $k encoder SAC stage A (to $ENC_MID_FRAMES, keeping the snapshot)"
            $FD sac-round encoder --output runs/v5/round$k/encoder --frames $ENC_MID_FRAMES --decoder $PREV_DEC --init $ENC_INIT \
                --workers 6 --keep-resume > runs/v5/round$k-encoder-a.log 2>&1
        fi
        log "T13 round $k mid-round probe gate"
        if ! mid_gate runs/v5/round$k-encoder-a.log; then
            rm -rf runs/v5/round$k/encoder/resume  # drop the one-shot snapshot
            log "T13 STOP RULE: round $k encoder is blind by the $ENC_MID_FRAMES checkpoint; keeping round $((k - 1))"
            break
        fi
        log "T13 round $k encoder SAC stage B (continue to $ENC_FRAMES)"
        $FD sac-round encoder --output runs/v5/round$k/encoder --frames $ENC_FRAMES --decoder $PREV_DEC --init $ENC_INIT --resume \
            --workers 6 > runs/v5/round$k-encoder.log 2>&1
        log "T13 round $k decoder SAC"
        $FD sac-round decoder --output runs/v5/round$k/decoder --frames $DEC_FRAMES \
            --encoder runs/v5/round$k/encoder/encoder.pt --init $PREV_DEC_ZIP --workers 6 > runs/v5/round$k-decoder.log 2>&1
        log "T13 round $k validate"
        $FD sac-validate --decoder runs/v5/round$k/decoder/decoder.json --encoder runs/v5/round$k/encoder/encoder.pt \
            --output runs/v5/round$k/validation.json
    fi
    ENC[$k]=runs/v5/round$k/encoder/encoder.pt
    DEC[$k]=runs/v5/round$k/decoder/decoder.json
    last=$k
    read -r elig best_so_far <<<"$(guard $k)"
    $PY -c "
import json, sys
from fly_drone.roam_eval import round_gate_report
last = int(sys.argv[1])
vals = [json.load(open('runs/v5/round0/validation.json' if i == 0 else f'runs/v5/round{i}/validation.json')) for i in range(last + 1)]
for r in round_gate_report(vals):
    print('T13 table', r)" "$last"
    log "T13 round $k eligible=$elig best_eligible=$best_so_far"
    if [ "$elig" -eq 0 ]; then
        if [ "$NO_STOP" = "1" ]; then
            log "T13 NO_STOP: round $k ineligible (foraging / ghost / E2); continuing anyway"
        else
            log "T13 STOP RULE: round $k ineligible (foraging / ghost / E2); no further rounds"
            break
        fi
    elif [ "$best_so_far" -ne "$k" ]; then
        if [ "$NO_STOP" = "1" ]; then
            log "T13 NO_STOP: round $k did not improve the best eligible near-dodge; continuing anyway"
        else
            log "T13 STOP RULE: round $k did not improve the best eligible near-dodge; no further rounds"
            break
        fi
    fi
done

# Step 6: best eligible round 0..last by near-dodge, ties to more beacons/min; round 0 if none.
best=$($PY -c "
import json, sys
from fly_drone.roam_eval import pick_best_round
last = int(sys.argv[1])
vals = [json.load(open('runs/v5/round0/validation.json' if i == 0 else f'runs/v5/round{i}/validation.json')) for i in range(last + 1)]
print(pick_best_round(vals))" "$last")
mkdir -p runs/v5/final
cp "${ENC[$best]}" runs/v5/final/encoder.pt
cp "${DEC[$best]}" runs/v5/final/decoder.json
echo "{\"round\": $best, \"encoder\": \"${ENC[$best]}\", \"decoder\": \"${DEC[$best]}\"}" > runs/v5/final/choice.json
log "T13 done: final pair = round $best"

log "T14 bypass SAC"
$FD sac-round bypass --output runs/v5/bypass --frames 450000 --encoder runs/v5/final/encoder.pt --workers 6 \
    > runs/v5/bypass.log 2>&1
log "T14 E3 screen"
$FD roam-screen runs/v5/final/decoder.json runs/v5/bypass/bypass.zip --encoder runs/v5/final/encoder.pt \
    --ablations none ghost --seeds 50 --seconds 120 --level 3 --seed-base 1000 --workers 6 --output runs/v5/e3-screen.json
$PY -c "
import json; from pathlib import Path; from fly_drone.roam_eval import bypass_comparison
r = json.loads(Path('runs/v5/e3-screen.json').read_text())['results']
full = next(v for k, v in r.items() if k.startswith('policy:') and k.endswith('|none'))
byp = next(v for k, v in r.items() if k.startswith('bypass:') and k.endswith('|none'))
out = bypass_comparison(full, byp); Path('runs/v5/e3.json').write_text(json.dumps(out, indent=2)); print(out)"
log "T14 done"

log "T15 evaluation"
$FD evaluate --task free_roam --policy runs/v5/final/decoder.json --encoder runs/v5/final/encoder.pt --episodes 50 \
    --workers 6 --output runs/v5/evaluation.json > runs/v5/evaluation.log 2>&1
log "T15 encoder checks"
$FD encoder-checks --policy runs/v5/final/decoder.json --encoder runs/v5/final/encoder.pt --output runs/v5/e1-e2.json
log "T15 E4"
$PY -m pytest -q tests/test_arena.py::test_legacy_room_mjcf_unchanged tests/test_env.py -k "legacy or replay_is_bit_identical" \
    > runs/v5/e4.log 2>&1 || log "T15 E4 pytest FAILED (see runs/v5/e4.log)"
log "pipeline done"
