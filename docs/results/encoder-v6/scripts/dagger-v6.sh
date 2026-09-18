#!/usr/bin/env bash
# DAgger iterations 0-3 for free roam on the frozen v6 pair (plan 08 Task 3).
#
# Iter 0-1 use levels 0-2 (no threats); iter 2-3 add level 3 so the threat/recovery drive is
# present. Iters 1-3 fly the previous student with probability 1-beta and always label with the
# teacher. Each student is screened at level 3 with the ghost ablation, against teacher and random.
#
# Run:  setsid nohup env -u PYTHONPATH docs/results/encoder-v6/scripts/dagger-v6.sh \
#         > runs/roam/dagger-v6-driver.log 2>&1 &
set -uo pipefail

ENC=runs/v6/clone/encoder.pt
OUT=runs/roam
W=6

FLIGHTS0=128
FLIGHTS=96
SEEDS=(4000 5000 6000 7000)
LEVELS_EARLY="0 1 2"
LEVELS_LATE="0 1 2 3"
BETAS=(- 0.5 0.25 0)

log() { echo "[$(date +%H:%M:%S)] $*"; }

for iter in 0 1 2 3; do
  levels=$LEVELS_EARLY
  [ "$iter" -ge 2 ] && levels=$LEVELS_LATE
  flights=$FLIGHTS
  [ "$iter" -eq 0 ] && flights=$FLIGHTS0
  beta=${BETAS[$iter]}
  seedbase=${SEEDS[$iter]}

  collect=(roam-collect --output "$OUT/v6-d$iter.npz" --flights "$flights" --seconds 60
           --encoder "$ENC" --workers "$W" --seed-base "$seedbase" --levels $levels)
  if [ "$iter" -gt 0 ]; then
    student="$OUT/v6-it$((iter - 1))/warm-actor.json"
    collect+=(--student "$student" --beta "$beta")
  fi
  log "iter $iter: collect flights=$flights levels='$levels' beta=$beta seed=$seedbase"
  if ! env -u PYTHONPATH .venv/bin/fly-drone "${collect[@]}" >> "$OUT/dagger-v6.log" 2>&1; then
    log "iter $iter: collect FAILED"; exit 1
  fi

  data=()
  for k in $(seq 0 "$iter"); do data+=("$OUT/v6-d$k.npz"); done
  log "iter $iter: refit on ${#data[@]} npz"
  if ! env -u PYTHONPATH .venv/bin/fly-drone roam-fit "${data[@]}" --encoder "$ENC" \
        --output "$OUT/v6-it$iter" >> "$OUT/dagger-v6.log" 2>&1; then
    log "iter $iter: fit FAILED"; exit 1
  fi

  log "iter $iter: screen (L3, none+ghost)"
  if ! env -u PYTHONPATH .venv/bin/fly-drone roam-screen \
        "$OUT/v6-it$iter/warm-actor.json" teacher random \
        --seeds 10 --seconds 60 --level 3 --ablations none ghost \
        --encoder "$ENC" --workers "$W" \
        --output "$OUT/v6-it$iter-screen.json" >> "$OUT/dagger-v6.log" 2>&1; then
    log "iter $iter: screen FAILED"; exit 1
  fi
  log "iter $iter: done"
done

log "DAgger 0-3 complete"
