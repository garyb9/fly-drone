#!/usr/bin/env bash
# Pre-Task-13 smoke: short 6-worker encoder and decoder SAC rounds on the real path.
# frames 9000 > learning_starts 5000; warm-up 6000 so the actor updates only in the last 3000 frames.
set -uo pipefail
cd /home/gb/projects/fly-drone
FD="env -u PYTHONPATH .venv/bin/fly-drone"
S=runs/v5/smoke
log() { echo "[$(date '+%F %T')] $*"; }
DRIVER_PID=${1:?driver pid}
DRIVER_LOG=${2:?driver log}
while kill -0 "$DRIVER_PID" 2>/dev/null; do sleep 15; done
grep -q "stage 2a done" "$DRIVER_LOG" || { log "SKIP: stage 2a did not finish"; exit 1; }
mkdir -p $S
( while true; do echo "$(date +%T) $(free -m | awk '/Mem:/{print $3, $7}')"; sleep 5; done ) > $S/ram.log &
RAM=$!

t0=$(date +%s)
$FD sac-round encoder --output $S/encoder --frames 9000 --actor-warmup 6000 --decoder runs/v5/round0/decoder.json \
    --init runs/v5/clone/encoder.pt --workers 6 --seed 7 > $S/encoder.log 2>&1
e1=$?; t1=$(date +%s); log "encoder round exit=$e1 wall=$((t1 - t0))s"

$FD sac-round decoder --output $S/decoder --frames 9000 --actor-warmup 6000 --encoder $S/encoder/encoder.pt \
    --init runs/v5/round0/decoder.zip --workers 6 --seed 7 > $S/decoder.log 2>&1
e2=$?; t2=$(date +%s); log "decoder round exit=$e2 wall=$((t2 - t1))s"
kill $RAM
log "peak used MB / min available MB: $(awk '{if($2>u)u=$2; if(a==""||$3<a)a=$3} END{print u, a}' $S/ram.log)"
log "smoke done"
