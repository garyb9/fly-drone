#!/usr/bin/env bash
# Task 11 (plan 05): Stage 1 DAgger decoder on L2 with encoder v4.
set -euo pipefail
cd /home/gb/projects/fly-drone
FD="env -u PYTHONPATH .venv/bin/fly-drone"
OUT=runs/v5/dagger
mkdir -p "$OUT"
stamp() { echo "[$(date '+%F %T')] $*"; }

stamp "it0 collect"
$FD roam-collect --output $OUT/it0.npz --flights 128 --seconds 60 --levels 2 --workers 6 --seed-base 200
stamp "it0 fit"
$FD roam-fit $OUT/it0.npz --output $OUT/it0
stamp "it0 screen"
$FD roam-screen $OUT/it0/warm-actor.json teacher random --level 2 --seeds 10 --seed-base 9000 --workers 6 --output $OUT/it0-screen.json

for pair in "1 0.5" "2 0.25" "3 0.0"; do
  set -- $pair
  k=$1
  beta=$2
  prev=$((k - 1))
  stamp "it$k collect (beta $beta)"
  $FD roam-collect --output $OUT/it$k.npz --student $OUT/it$prev/warm-actor.json --beta $beta --flights 128 --seconds 60 --levels 2 --workers 6 --seed-base $((200 + 1000 * k))
  stamp "it$k fit"
  $FD roam-fit $OUT/it*.npz --output $OUT/it$k
  stamp "it$k screen"
  $FD roam-screen $OUT/it$k/warm-actor.json --level 2 --seeds 10 --seed-base 9000 --workers 6 --output $OUT/it$k-screen.json
done
stamp "stage 1 done"
