#!/usr/bin/env bash
# Phase 0 D2/D3: behavioral timeline. Runs after D1 finishes; sequential to bound RAM.
set -uo pipefail
cd /home/gb/projects/fly-drone
FD="env -u PYTHONPATH .venv/bin/fly-drone"

run() { # tag decoder encoder
    local tag=$1 dec=$2 enc=$3
    local dir="runs/v5/diag/$tag"
    mkdir -p "$dir"
    echo "[$(date '+%F %T')] $tag start"
    $FD sac-validate --decoder "$dec" --encoder "$enc" --output "$dir/validation.json" \
        > "$dir/validate.log" 2>&1
    echo "[$(date '+%F %T')] $tag exit $?"
}

# D2: round-0 decoder under round-1 encoder checkpoints (isolates the encoder)
run d2-100k runs/v5/diag/d2/decoder_99996.json runs/v5/diag/d2/encoder_99996.pt
run d2-150k runs/v5/diag/d2/decoder_149994.json runs/v5/diag/d2/encoder_149994.pt
# D3: round-1 decoder checkpoints under the round-1 final encoder (isolates the decoder)
run d3-50k runs/v5/diag/d3/decoder_49998.json runs/v5/round1/encoder/encoder.pt
run d3-100k runs/v5/diag/d3/decoder_99996.json runs/v5/round1/encoder/encoder.pt
echo "[$(date '+%F %T')] d2/d3 chain done"
