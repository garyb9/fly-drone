#!/usr/bin/env bash
# Waits for the smoke test to finish, runs smoke_check.py, launches the T13–15 pipeline only on SMOKE_CHECK OK.
set -uo pipefail
cd /home/gb/projects/fly-drone
D=.superpowers/sdd/2026-09-14-fly-drone-05-encoder-v5-sac
log() { echo "[$(date '+%F %T')] $*"; }
until grep -qE "smoke done|SKIP" runs/v5/smoke13.log; do sleep 15; done
if grep -q SKIP runs/v5/smoke13.log; then log "smoke skipped: not launching pipeline"; exit 1; fi
grep -E "exit=[1-9]" runs/v5/smoke13.log && { log "smoke round failed: not launching pipeline"; exit 1; }
env -u PYTHONPATH .venv/bin/python runs/v5/diag/smoke_check.py > runs/v5/smoke-check.log 2>&1
cat runs/v5/smoke-check.log
if grep -q "SMOKE_CHECK OK" runs/v5/smoke-check.log; then
    setsid nohup bash $D/pipeline13-15.sh > runs/v5/pipeline13-15.log 2>&1 &
    log "SMOKE OK: pipeline13-15 launched pid $!"
else
    log "SMOKE_CHECK FAILED: not launching pipeline"; exit 2
fi
