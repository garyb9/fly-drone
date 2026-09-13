#!/usr/bin/env bash
# Run a command inside the project venv without requiring activation.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
if [ ! -x "$root/.venv/bin/python" ]; then
  echo "Missing .venv — run: yarn setup" >&2
  exit 1
fi
# Unrelated ROS/system Python paths break MuJoCo and pytest plugin discovery.
unset PYTHONPATH
export VIRTUAL_ENV="$root/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
exec "$@"
