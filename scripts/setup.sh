#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Avoid inheriting unrelated ROS Python package/plugin paths.
unset PYTHONPATH
git submodule update --init --recursive
[ -x .venv/bin/python ] || python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'maturin>=1.7,<2'
python -m pip install -e vendor/mujoco-drones
maturin develop --release --extras dev,train
command -v yarn >/dev/null || corepack enable
yarn install --frozen-lockfile
yarn build
