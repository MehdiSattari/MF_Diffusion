#!/bin/bash
# One-time LOCAL venv for small, non-Sionna tasks (Mac/Linux).
#   bash setup_env_local.sh
#   source .venv/bin/activate
# Sionna data-gen and GPU training stay on Alvis (setup_env_alvis.sh).
set -euo pipefail
cd "$(dirname "$0")"
PYBIN="${PYBIN:-python3}"
"$PYBIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
pip install -r requirements-local.txt
echo ""
python -c "import torch, numpy, matplotlib; print('local env OK | torch', torch.__version__)"
echo "activate later with:  source .venv/bin/activate"
