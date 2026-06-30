#!/bin/bash
# =============================================================================
# One-time virtual-environment setup for the MeanFlow CSI project on Alvis.
#
# Strategy: a self-contained pip venv on the same Python module your previous
# `miae_sionna` venv used. PyTorch gets a CUDA wheel (owns the GPU);
# TensorFlow is installed CPU-only (`tensorflow-cpu`) because Sionna only does
# channel generation here, which we pin to CPU. Installing tensorflow-cpu
# (instead of full `tensorflow`) avoids CUDA-library conflicts with PyTorch.
#
# Run on a login node (pip-only, no compute) or inside an interactive job:
#     bash setup_env_alvis.sh
# Override the venv location if you like:
#     VENV_DIR=/path/to/venv bash setup_env_alvis.sh
# =============================================================================
set -euo pipefail

ml purge
module load Python/3.11.5-GCCcore-13.2.0
# LLVM provides libLLVM.so, needed by Mitsuba (pulled in by Sionna's rt module,
# which Sionna 0.19 imports on `import sionna`). Adjust version if `ml spider
# LLVM` shows a different one available.
module load LLVM/16.0.6-GCCcore-13.2.0
export DRJIT_LIBLLVM_PATH="$(ls "$EBROOTLLVM"/lib/libLLVM*.so* 2>/dev/null | head -n1)"

# Where the venv lives (project-group group storage; override freely).
VENV_DIR="${VENV_DIR:-$HOME/MF_CSI_Prediction/venv}"
CACHE_PARENT="$(dirname "$(dirname "$VENV_DIR")")"   # .../project-group/user

# Keep pip cache + temp off the small $HOME quota (and out of the repo).
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CACHE_PARENT}/.pipcache}"
export TMPDIR="${TMPDIR:-${PIP_CACHE_DIR}/tmp}"
mkdir -p "$(dirname "$VENV_DIR")" "$PIP_CACHE_DIR" "$TMPDIR"

echo ">>> Creating venv at: $VENV_DIR"
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# 1) PyTorch with CUDA 12.1 (runs on Alvis T4/V100/A40 drivers).
echo ">>> Installing PyTorch (CUDA 12.1)"
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2) TensorFlow CPU-only — Sionna's backend, kept off the GPU by design.
echo ">>> Installing tensorflow-cpu"
pip install "tensorflow-cpu>=2.13,<2.16"

# 3) Sionna without deps (so it does NOT pull the full GPU 'tensorflow' and
#    clash with PyTorch's CUDA libs), then its runtime deps explicitly.
#    NOTE: Sionna 0.19's top-level __init__ eagerly imports its ray-tracing
#    module, so even though we only use the channel models we must install
#    mitsuba + the widget packages or `import sionna` fails. Adjust the sionna
#    version if your cluster's working version differs (old venv was 'miae_sionna').
echo ">>> Installing Sionna (+ runtime deps)"
pip install --no-deps "sionna>=0.19,<1.0"
pip install "numpy<2.0" scipy matplotlib importlib_resources \
            "mitsuba>=3.2.0,<3.6.0" "ipywidgets>=8.0.4" "ipydatawidgets==4.3.2" \
            "pythreejs>=2.4.2" "jupyterlab-widgets==3.0.5"

# 4) Project utilities.
pip install pyyaml tqdm

echo ">>> Verifying imports"
python - <<'PY'
import torch
print("torch       :", torch.__version__, "| CUDA build:", torch.version.cuda,
      "| cuda.is_available (False on a login node is fine):", torch.cuda.is_available())
import tensorflow as tf
print("tensorflow  :", tf.__version__, "| GPUs visible to TF (should be []):",
      tf.config.list_physical_devices("GPU"))
import sionna
print("sionna      :", sionna.__version__)
PY

echo ""
echo ">>> Done. venv ready at: $VENV_DIR"
echo ">>> If torch.cuda.is_available() was False here, that's expected on a"
echo "    login node; the GPU check that matters runs inside the SLURM job."
