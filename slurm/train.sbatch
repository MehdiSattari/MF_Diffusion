#!/bin/bash
#SBATCH -A your-slurm-account              # compute project (charge): NAISS 2026/3-189
#SBATCH -p alvis
#SBATCH -N 1 --gpus-per-node=T4:1       # 1 T4 GPU (override at submit for A40/V100)
#SBATCH -J mf_csi
#SBATCH --time=0-01:00:00               # 1h is plenty for the data smoke test
#SBATCH --output=logs/%x_%j.out         # SLURM stdout
#SBATCH --error=logs/%x_%j.err          # SLURM stderr
# -----------------------------------------------------------------------------
# Alvis limits per job: 4 GPUs, 40 CPUs, 2 days, 117 GB memory.
#
# Submit:
#     sbatch slurm/train.sbatch
# Override venv / GPU at submit time, e.g.:
#     sbatch --gpus-per-node=A40:1 \
#            --export=ALL,VENV_DIR=/path/to/venv slurm/train.sbatch
# -----------------------------------------------------------------------------
set -euo pipefail

ml purge
module load Python/3.11.5-GCCcore-13.2.0
# libLLVM.so for Mitsuba (Sionna 0.19 imports its rt module on import).
module load LLVM/16.0.6-GCCcore-13.2.0
export DRJIT_LIBLLVM_PATH="$(ls "$EBROOTLLVM"/lib/libLLVM*.so* 2>/dev/null | head -n1)"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
echo "directory: $(pwd)"

VENV_DIR="${VENV_DIR:-$HOME/MF_CSI_Prediction/venv}"
source "${VENV_DIR}/bin/activate"

# Quiet TensorFlow; it stays on CPU (tensorflow-cpu + tf.config in our code).
# NOTE: deliberately do NOT set CUDA_VISIBLE_DEVICES here — PyTorch needs the GPU.
export TF_CPP_MIN_LOG_LEVEL=3

timestamp=$(date +%y%m%d_%H%M)
LOG="logs/mf_csi_${timestamp}.txt"
echo "===== JOB STARTED AT $(date) =====" | tee "$LOG"

# Confirm PyTorch sees the GPU on the compute node.
python - <<'PY' >> "$LOG" 2>&1
import torch
print("torch", torch.__version__, "| cuda.is_available:", torch.cuda.is_available(),
      "| device:", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))
PY

# Validation so far (data pipeline + ConvLSTM encoder + U-Net generator).
# In later steps this becomes:  python -m scripts.train --config configs/default.yaml
{
    echo "### unit tests ###"
    python -m tests.test_dataset_mock
    python -m tests.test_encoder
    python -m tests.test_unet
    echo "### end-to-end smoke (data -> encoder -> generator) ###"
    python -m scripts.smoke_generator
} >> "$LOG" 2>&1

echo "===== JOB FINISHED AT $(date) =====" | tee -a "$LOG"
