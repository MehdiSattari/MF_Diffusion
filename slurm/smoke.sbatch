#!/bin/bash
#SBATCH -A your-slurm-account              # compute project (charge): NAISS 2026/3-189
#SBATCH -p alvis
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH -J mf_csi_smoke
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Runs the full test + smoke suite on a GPU node.  Submit:  sbatch slurm/smoke.sbatch
set -euo pipefail

ml purge
module load Python/3.11.5-GCCcore-13.2.0
module load LLVM/16.0.6-GCCcore-13.2.0
export DRJIT_LIBLLVM_PATH="$(ls "$EBROOTLLVM"/lib/libLLVM*.so* 2>/dev/null | head -n1)"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
VENV_DIR="${VENV_DIR:-$HOME/MF_CSI_Prediction/venv}"
source "${VENV_DIR}/bin/activate"
export TF_CPP_MIN_LOG_LEVEL=3

timestamp=$(date +%y%m%d_%H%M)
LOG="logs/smoke_${timestamp}.txt"
echo "===== SMOKE STARTED AT $(date) =====" | tee "$LOG"
{
    python -c "import torch; print('cuda:', torch.cuda.is_available())"
    python -m tests.test_dataset_mock
    python -m tests.test_encoder
    python -m tests.test_unet
    python -m tests.test_meanflow
    python -m tests.test_inference
    python -m scripts.smoke_generator
    python -m scripts.smoke_meanflow
    python -m scripts.smoke_inference
} >> "$LOG" 2>&1
echo "===== SMOKE FINISHED AT $(date) =====" | tee -a "$LOG"
