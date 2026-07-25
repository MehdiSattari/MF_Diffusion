#!/bin/bash
#SBATCH -A your-slurm-account
#SBATCH -p alvis
#SBATCH -N 1 --gpus-per-node=T4:1        # override for real runs: --gpus-per-node=A40:1
#SBATCH -J mf_reg
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Joint-horizon ConvLSTM regression baseline. Submit:
#     BATCH=256 STEPS=50000 sbatch --gpus-per-node=A40:1 slurm/train_regression.sbatch
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

OUT="${OUT:-runs/reg_${SLURM_JOB_ID}}"
STEPS="${STEPS:-50000}"
EXTRA=""
[ -n "${RESUME:-}" ] && EXTRA="${EXTRA} --resume ${RESUME}"
[ -n "${LR:-}" ] && EXTRA="${EXTRA} --lr ${LR}"
[ -n "${BATCH:-}" ] && EXTRA="${EXTRA} --batch-size ${BATCH}"

timestamp=$(date +%y%m%d_%H%M)
LOG="logs/reg_${timestamp}.txt"
echo "===== REGRESSION TRAIN STARTED AT $(date) =====" | tee "$LOG"
python -m scripts.train_regression --out-dir "$OUT" --steps "$STEPS" $EXTRA >> "$LOG" 2>&1
echo "===== REGRESSION TRAIN FINISHED AT $(date) =====" | tee -a "$LOG"
