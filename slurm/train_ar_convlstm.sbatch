#!/bin/bash
#SBATCH -A your-slurm-account
#SBATCH -p alvis
#SBATCH -N 1 --gpus-per-node=T4:1        # override: --gpus-per-node=A40:1
#SBATCH -J ar_lstm
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# AR ConvLSTM baseline. Submit:
#   BATCH=256 STEPS=50000 sbatch --gpus-per-node=A40:1 slurm/train_ar_convlstm.sbatch
set -euo pipefail
ml purge
module load Python/3.11.5-GCCcore-13.2.0
module load LLVM/16.0.6-GCCcore-13.2.0
export DRJIT_LIBLLVM_PATH="$(ls "$EBROOTLLVM"/lib/libLLVM*.so* 2>/dev/null | head -n1)"
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "${VENV_DIR:-$HOME/MF_CSI_Prediction/venv}/bin/activate"
export TF_CPP_MIN_LOG_LEVEL=3
OUT="${OUT:-runs/arlstm_${SLURM_JOB_ID}}"; STEPS="${STEPS:-50000}"
EXTRA=""; [ -n "${BATCH:-}" ] && EXTRA="--batch-size ${BATCH}"; [ -n "${RESUME:-}" ] && EXTRA="$EXTRA --resume ${RESUME}"
ts=$(date +%y%m%d_%H%M); LOG="logs/arlstm_${ts}.txt"
echo "===== AR-CONVLSTM STARTED $(date) =====" | tee "$LOG"
python -m scripts.train_ar_convlstm --out-dir "$OUT" --steps "$STEPS" $EXTRA >> "$LOG" 2>&1
echo "===== FINISHED $(date) =====" | tee -a "$LOG"
