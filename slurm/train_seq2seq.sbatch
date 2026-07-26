#!/bin/bash
#SBATCH -A your-slurm-account
#SBATCH -p alvis
#SBATCH -N 1 --gpus-per-node=T4:1        # override: --gpus-per-node=A40:1
#SBATCH -J s2s
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Controlled seq2seq trainer. Submit:
#   OBJ=meanflow SIZE=large sbatch --gpus-per-node=A40:1 slurm/train_seq2seq.sbatch
#   OBJ=diffusion SIZE=large sbatch --gpus-per-node=A40:1 slurm/train_seq2seq.sbatch
#   OBJ=regression sbatch --gpus-per-node=A40:1 slurm/train_seq2seq.sbatch
set -euo pipefail
ml purge
module load Python/3.11.5-GCCcore-13.2.0
module load LLVM/16.0.6-GCCcore-13.2.0
export DRJIT_LIBLLVM_PATH="$(ls "$EBROOTLLVM"/lib/libLLVM*.so* 2>/dev/null | head -n1)"
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "${VENV_DIR:-$HOME/MF_CSI_Prediction/venv}/bin/activate"
export TF_CPP_MIN_LOG_LEVEL=3

OBJ="${OBJ:-meanflow}"; SIZE="${SIZE:-large}"; STEPS="${STEPS:-50000}"
OUT="${OUT:-runs/s2s_${OBJ}_${SIZE}_${SLURM_JOB_ID}}"
EXTRA=""; [ -n "${BATCH:-}" ] && EXTRA="--batch-size ${BATCH}"; [ -n "${RESUME:-}" ] && EXTRA="$EXTRA --resume ${RESUME}"

ts=$(date +%y%m%d_%H%M); LOG="logs/s2s_${OBJ}_${SIZE}_${ts}.txt"
echo "===== SEQ2SEQ ($OBJ/$SIZE) STARTED $(date) =====" | tee "$LOG"
python -m scripts.train_seq2seq --objective "$OBJ" --size "$SIZE" --out-dir "$OUT" --steps "$STEPS" $EXTRA >> "$LOG" 2>&1
echo "===== FINISHED $(date) =====" | tee -a "$LOG"
