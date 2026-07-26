#!/bin/bash
#SBATCH -A your-slurm-account
#SBATCH -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH -J s2s_eval
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Seq2seq three-way comparison. Submit:
#   SIZE=large MF=runs/s2s_mf_L/ckpt_best.pt DIFF=runs/s2s_di_L/ckpt_best.pt \
#     REG=runs/s2s_reg/ckpt_best.pt SNRS=0,10,20 CLEAN=1 sbatch slurm/eval_seq2seq.sbatch
set -euo pipefail
ml purge
module load Python/3.11.5-GCCcore-13.2.0
module load LLVM/16.0.6-GCCcore-13.2.0
export DRJIT_LIBLLVM_PATH="$(ls "$EBROOTLLVM"/lib/libLLVM*.so* 2>/dev/null | head -n1)"
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs runs/s2s_compare
source "${VENV_DIR:-$HOME/MF_CSI_Prediction/venv}/bin/activate"
export TF_CPP_MIN_LOG_LEVEL=3

SIZE="${SIZE:-large}"; SNRS="${SNRS:-20}"
EXTRA=""
[ -n "${MF:-}" ] && EXTRA="$EXTRA --mf-ckpt ${MF}"
[ -n "${DIFF:-}" ] && EXTRA="$EXTRA --diff-ckpt ${DIFF}"
[ -n "${REG:-}" ] && EXTRA="$EXTRA --reg-ckpt ${REG}"
[ -n "${CLEAN:-}" ] && EXTRA="$EXTRA --clean"

ts=$(date +%y%m%d_%H%M); LOG="logs/s2s_eval_${ts}.txt"
echo "===== SEQ2SEQ EVAL STARTED $(date) =====" | tee "$LOG"
python -m scripts.evaluate_seq2seq --size "$SIZE" --snrs "$SNRS" $EXTRA >> "$LOG" 2>&1
echo "===== FINISHED $(date) =====" | tee -a "$LOG"
