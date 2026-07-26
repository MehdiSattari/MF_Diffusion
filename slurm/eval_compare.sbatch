#!/bin/bash
#SBATCH -A your-slurm-account              # compute project: NAISS 2026/3-189
#SBATCH -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH -J mf_eval_compare
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# DiU-vs-MeanFlow NMSE-vs-step comparison. Submit:
#     DIU=runs/diu_6793522/ckpt_best.pt MF=runs/mf_6795740/ckpt_best.pt \
#         sbatch slurm/eval_compare.sbatch
# Optional: SNRS="0,10,20" CLEAN=1 NSAMPLES=512 sbatch slurm/eval_compare.sbatch
set -euo pipefail

ml purge
module load Python/3.11.5-GCCcore-13.2.0
module load LLVM/16.0.6-GCCcore-13.2.0
export DRJIT_LIBLLVM_PATH="$(ls "$EBROOTLLVM"/lib/libLLVM*.so* 2>/dev/null | head -n1)"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs/compare
VENV_DIR="${VENV_DIR:-$HOME/MF_CSI_Prediction/venv}"
source "${VENV_DIR}/bin/activate"
export TF_CPP_MIN_LOG_LEVEL=3

DIU="${DIU:-runs/diu_6793522/ckpt_best.pt}"
MF="${MF:-runs/mf_6795740/ckpt_best.pt}"
SNRS="${SNRS:-20}"
NSAMPLES="${NSAMPLES:-256}"
OUT="${OUT:-runs/compare}"
EXTRA=""
if [ -n "${CLEAN:-}" ]; then EXTRA="${EXTRA} --clean"; fi
[ -n "${REG:-}" ] && EXTRA="${EXTRA} --reg-ckpt ${REG}"
[ -n "${CONVLSTM:-}" ] && EXTRA="${EXTRA} --convlstm-ckpt ${CONVLSTM}"

timestamp=$(date +%y%m%d_%H%M)
LOG="logs/eval_compare_${timestamp}.txt"
echo "===== EVAL COMPARE STARTED AT $(date) =====" | tee "$LOG"
echo "DIU=$DIU  MF=$MF  SNRS=$SNRS  NSAMPLES=$NSAMPLES" | tee -a "$LOG"
python -m scripts.evaluate_compare \
    --diu-ckpt "$DIU" --mf-ckpt "$MF" \
    --snrs "$SNRS" --n-samples "$NSAMPLES" --out-dir "$OUT" $EXTRA >> "$LOG" 2>&1
echo "===== EVAL COMPARE FINISHED AT $(date) =====" | tee -a "$LOG"
