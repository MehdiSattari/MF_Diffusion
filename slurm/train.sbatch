#!/bin/bash
#SBATCH -A your-slurm-account              # compute project (charge): NAISS 2026/3-189
#SBATCH -p alvis
#SBATCH -N 1 --gpus-per-node=T4:1       # override for real runs: --gpus-per-node=A40:1
#SBATCH -J mf_csi_train
#SBATCH --time=1-00:00:00               # 1 day
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# -----------------------------------------------------------------------------
# Real MeanFlow DiU training. Submit:
#     sbatch slurm/train.sbatch
# Faster / longer:
#     sbatch --gpus-per-node=A40:1 slurm/train.sbatch
# Resume a run:
#     STEPS=100000 RESUME=runs/exp1/ckpt_last.pt OUT=runs/exp1 sbatch slurm/train.sbatch
# -----------------------------------------------------------------------------
set -euo pipefail

ml purge
module load Python/3.11.5-GCCcore-13.2.0
module load LLVM/16.0.6-GCCcore-13.2.0
export DRJIT_LIBLLVM_PATH="$(ls "$EBROOTLLVM"/lib/libLLVM*.so* 2>/dev/null | head -n1)"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
VENV_DIR="${VENV_DIR:-$HOME/MF_CSI_Prediction/venv}"
source "${VENV_DIR}/bin/activate"
export TF_CPP_MIN_LOG_LEVEL=3          # keep TF/Sionna quiet; it stays on CPU

# Run parameters (override at submit time via --export=ALL,VAR=...).
OUT="${OUT:-runs/mf_${SLURM_JOB_ID}}"
STEPS="${STEPS:-80000}"
EXTRA=""
if [ -n "${RESUME:-}" ]; then EXTRA="${EXTRA} --resume ${RESUME}"; fi
if [ -n "${LR:-}" ]; then EXTRA="${EXTRA} --lr ${LR}"; fi
if [ -n "${BATCH:-}" ]; then EXTRA="${EXTRA} --batch-size ${BATCH}"; fi
if [ -n "${SIGMA:-}" ]; then EXTRA="${EXTRA} --source-std ${SIGMA}"; fi

timestamp=$(date +%y%m%d_%H%M)
LOG="logs/train_${timestamp}.txt"
echo "===== TRAIN STARTED AT $(date) =====" | tee "$LOG"
python -m scripts.train --out-dir "$OUT" --steps "$STEPS" $EXTRA >> "$LOG" 2>&1

# Auto-generate the DiU-vs-MeanFlow NMSE-vs-step comparison plot from the best
# checkpoint as soon as training finishes (set AUTO_COMPARE=0 to skip). Output
# lands in $OUT/compare/: nmse_vs_step_compare.png + .json.
DIU_CKPT="${DIU_CKPT:-runs/diu_6793522/ckpt_best.pt}"
if [ "${AUTO_COMPARE:-1}" != "0" ] && [ -f "$OUT/ckpt_best.pt" ] && [ -f "$DIU_CKPT" ]; then
  echo "===== AUTO COMPARE PLOT AT $(date) =====" | tee -a "$LOG"
  REG_ARG=""; [ -n "${REG:-}" ] && [ -f "${REG}" ] && REG_ARG="--reg-ckpt ${REG}"
  python -m scripts.evaluate_compare --diu-ckpt "$DIU_CKPT" --mf-ckpt "$OUT/ckpt_best.pt" \
      $REG_ARG --snrs 20 --clean --out-dir "$OUT/compare" >> "$LOG" 2>&1 \
      && echo "compare plot -> $OUT/compare/nmse_vs_step_compare.png" | tee -a "$LOG" \
      || echo "auto-compare failed (see $LOG)" | tee -a "$LOG"
fi
echo "===== TRAIN FINISHED AT $(date) =====" | tee -a "$LOG"
