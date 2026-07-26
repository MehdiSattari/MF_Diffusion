#!/bin/bash
#SBATCH -A your-slurm-account
#SBATCH -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH -J uq_eval
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Uncertainty + downstream eval. Submit:
#   MF=runs/mf_6913678/ckpt_best.pt REG=runs/reg_6915230/ckpt_best.pt \
#     DIU=runs/diu_6793522/ckpt_best.pt K=30 SNR=20 sbatch slurm/eval_uncertainty.sbatch
set -euo pipefail
ml purge
module load Python/3.11.5-GCCcore-13.2.0
module load LLVM/16.0.6-GCCcore-13.2.0
export DRJIT_LIBLLVM_PATH="$(ls "$EBROOTLLVM"/lib/libLLVM*.so* 2>/dev/null | head -n1)"
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs runs/uq
source "${VENV_DIR:-$HOME/MF_CSI_Prediction/venv}/bin/activate"
export TF_CPP_MIN_LOG_LEVEL=3

K="${K:-30}"; SNR="${SNR:-20}"
EXTRA=""
[ -n "${MF:-}" ] && EXTRA="$EXTRA --mf-ckpt ${MF}"
[ -n "${DIU:-}" ] && EXTRA="$EXTRA --diu-ckpt ${DIU}"
[ -n "${REG:-}" ] && EXTRA="$EXTRA --reg-ckpt ${REG}"

ts=$(date +%y%m%d_%H%M); LOG="logs/uq_${ts}.txt"
echo "===== UQ EVAL STARTED $(date) =====" | tee "$LOG"
python -m scripts.evaluate_uncertainty --K "$K" --snr "$SNR" $EXTRA >> "$LOG" 2>&1
echo "===== FINISHED $(date) =====" | tee -a "$LOG"
