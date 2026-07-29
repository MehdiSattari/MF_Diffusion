#!/bin/bash
#SBATCH -A your-slurm-account
#SBATCH -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH -J uq2x2
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Final 2x2 mu-ablation UQ eval. Submit with the five checkpoints:
#   MF_MU=... MF_NOMU=... DIFF_MU=... DIFF_NOMU=... CONVLSTM=... \
#     K=30 SNR=20 DIFF_STEPS=3 sbatch slurm/eval_uncertainty_2x2.sbatch
set -euo pipefail
ml purge
module load Python/3.11.5-GCCcore-13.2.0
module load LLVM/16.0.6-GCCcore-13.2.0
export DRJIT_LIBLLVM_PATH="$(ls "$EBROOTLLVM"/lib/libLLVM*.so* 2>/dev/null | head -n1)"
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "${VENV_DIR:-$HOME/MF_CSI_Prediction/venv}/bin/activate"
export TF_CPP_MIN_LOG_LEVEL=3
K="${K:-30}"; SNR="${SNR:-20}"; DIFF_STEPS="${DIFF_STEPS:-3}"; SN="${SN:-0.05}"; ETA="${ETA:-1.0}"; SEED="${SEED:-0}"
tag=$([ "$ETA" = "0.0" ] && echo "det" || echo "stoch")
ts=$(date +%y%m%d_%H%M); OUT="runs/uq2x2_snr${SNR}_${tag}_${ts}"; LOG="logs/uq2x2_${tag}_${ts}.txt"
EXTRA=""
[ -n "${MF_MU:-}" ] && EXTRA="$EXTRA --mf-mu ${MF_MU}"
[ -n "${MF_NOMU:-}" ] && EXTRA="$EXTRA --mf-nomu ${MF_NOMU}"
[ -n "${DIFF_MU:-}" ] && EXTRA="$EXTRA --diff-mu ${DIFF_MU}"
[ -n "${DIFF_NOMU:-}" ] && EXTRA="$EXTRA --diff-nomu ${DIFF_NOMU}"
[ -n "${CONVLSTM:-}" ] && EXTRA="$EXTRA --convlstm ${CONVLSTM}"
echo "===== UQ 2x2 STARTED $(date) =====" | tee "$LOG"
python -m scripts.evaluate_uncertainty_2x2 --K "$K" --snr "$SNR" --diff-steps "$DIFF_STEPS" \
    --ddim-eta "$ETA" --seed "$SEED" --step-noise "$SN" --out-dir "$OUT" $EXTRA >> "$LOG" 2>&1
echo "===== FINISHED $(date) =====" | tee -a "$LOG"
