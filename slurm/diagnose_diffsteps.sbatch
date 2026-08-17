#!/bin/bash
#SBATCH -A your-slurm-account
#SBATCH -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH -J diffdiag
#SBATCH --time=03:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Root-cause sweep for "diffusion NMSE degrades with more DDIM steps".
# Sweeps steps x {init, spacing, eta} on the SAME paired channels and reports
# ensemble-mean NMSE (accuracy) vs per-sample NMSE + coverage/CRPS (calibration).
#
#   DIFF_MU=runs/ar_diffusion_muon_XXXX/ckpt_best.pt \
#     BATCHES=runs/eval_ch_sweep_seed0.pt K=100 SNR=20 sbatch slurm/diagnose_diffsteps.sbatch
set -euo pipefail
ml purge
module load Python/3.11.5-GCCcore-13.2.0
module load LLVM/16.0.6-GCCcore-13.2.0
export DRJIT_LIBLLVM_PATH="$(ls "$EBROOTLLVM"/lib/libLLVM*.so* 2>/dev/null | head -n1)"
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "${VENV_DIR:-$HOME/MF_CSI_Prediction/venv}/bin/activate"
export TF_CPP_MIN_LOG_LEVEL=3

: "${DIFF_MU:?set DIFF_MU=path/to/ckpt_best.pt}"
K="${K:-64}"; SNR="${SNR:-20}"; SEED="${SEED:-0}"   # lower K or trim ETAS if a cell is slow
STEPS_LIST="${STEPS_LIST:-1,3,10,20,50}"
INITS="${INITS:-zeros,gaussian}"; SPACINGS="${SPACINGS:-leading,trailing}"; ETAS="${ETAS:-0.0,1.0}"
USE_MU="${USE_MU:-1}"
BATCHES="${BATCHES:-runs/eval_ch_sweep_seed0.pt}"
GS="${GEN_SIZE:+--gen-size ${GEN_SIZE}}"
PT="${PRED_TYPE:+--prediction-type ${PRED_TYPE}}"

ts=$(date +%y%m%d_%H%M%S); OUT="runs/diffdiag_snr${SNR}_${ts}"; LOG="logs/diffdiag_snr${SNR}_${ts}.txt"
if [ ! -f "$BATCHES" ]; then
  echo "WARN: $BATCHES not found; generating a fresh paired set into it." | tee "$LOG"
  BATCH_ARG="--save-batches $BATCHES"
else
  BATCH_ARG="--load-batches $BATCHES"
fi
echo "===== DIFF-STEPS DIAGNOSIS STARTED $(date) =====" | tee -a "$LOG"
python -m scripts.diagnose_diffsteps --diff-mu "$DIFF_MU" --use-mu "$USE_MU" \
    --steps-list "$STEPS_LIST" --inits "$INITS" --spacings "$SPACINGS" --etas "$ETAS" \
    --K "$K" --snr "$SNR" --seed "$SEED" --out-dir "$OUT" $BATCH_ARG $GS $PT >> "$LOG" 2>&1
echo "===== FINISHED $(date) =====" | tee -a "$LOG"
echo "log: $LOG"
