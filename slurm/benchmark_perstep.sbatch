#!/bin/bash
#SBATCH -A your-slurm-account
#SBATCH -p alvis
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH -J perstep
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Per-step (B=1) inference-complexity benchmark on a T4.
#   sbatch slurm/benchmark_perstep.sbatch            # medium + full size sweep
#   NFE=1,3,10,20,50 sbatch slurm/benchmark_perstep.sbatch
set -euo pipefail
ml purge
module load Python/3.11.5-GCCcore-13.2.0
module load LLVM/16.0.6-GCCcore-13.2.0
export DRJIT_LIBLLVM_PATH="$(ls "$EBROOTLLVM"/lib/libLLVM*.so* 2>/dev/null | head -n1)"
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "${VENV_DIR:-$HOME/MF_CSI_Prediction/venv}/bin/activate"
export TF_CPP_MIN_LOG_LEVEL=3
NFE="${NFE:-1,3,10,20,50}"
ts=$(date +%y%m%d_%H%M); LOG="logs/perstep_${ts}.txt"
echo "===== PER-STEP BENCH STARTED $(date) =====" | tee "$LOG"
echo "----- medium (detailed) -----" | tee -a "$LOG"
python -m scripts.benchmark_perstep --gen-size medium --nfe-sweep "$NFE" --repeats 100 >> "$LOG" 2>&1
for s in xs small large xl; do
  echo "----- ${s} -----" | tee -a "$LOG"
  python -m scripts.benchmark_perstep --gen-size "$s" --nfe-sweep "$NFE" --repeats 60 >> "$LOG" 2>&1
done
echo "===== FINISHED $(date) =====" | tee -a "$LOG"
echo "log: $LOG"
