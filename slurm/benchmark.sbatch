#!/bin/bash
#SBATCH -A your-slurm-account
#SBATCH -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH -J mf_bench
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Efficiency benchmark (params / FLOPs / latency / memory / NFE). Submit:
#     sbatch slurm/benchmark.sbatch
# Options: BATCH=64 REPEATS=30 CPU=1 DIU_STEPS="1,2,3,5,10,20" sbatch slurm/benchmark.sbatch
set -euo pipefail

ml purge
module load Python/3.11.5-GCCcore-13.2.0
module load LLVM/16.0.6-GCCcore-13.2.0
export DRJIT_LIBLLVM_PATH="$(ls "$EBROOTLLVM"/lib/libLLVM*.so* 2>/dev/null | head -n1)"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs/bench
VENV_DIR="${VENV_DIR:-$HOME/MF_CSI_Prediction/venv}"
source "${VENV_DIR}/bin/activate"
export TF_CPP_MIN_LOG_LEVEL=3

BATCH="${BATCH:-64}"; REPEATS="${REPEATS:-30}"; DIU_STEPS="${DIU_STEPS:-1,2,3,5,10,20}"
EXTRA=""
[ -n "${CPU:-}" ] && EXTRA="${EXTRA} --cpu"
[ -n "${DIU_CKPT:-}" ] && EXTRA="${EXTRA} --diu-ckpt ${DIU_CKPT}"
[ -n "${MF_CKPT:-}" ] && EXTRA="${EXTRA} --mf-ckpt ${MF_CKPT}"

timestamp=$(date +%y%m%d_%H%M)
LOG="logs/bench_${timestamp}.txt"
echo "===== BENCHMARK STARTED AT $(date) =====" | tee "$LOG"
python -m scripts.benchmark_efficiency --batch "$BATCH" --repeats "$REPEATS" \
    --diu-steps "$DIU_STEPS" $EXTRA >> "$LOG" 2>&1
echo "===== BENCHMARK FINISHED AT $(date) =====" | tee -a "$LOG"
