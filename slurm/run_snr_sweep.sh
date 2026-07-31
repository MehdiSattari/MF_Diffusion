#!/bin/bash
# Wide-posterior / low-SNR sweep for the 2x2 mu-ablation.
#
# Runs the paired UQ eval at SNR in {0,5,10,20} dB x DDIM eta in {1.0 stochastic, 0.0
# deterministic}, ALL on ONE byte-identical channel set (generated once by the first
# job, loaded by the rest via a Slurm afterok dependency). This makes every panel of
# the paper-style Fig. 2 (per-step NMSE at each SNR) and Fig. 3 (NMSE vs SNR) mutually
# consistent, and keeps eta=0 vs eta=1 paired at every SNR.
#
# Usage (from the repo root on Alvis, after `git pull`):
#   bash slurm/run_snr_sweep.sh
set -euo pipefail

CK_MF_MU="${MF_MU:-runs/mf_6913678/ckpt_best.pt}"
CK_MF_NOMU="${MF_NOMU:-runs/ar_meanflow_muoff_6917020/ckpt_best.pt}"
CK_DIFF_MU="${DIFF_MU:-runs/ar_diffusion_muon_6917023/ckpt_best.pt}"
CK_DIFF_NOMU="${DIFF_NOMU:-runs/ar_diffusion_muoff_6917024/ckpt_best.pt}"
CK_CONVLSTM="${CONVLSTM:-runs/arlstm_6916298/ckpt_best.pt}"
SEED="${SEED:-0}"
BATCHES="runs/eval_ch_sweep_seed${SEED}.pt"

export MF_MU="$CK_MF_MU" MF_NOMU="$CK_MF_NOMU" DIFF_MU="$CK_DIFF_MU" \
       DIFF_NOMU="$CK_DIFF_NOMU" CONVLSTM="$CK_CONVLSTM" SEED="$SEED"

SB=slurm/eval_uncertainty_2x2.sbatch

# First job: generate + save the shared channel set, run SNR=0 dB stochastic.
JID=$(env SNR=0  ETA=1.0 SAVE_BATCHES="$BATCHES" sbatch --parsable "$SB")
echo "submitted generator+0dB/stoch as $JID (saves $BATCHES)"

# All remaining SNR x eta combos load the same channels once the generator finishes.
for combo in "0 0.0" "5 1.0" "5 0.0" "10 1.0" "10 0.0" "20 1.0" "20 0.0"; do
  set -- $combo; snr=$1; eta=$2
  jid=$(env SNR="$snr" ETA="$eta" LOAD_BATCHES="$BATCHES" \
        sbatch --parsable --dependency=afterok:"$JID" "$SB")
  echo "submitted SNR=${snr}dB eta=${eta} as $jid (afterok:$JID)"
done
echo "done: 8 jobs queued (1 generator + 7 dependents)."
