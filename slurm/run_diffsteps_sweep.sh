#!/bin/bash
# Does giving diffusion MORE DDIM steps fix its overconfidence?
#
# Runs the UQ eval at DDIM steps in {3,10,20,50}, all stochastic (eta=1), at 20 dB, on
# the SAME byte-identical channel set produced by the SNR sweep
# (runs/eval_ch_sweep_seed0.pt), so every diffusion-step count is paired and directly
# comparable to the fixed 1-NFE MeanFlow reference. Only the diffusion cells change with
# the step count; MeanFlow+mu is included as an invariant reference/consistency check.
#
# Usage (repo root on Alvis, after `git pull`):
#   bash slurm/run_diffsteps_sweep.sh
set -euo pipefail

BATCHES="${BATCHES:-runs/eval_ch_sweep_seed0.pt}"
if [ ! -f "$BATCHES" ]; then
  echo "ERROR: $BATCHES not found. Run slurm/run_snr_sweep.sh first (it saves the channels)."; exit 1
fi

export MF_MU="${MF_MU:-runs/mf_6913678/ckpt_best.pt}"
export DIFF_MU="${DIFF_MU:-runs/ar_diffusion_muon_6917023/ckpt_best.pt}"
export DIFF_NOMU="${DIFF_NOMU:-runs/ar_diffusion_muoff_6917024/ckpt_best.pt}"

SNR="${SNR:-20}"                     # set SNR=10 (or 5) for a rate-discriminating regime
ETA="${ETA:-1.0}"                    # 1.0 stochastic (calibration); 0.0 deterministic (accuracy-optimal)
for ds in 3 10 20 50; do
  jid=$(env K=30 SNR="$SNR" SEED=0 ETA="$ETA" DIFF_STEPS="$ds" LOAD_BATCHES="$BATCHES" \
        sbatch --parsable slurm/eval_uncertainty_2x2.sbatch)
  echo "submitted SNR=${SNR}dB eta=${ETA} DDIM steps=${ds} as $jid"
done
echo "done: 4 jobs queued (SNR=${SNR}dB, eta=${ETA}, DDIM steps 3/10/20/50, paired on $BATCHES)."
