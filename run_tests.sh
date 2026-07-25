#!/bin/bash
# Fast CPU-only correctness tests (no Sionna, no GPU). Runs in seconds.
#
# On Alvis (recommended) -- on a LOGIN NODE inside the project venv:
#     module load Python/3.11.5-GCCcore-13.2.0
#     source $HOME/MF_CSI_Prediction/venv/bin/activate
#     bash run_tests.sh
#
# Locally (optional fallback env):
#     source .venv/bin/activate && bash run_tests.sh
set -euo pipefail
cd "$(dirname "$0")"
TESTS="test_unet test_encoder test_meanflow test_inference test_regression test_diffusion"
fail=0
for t in $TESTS; do
  echo "=== tests/$t ==="
  python -m tests."$t" || fail=1
done
if [ "$fail" = 0 ]; then echo ""; echo "ALL TESTS PASSED"; else echo ""; echo "SOME TESTS FAILED"; exit 1; fi
