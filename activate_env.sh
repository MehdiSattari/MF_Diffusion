# Source this (do NOT execute) to use the venv interactively on Alvis:
#     source activate_env.sh
#
# It loads the Python module FIRST (so libpython3.11.so.1.0 is found) and then
# activates the venv. Forgetting the module load is the usual cause of
# "error while loading shared libraries: libpython3.11.so.1.0".

ml purge
module load Python/3.11.5-GCCcore-13.2.0
# Sionna 0.19 imports its ray-tracing module on `import sionna`, which needs
# Mitsuba's LLVM backend (libLLVM.so). We don't use ray tracing, but must
# satisfy the import. Adjust the LLVM version if `ml spider LLVM` shows another.
module load LLVM/16.0.6-GCCcore-13.2.0
export DRJIT_LIBLLVM_PATH="$(ls "$EBROOTLLVM"/lib/libLLVM*.so* 2>/dev/null | head -n1)"
source $HOME/MF_CSI_Prediction/venv/bin/activate
echo "venv active: $(which python)"
echo "libLLVM: ${DRJIT_LIBLLVM_PATH:-NOT FOUND}"
