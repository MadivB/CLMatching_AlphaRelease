#!/bin/bash
# Kick off the recurring production chain.  Call this once; the SBATCH job
# re-submits itself until all files in folder 0000000 are processed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/sbatch_production_0000000.sh"

# Make sure the log dir exists before sbatch tries to write to it.
PROD_DIR=${PROD_DIR:-/pscratch/sd/y/yuxuan/CLMatching_AlphaRelease_prod/0000000}
mkdir -p "$PROD_DIR/logs"

# Quick preflight: does the perceiver weight exist?  Use the repo's own
# validator -- non-zero exit means missing required assets.
# Use the same conda python the production pipeline uses (system python on
# the login node is 3.6 and can't parse our type hints).
PY=${PY:-/global/common/software/nersc/pe/conda-envs/26.1.0/python-3.13/nersc-python/bin/python}
echo "Preflight: checking install ..."
"$PY" "${SCRIPT_DIR}/check_install.py" || {
    echo
    echo "ERROR: required assets missing.  Fix the above before submitting."
    exit 2
}

# Submit the first instance.
echo
echo "Submitting initial job: $SBATCH_SCRIPT"
sbatch "$SBATCH_SCRIPT"
echo
echo "Monitor with:"
echo "  squeue --me -o '%.10i %.9P %.20j %.2t %.10M %.20S %R'"
echo "  bash ${SCRIPT_DIR}/monitor_production_0000000.sh"
