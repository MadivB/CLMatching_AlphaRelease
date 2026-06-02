#!/bin/bash
# Launch N parallel, preemption-robust production chains for folder 0000000.
# Each chain (sbatch_production_robust.sh) maintains its own afterany-successor,
# and all chains cooperate via the shared pt_outputs/ + atomic mkdir claims.
#
# Usage:
#   bash submit_production_robust.sh [N_CHAINS]
# Default N_CHAINS=6.

set -euo pipefail

N_CHAINS=${1:-6}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=${REPO:-/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_AlphaRelease}
SBATCH_SCRIPT="${SCRIPT_DIR}/sbatch_production_robust.sh"
PROD_DIR=${PROD_DIR:-/pscratch/sd/y/yuxuan/CLMatching_AlphaRelease_prod/0000000}
PY=${PY:-/global/common/software/nersc/pe/conda-envs/26.1.0/python-3.13/nersc-python/bin/python}
mkdir -p "$PROD_DIR/logs"

echo "Preflight: checking install ..."
"$PY" "${SCRIPT_DIR}/check_install.py" || {
    echo; echo "ERROR: required assets missing.  Fix before submitting."; exit 2
}

echo
echo "Launching $N_CHAINS robust chains ..."
for i in $(seq 1 "$N_CHAINS"); do
    jid=$(sbatch --parsable "$SBATCH_SCRIPT")
    echo "  chain $i -> job $jid"
done

echo
echo "Monitor with:"
echo "  squeue --me -o '%.10i %.9P %.16j %.2t %.10M %.12L %R'"
echo "  bash ${SCRIPT_DIR}/monitor_production_0000000.sh"
