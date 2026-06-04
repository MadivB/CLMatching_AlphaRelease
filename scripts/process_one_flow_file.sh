#!/bin/bash
# Process exactly ONE FLOW hdf5 file end-to-end and write a per-file .pt.
#
# Assumes you are ALREADY on an interactive GPU node (4 GPUs), e.g. after:
#   salloc -A dune -q interactive -C gpu --gpus-per-node=4 -N 1 -t 60
#
# This is the third of the repo's three run modes:
#   1. batch submission     -> scripts/submit_production_robust.sh
#   2. interactive folder   -> scripts/run_interactive_forward_0000000.sh
#   3. single file (THIS)   -> scripts/process_one_flow_file.sh <flow.hdf5> [out_dir]
#
# Usage:
#   bash scripts/process_one_flow_file.sh /path/to/SOMEFILE.FLOW.hdf5
#   bash scripts/process_one_flow_file.sh /path/to/SOMEFILE.FLOW.hdf5 /my/out/dir
#
# Output:
#   <out_dir>/pt_outputs/<basename>.v_alpha_test.pt   (the per-file result)
#   default out_dir = <repo>/output/single/<basename>

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO=${REPO:-"$HERE"}
PY=${PY:-/global/common/software/nersc/pe/conda-envs/26.1.0/python-3.13/nersc-python/bin/python}
LAUNCHER="${REPO}/scripts/run_v_alpha_test_pt_parallel8.sh"

# ---- Argument handling ----
FILE=${1:-}
if [[ -z "$FILE" ]]; then
    echo "usage: bash scripts/process_one_flow_file.sh <flow.hdf5> [out_dir]" >&2
    echo "  <flow.hdf5> : path to a single FLOW hdf5 file to process" >&2
    echo "  [out_dir]   : optional output directory (default <repo>/output/single/<basename>)" >&2
    exit 1
fi
if [[ ! -f "$FILE" ]]; then
    echo "ERROR: input file not found: $FILE" >&2
    exit 2
fi

BASENAME=$(basename "$FILE" .hdf5)
OUT_DIR=${2:-${REPO}/output/single/${BASENAME}}
PT_DIR="${OUT_DIR}/pt_outputs"

echo "================================================================"
echo "process_one_flow_file"
echo "  host    : $(hostname)"
echo "  file    : $FILE"
echo "  out_dir : $OUT_DIR"
echo "  pt out  : ${PT_DIR}/${BASENAME}.v_alpha_test.pt"
echo "================================================================"

# ---- Preflight: required assets present? ----
"$PY" "${REPO}/scripts/check_install.py" >/dev/null || {
    echo "ERROR: required assets missing. Run: $PY scripts/check_install.py" >&2
    exit 3
}

# ---- Run the canonical 8-worker engine on this single file (auto-aggregates) ----
OUT_DIR="$OUT_DIR" PT_DIR="$PT_DIR" LOG_DIR="${OUT_DIR}/worker_logs" \
    bash "$LAUNCHER" "$FILE"
rc=$?

# ---- Report result ----
PT="${PT_DIR}/${BASENAME}.v_alpha_test.pt"
echo "================================================================"
if [[ -f "$PT" ]]; then
    echo "SUCCESS: $PT"
    ls -la "$PT"
    echo
    echo "Inspect with:"
    echo "  $PY ${REPO}/scripts/inspect_pt.py $PT"
else
    echo "FAILED: no .pt produced (launcher rc=$rc). Check ${OUT_DIR}/worker_logs/"
    exit "${rc:-1}"
fi
echo "================================================================"
