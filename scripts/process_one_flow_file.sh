#!/bin/bash
# Process exactly ONE FLOW hdf5 file end-to-end.
#
# Assumes you are ALREADY on an interactive GPU node (4 GPUs), e.g. after:
#   salloc -A dune -q interactive -C gpu --gpus-per-node=4 -N 1 -t 60
#
# This script has two output modes, auto-detected from the source HDF5:
#   * Mode A (NEW): the flow file's calib_prompt_hits & calib_final_hits dtypes
#                   reserve t_0 and t_cluster_id fields. The QL matching outputs
#                   are written BACK INTO the HDF5 file in-place. No .pt produced.
#                   A WARNING is printed if either field had non-zero values prior.
#   * Mode B (legacy): older flow files without t_0/t_cluster_id fields. The
#                   QL matching outputs are written to a .pt under
#                   <repo>/output/QLmatchingvAlpha/<basename>.v_alpha_test.pt
#                   (default), or <out_dir>/pt_outputs/... if [out_dir] is given.
#
# This is the third of the repo's three run modes:
#   1. batch submission     -> scripts/submit_production_robust.sh
#   2. interactive folder   -> scripts/run_interactive_forward_0000000.sh
#   3. single file (THIS)   -> scripts/process_one_flow_file.sh <flow.hdf5> [out_dir]
#
# Usage:
#   bash scripts/process_one_flow_file.sh /path/to/SOMEFILE.FLOW.hdf5
#   bash scripts/process_one_flow_file.sh /path/to/SOMEFILE.FLOW.hdf5 /my/out/dir

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
    echo "  [out_dir]   : optional output directory" >&2
    echo "                  Mode A: overrides where shards/logs live; HDF5 is still written in-place." >&2
    echo "                  Mode B: <out_dir>/pt_outputs/<basename>.v_alpha_test.pt" >&2
    echo "                  default Mode A: <repo>/output/QLmatchingvAlpha/<basename>/" >&2
    echo "                  default Mode B: <repo>/output/QLmatchingvAlpha/<basename>/ (.pt sits in QLmatchingvAlpha/)" >&2
    exit 1
fi
if [[ ! -f "$FILE" ]]; then
    echo "ERROR: input file not found: $FILE" >&2
    exit 2
fi

BASENAME=$(basename "$FILE" .hdf5)

# ---- Detect mode from the source flow file's calib_prompt_hits dtype ----
MODE=$("$PY" - <<PY 2>/dev/null
import sys, h5py
try:
    with h5py.File("$FILE", "r") as h:
        for path in ("charge/calib_prompt_hits/data", "charge/calib_final_hits/data"):
            if path not in h or not all(f in (h[path].dtype.names or ()) for f in ("t_0","t_cluster_id")):
                print("B"); sys.exit(0)
        print("A")
except Exception:
    print("B")
PY
)
MODE=${MODE:-B}

QL_BASE="${REPO}/output/QLmatchingvAlpha"
mkdir -p "$QL_BASE"

# OUT_DIR (worker shards + logs) is always per-basename so files don't collide.
# PT_DIR (where .pt lands, Mode B only) is the flat QLmatchingvAlpha/ dir.
OUT_DIR=${2:-${QL_BASE}/${BASENAME}}
if [[ "$MODE" == "B" ]]; then
    # If user passed an explicit [out_dir], keep pt_outputs colocated under it
    # (matches the prior behavior); otherwise drop the .pt flat under QLmatchingvAlpha/.
    if [[ -n "${2:-}" ]]; then
        PT_DIR="${OUT_DIR}/pt_outputs"
    else
        PT_DIR="${QL_BASE}"
    fi
else
    # Mode A doesn't produce a .pt. Still pass a valid path to satisfy the
    # launcher; the aggregator will simply not write into it.
    PT_DIR="${OUT_DIR}/pt_outputs"
fi

echo "================================================================"
echo "process_one_flow_file"
echo "  host    : $(hostname)"
echo "  file    : $FILE"
echo "  mode    : $MODE"
if [[ "$MODE" == "A" ]]; then
    echo "  action  : write t_0 and t_cluster_id back into HDF5 in-place"
else
    echo "  out_dir : $OUT_DIR"
    echo "  pt out  : ${PT_DIR}/${BASENAME}.v_alpha_test.pt"
fi
echo "================================================================"

# ---- Preflight: required assets present? ----
"$PY" "${REPO}/scripts/check_install.py" >/dev/null || {
    echo "ERROR: required assets missing. Run: $PY scripts/check_install.py" >&2
    exit 3
}

# ---- Run the canonical 8-worker engine on this single file (auto-aggregates) ----
# Use `|| rc=$?` so the launcher's exit code is captured without triggering the
# outer `set -e`. Mode A vs Mode B success is judged below by inspecting the
# actual HDF5 (Mode A) or the produced .pt (Mode B), not by trusting rc alone.
rc=0
OUT_DIR="$OUT_DIR" PT_DIR="$PT_DIR" LOG_DIR="${OUT_DIR}/worker_logs" \
    bash "$LAUNCHER" "$FILE" || rc=$?

# ---- Report result ----
echo "================================================================"
if [[ "$MODE" == "A" ]]; then
    # Confirm the in-place fields are non-zero now.
    POPULATED=$("$PY" - <<PY 2>/dev/null
import h5py, numpy as np
with h5py.File("$FILE", "r") as h:
    p_nz = int((h["charge/calib_prompt_hits/data"]["t_cluster_id"][:] != 0).any() or (h["charge/calib_prompt_hits/data"]["t_0"][:] != 0).any())
    f_nz = int((h["charge/calib_final_hits/data"]["t_cluster_id"][:] != 0).any() or (h["charge/calib_final_hits/data"]["t_0"][:] != 0).any())
print("1" if (p_nz or f_nz) else "0")
PY
)
    if [[ "$POPULATED" == "1" ]]; then
        echo "SUCCESS (Mode A, in-place HDF5): $FILE"
        echo "  t_0 and t_cluster_id were updated in calib_prompt_hits and calib_final_hits."
    else
        echo "FAILED: HDF5 t_0/t_cluster_id appear still zero. launcher rc=$rc; check ${OUT_DIR}/worker_logs/"
        exit "${rc:-1}"
    fi
else
    PT="${PT_DIR}/${BASENAME}.v_alpha_test.pt"
    if [[ -f "$PT" ]]; then
        echo "SUCCESS (Mode B, .pt): $PT"
        ls -la "$PT"
        echo
        echo "Inspect with:"
        echo "  $PY ${REPO}/scripts/inspect_pt.py $PT"
    else
        echo "FAILED: no .pt produced (launcher rc=$rc). Check ${OUT_DIR}/worker_logs/"
        exit "${rc:-1}"
    fi
fi
echo "================================================================"
