#!/bin/bash
# Interactive FORWARD production loop for folder 0000000.
#
# Run this INSIDE an existing interactive allocation, e.g. after:
#   salloc -A dune -q interactive -C gpu --gpus-per-node=4 -N 1 -t 240
#
# It processes files low->high (forward), cooperating safely with the
# reverse-order batch chains (sbatch_production_robust.sh): both share the
# same pt_outputs/ and claim each file atomically with mkdir, so no file is
# ever processed twice -- the two directions simply meet in the middle.
#
# Stops cleanly ~MAX_MINUTES into the run so the last file finishes before
# the allocation expires.

set -uo pipefail

REPO=${REPO:-/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_AlphaRelease}
LAUNCHER=${REPO}/scripts/run_v_alpha_test_pt_parallel8.sh
DATA_DIR=${DATA_DIR:-/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000}
PROD_DIR=${PROD_DIR:-/pscratch/sd/y/yuxuan/CLMatching_AlphaRelease_prod/0000000}

# Stop launching new files after this many minutes (leave buffer before the
# 240-min alloc ends so the in-flight file completes; one file ~20 min).
MAX_MINUTES=${MAX_MINUTES:-225}

PT_OUT=$PROD_DIR/pt_outputs
WORK_BASE=$PROD_DIR/work
WORKER_LOG_DIR=$PROD_DIR/worker_logs
MANIFEST=$PROD_DIR/manifest_interactive.txt
mkdir -p "$PT_OUT" "$WORK_BASE" "$WORKER_LOG_DIR"

# On a Perlmutter interactive alloc you land on the compute node (nid*), where
# the launcher runs directly (same as the batch path).  If somehow on a login
# node, fall back to srun to step onto the allocated node.
if [[ "$(hostname)" == nid* ]]; then
    RUN=(bash "$LAUNCHER")
else
    RUN=(srun -N1 -n1 --gpus-per-node=4 bash "$LAUNCHER")
fi

echo "================================================================"
echo "[$(date)] INTERACTIVE FORWARD  host=$(hostname)  job=${SLURM_JOB_ID:-none}"
echo "REPO=$REPO"
echo "PROD_DIR=$PROD_DIR   MAX_MINUTES=$MAX_MINUTES"
echo "run cmd: ${RUN[*]} <file>"
nvidia-smi -L 2>/dev/null | head -8 || echo "no nvidia-smi"
echo "================================================================"

mapfile -t FILES < <(ls "$DATA_DIR"/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.*.FLOW.hdf5 2>/dev/null | sort)
N_TOTAL=${#FILES[@]}
[[ $N_TOTAL -eq 0 ]] && { echo "no input files found"; exit 1; }

START=$SECONDS
N_THIS=0
for f in "${FILES[@]}"; do
    mins=$(( (SECONDS - START) / 60 ))
    if [[ $mins -ge $MAX_MINUTES ]]; then
        echo "[$(date)] reached MAX_MINUTES=$MAX_MINUTES; stopping cleanly"
        break
    fi

    base=$(basename "$f" .hdf5)
    [[ -f "$PT_OUT/${base}.v_alpha_test.pt" ]] && continue

    work="$WORK_BASE/$base"
    # Atomic claim: fails if a batch chain (or another shell) already has it.
    if ! mkdir "$work" 2>/dev/null; then
        continue
    fi

    echo "[$(date)] [${mins}/${MAX_MINUTES}min] PROCESSING $base"
    SECONDS_F=$SECONDS
    OUT_DIR="$work" PT_DIR="$work/pt" LOG_DIR="$work/worker_logs" \
        "${RUN[@]}" "$f"
    rc=$?
    echo "[$(date)] $base launcher rc=$rc elapsed=$((SECONDS - SECONDS_F))s"

    if [[ -f "$work/pt/${base}.v_alpha_test.pt" ]]; then
        mv -f "$work/pt/${base}.v_alpha_test.pt" "$PT_OUT/"
        [[ -f "$work/pt/v_alpha_test_aggregator_summary.json" ]] && \
            mv -f "$work/pt/v_alpha_test_aggregator_summary.json" "$PT_OUT/${base}.aggregator_summary.json"
        [[ -d "$work/worker_logs" ]] && \
            tar -C "$work" -czf "$WORKER_LOG_DIR/${base}.worker_logs.tar.gz" worker_logs 2>/dev/null || true
        rm -rf "$work"
        N_THIS=$((N_THIS + 1))
        echo "$(date '+%Y-%m-%d %H:%M:%S') ${SLURM_JOB_ID:-interactive} ${base}" >> "$MANIFEST"
    else
        echo "[$(date)] WARN: $base produced no .pt (rc=$rc); leaving $work"
    fi
done

N_DONE=$(ls "$PT_OUT"/*.v_alpha_test.pt 2>/dev/null | wc -l)
echo "[$(date)] interactive forward done: processed $N_THIS this session; total $N_DONE/$N_TOTAL"
