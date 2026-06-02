#!/bin/bash
#SBATCH -A dune
#SBATCH -q preempt
#SBATCH -C gpu
#SBATCH --gpus-per-node=4
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -J clmatch_prod
#SBATCH -o /pscratch/sd/y/yuxuan/CLMatching_AlphaRelease_prod/0000000/logs/robust_%j.out
#SBATCH -e /pscratch/sd/y/yuxuan/CLMatching_AlphaRelease_prod/0000000/logs/robust_%j.err
#SBATCH --open-mode=append
#
# Preemption-ROBUST production job for folder 0000000 of MiniProdN5.
#
# Key robustness vs the old script:
#   1. Submits its OWN successor at the START via --dependency=afterany so a
#      preemption mid-file can never break the chain (the old script only
#      resubmitted at the END, after the grace window, so preempt killed it
#      before it could requeue).
#   2. Claims each file ATOMICALLY with mkdir, so any number of these jobs can
#      run in PARALLEL and never double-process a file.
#   3. Reclaims STALE work dirs (claimed but idle > STALE_MIN minutes with no
#      .pt) left behind by previously-preempted jobs.
#
# Launch several of these at once (see submit_production_robust.sh) to finish
# faster -- they cooperate through the shared pt_outputs/ + atomic claims.

set -uo pipefail

# SLURM copies this script to /var/spool/slurmd/<jobid>/, so $BASH_SOURCE is
# useless on the compute node.  Hardcode absolute repo paths.
REPO=${REPO:-/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_AlphaRelease}
SCRIPT_PATH=${SCRIPT_PATH:-${REPO}/scripts/sbatch_production_robust.sh}
LAUNCHER=${REPO}/scripts/run_v_alpha_test_pt_parallel8.sh

DATA_DIR=${DATA_DIR:-/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000}
PROD_DIR=${PROD_DIR:-/pscratch/sd/y/yuxuan/CLMatching_AlphaRelease_prod/0000000}

# How long a claimed-but-idle work dir must sit before another job may reclaim
# it (minutes).  One file takes ~20 min; 90 min idle ==> the claimer is dead.
STALE_MIN=${STALE_MIN:-90}

PT_OUT=$PROD_DIR/pt_outputs
WORK_BASE=$PROD_DIR/work
LOG_DIR=$PROD_DIR/logs
WORKER_LOG_DIR=$PROD_DIR/worker_logs
MANIFEST=$PROD_DIR/manifest.txt
mkdir -p "$PT_OUT" "$WORK_BASE" "$LOG_DIR" "$WORKER_LOG_DIR"

JOB=${SLURM_JOB_ID:-local}
echo "================================================================"
echo "[$(date)] ROBUST job=$JOB  node=$(hostname)"
echo "REPO=$REPO  PROD_DIR=$PROD_DIR  STALE_MIN=$STALE_MIN"
nvidia-smi -L 2>/dev/null | head -8 || echo "no nvidia-smi"
echo "================================================================"

count_done() { ls "$PT_OUT"/*.v_alpha_test.pt 2>/dev/null | wc -l; }

mapfile -t FILES < <(ls "$DATA_DIR"/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.*.FLOW.hdf5 2>/dev/null | sort -r)
N_TOTAL=${#FILES[@]}
if [[ $N_TOTAL -eq 0 ]]; then
    echo "[$(date)] no input files found; exiting"
    exit 0
fi

# ---- 1) Submit successor NOW (before processing) so preempt can't break us ----
# Only if there's still work to do.  Dependency afterany => the successor runs
# after THIS job ends in ANY state (complete, preempted, failed, timeout).
N_DONE_START=$(count_done)
if [[ "$JOB" != "local" && $N_DONE_START -lt $N_TOTAL ]]; then
    succ=$(sbatch --parsable --dependency=afterany:"$JOB" "$SCRIPT_PATH" 2>/dev/null)
    echo "[$(date)] queued successor job $succ (afterany:$JOB)"
else
    echo "[$(date)] not queueing successor (done=$N_DONE_START/$N_TOTAL or local run)"
fi

# ---- 2) Reclaim stale work dirs (claimed but idle, no .pt) ----
reclaimed=0
for d in "$WORK_BASE"/*/; do
    [[ -d "$d" ]] || continue
    b=$(basename "$d")
    [[ -f "$PT_OUT/${b}.v_alpha_test.pt" ]] && { rm -rf "$d"; continue; }
    # If nothing inside was modified within STALE_MIN minutes, it's dead.
    if ! find "$d" -mmin -"$STALE_MIN" -type f 2>/dev/null | grep -q .; then
        rm -rf "$d" && reclaimed=$((reclaimed + 1))
    fi
done
[[ $reclaimed -gt 0 ]] && echo "[$(date)] reclaimed $reclaimed stale work dirs"

# ---- 3) Process files, claiming each atomically ----
N_THIS=0
for f in "${FILES[@]}"; do
    base=$(basename "$f" .hdf5)
    [[ -f "$PT_OUT/${base}.v_alpha_test.pt" ]] && continue

    work="$WORK_BASE/$base"
    # Atomic claim: mkdir fails if another job already claimed this file.
    if ! mkdir "$work" 2>/dev/null; then
        continue
    fi

    echo "[$(date)] [$JOB] PROCESSING $base"
    SECONDS=0
    OUT_DIR="$work" PT_DIR="$work/pt" LOG_DIR="$work/worker_logs" \
        bash "$LAUNCHER" "$f"
    rc=$?
    echo "[$(date)] [$JOB] $base launcher rc=$rc elapsed=${SECONDS}s"

    if [[ -f "$work/pt/${base}.v_alpha_test.pt" ]]; then
        mv -f "$work/pt/${base}.v_alpha_test.pt" "$PT_OUT/"
        [[ -f "$work/pt/v_alpha_test_aggregator_summary.json" ]] && \
            mv -f "$work/pt/v_alpha_test_aggregator_summary.json" "$PT_OUT/${base}.aggregator_summary.json"
        [[ -d "$work/worker_logs" ]] && \
            tar -C "$work" -czf "$WORKER_LOG_DIR/${base}.worker_logs.tar.gz" worker_logs 2>/dev/null || true
        rm -rf "$work"
        N_THIS=$((N_THIS + 1))
        echo "$(date '+%Y-%m-%d %H:%M:%S') $JOB ${base}" >> "$MANIFEST"
    else
        # Leave the work dir; stale-reclaim will retry it later if this job dies.
        echo "[$(date)] [$JOB] WARN: $base produced no .pt (rc=$rc); leaving $work"
    fi
done

echo "[$(date)] [$JOB] done: processed $N_THIS this job; total $(count_done)/$N_TOTAL"
