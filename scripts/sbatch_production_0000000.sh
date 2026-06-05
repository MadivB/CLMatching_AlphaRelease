#!/bin/bash
#SBATCH -A dune
#SBATCH -q preempt
#SBATCH -C gpu
#SBATCH --gpus-per-node=4
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -J clmatch_prod_0000000
#SBATCH -o /pscratch/sd/y/yuxuan/CLMatching_AlphaRelease_prod/0000000/logs/job_%j.out
#SBATCH -e /pscratch/sd/y/yuxuan/CLMatching_AlphaRelease_prod/0000000/logs/job_%j.err
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@300
#
# Recurring preempt-QoS production job for folder 0000000 of MiniProdN5.
#
# - One file per parallel8 invocation (8 workers x 4 GPUs).
# - Files processed in REVERSE order (highest filenumber first).
# - Per-file resume-skip: if a file's .pt already exists, skip it.
# - Auto re-submits itself at the end so the chain keeps going until done.
# - Trap on SIGTERM/SIGUSR1 (preempt warning): finish current file, then exit
#   cleanly and re-submit.
#
# NOTE: this batch script currently assumes Mode B output (.pt files). New
# flow files whose calib_prompt_hits dtype reserves t_0/t_cluster_id are
# processed by process_one_flow_file.sh in Mode A (in-place HDF5 writeback)
# instead.  TODO: add Mode A support to the batch path.

set -uo pipefail

# IMPORTANT: SLURM copies the sbatch script to /var/spool/slurmd/<jobid>/
# on the compute node before running, so $BASH_SOURCE auto-detect resolves
# to /var/spool/slurmd (= wrong REPO) and every launcher invocation fails
# with rc=127.  Hardcode REPO + SCRIPT_PATH to absolute paths in the
# user's filesystem so the script always finds the launcher.
REPO=${REPO:-/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/CLMatching_AlphaRelease}
SCRIPT_PATH=${SCRIPT_PATH:-${REPO}/scripts/sbatch_production_0000000.sh}
SCRIPT_DIR="${REPO}/scripts"

DATA_DIR=${DATA_DIR:-/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000}
PROD_DIR=${PROD_DIR:-/pscratch/sd/y/yuxuan/CLMatching_AlphaRelease_prod/0000000}
LAUNCHER=${REPO}/scripts/run_v_alpha_test_pt_parallel8.sh

PT_OUT=$PROD_DIR/pt_outputs
WORK_BASE=$PROD_DIR/work
LOG_DIR=$PROD_DIR/logs
WORKER_LOG_DIR=$PROD_DIR/worker_logs
MANIFEST=$PROD_DIR/manifest.txt
mkdir -p "$PT_OUT" "$WORK_BASE" "$LOG_DIR" "$WORKER_LOG_DIR"

echo "================================================================"
echo "[$(date)] job=${SLURM_JOB_ID:-(local)}  node=$(hostname)"
echo "REPO     : $REPO"
echo "SCRIPT   : $SCRIPT_PATH"
echo "DATA_DIR : $DATA_DIR"
echo "PROD_DIR : $PROD_DIR"
nvidia-smi -L 2>/dev/null | head -8 || echo "no nvidia-smi"
echo "================================================================"

# ---- Trap preempt signal so we exit cleanly after current file ----
SHOULD_STOP=0
_on_signal() {
    SHOULD_STOP=1
    echo "[$(date)] preempt/term signal received; will finish current file then exit"
}
trap _on_signal TERM USR1

# ---- Build reverse-ordered file list ----
mapfile -t FILES < <(ls "$DATA_DIR"/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.*.FLOW.hdf5 2>/dev/null | sort -r)
N_TOTAL=${#FILES[@]}
N_DONE=$(ls "$PT_OUT"/*.v_alpha_test.pt 2>/dev/null | wc -l)
echo "[$(date)] backlog: $N_DONE done / $N_TOTAL total"

if [[ $N_TOTAL -eq 0 ]]; then
    echo "[$(date)] No files found in $DATA_DIR; not re-submitting."
    exit 0
fi

# ---- Process files in reverse order ----
N_DONE_THIS_JOB=0
for f in "${FILES[@]}"; do
    if [[ $SHOULD_STOP -eq 1 ]]; then
        echo "[$(date)] stopping before next file due to signal"
        break
    fi

    base=$(basename "$f" .hdf5)              # e.g. MiniProdN5p1_..._0000123.FLOW
    pt="$PT_OUT/${base}.v_alpha_test.pt"
    if [[ -f "$pt" ]]; then continue; fi

    # Isolated work dir per file so shards don't collide across files
    work="$WORK_BASE/$base"
    rm -rf "$work"
    mkdir -p "$work"

    echo "[$(date)] PROCESSING $base"
    SECONDS=0
    OUT_DIR="$work" PT_DIR="$work/pt" LOG_DIR="$work/worker_logs" \
        bash "$LAUNCHER" "$f"
    rc=$?
    echo "[$(date)] $base launcher rc=$rc  elapsed=${SECONDS}s"

    # Promote outputs + cleanup
    if [[ -f "$work/pt/${base}.v_alpha_test.pt" ]]; then
        mv "$work/pt/${base}.v_alpha_test.pt" "$PT_OUT/"
        if [[ -f "$work/pt/v_alpha_test_aggregator_summary.json" ]]; then
            mv "$work/pt/v_alpha_test_aggregator_summary.json" "$PT_OUT/${base}.aggregator_summary.json"
        fi
        # Save worker logs (compressed) for audit
        if [[ -d "$work/worker_logs" ]]; then
            tar -C "$work" -czf "$WORKER_LOG_DIR/${base}.worker_logs.tar.gz" worker_logs 2>/dev/null || true
        fi
        rm -rf "$work"
        N_DONE_THIS_JOB=$((N_DONE_THIS_JOB + 1))
        echo "$(date '+%Y-%m-%d %H:%M:%S') ${SLURM_JOB_ID:-local} ${base}" >> "$MANIFEST"
    else
        echo "[$(date)] WARN: $base produced no .pt; leaving $work for inspection"
    fi
done

# ---- Resubmit if not all done ----
N_DONE_FINAL=$(ls "$PT_OUT"/*.v_alpha_test.pt 2>/dev/null | wc -l)
echo "[$(date)] this job processed $N_DONE_THIS_JOB files; total done $N_DONE_FINAL / $N_TOTAL"

if [[ $N_DONE_FINAL -lt $N_TOTAL ]]; then
    echo "[$(date)] $((N_TOTAL - N_DONE_FINAL)) files remaining, re-submitting $SCRIPT_PATH"
    sbatch "$SCRIPT_PATH"
else
    echo "[$(date)] ALL $N_TOTAL files done!  Chain ends here."
fi
