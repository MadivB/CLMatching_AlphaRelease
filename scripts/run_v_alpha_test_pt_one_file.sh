#!/bin/bash
# Single-file v_alpha_test run: 8 workers (2 per GPU) processing one HDF5,
# then auto-aggregating into a per-file .pt.
#
# Default file = MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000001.FLOW.hdf5
#
# Override via:
#   FILE=/path/to/file.hdf5 bash run_v_alpha_test_pt_one_file.sh
# or as a positional arg.

set -euo pipefail

PY=/global/common/software/nersc/pe/conda-envs/26.1.0/python-3.13/nersc-python/bin/python
REPO=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/M5p1ReleaseVersion
HERE=/global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/v_alpha_test
DATA_DIR=/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000

FILE=${FILE:-${1:-${DATA_DIR}/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000001.FLOW.hdf5}}
OUT_DIR=${OUT_DIR:-/pscratch/sd/y/yuxuan/light_rescue_test/valpha_runs/test_one_file}
PT_DIR=${PT_DIR:-${OUT_DIR}/pt_outputs}
LOG_DIR=${LOG_DIR:-${OUT_DIR}/parallel8_logs}
N_OUTER_PASSES=${N_OUTER_PASSES:-1}
N_GPUS=${N_GPUS:-4}
N_WORKERS_PER_GPU=${N_WORKERS_PER_GPU:-2}
PREFETCH_DEPTH=${PREFETCH_DEPTH:-2}
EXTRA_ARGS=${EXTRA_ARGS:-}
SKIP_AGGREGATE=${SKIP_AGGREGATE:-0}
AGGREGATE_OVERWRITE=${AGGREGATE_OVERWRITE:-1}

mkdir -p "$OUT_DIR" "$LOG_DIR" "$PT_DIR"

if [[ ! -f "$FILE" ]]; then
    echo "ERROR: file not found: $FILE" >&2
    exit 2
fi

N_WORKERS=$((N_GPUS * N_WORKERS_PER_GPU))
echo "node=$(hostname); 1 file; ${N_WORKERS} workers (${N_WORKERS_PER_GPU} per GPU x ${N_GPUS} GPUs)"
echo "file:    ${FILE}"
echo "shards:  ${OUT_DIR}"
echo "pt out:  ${PT_DIR}"
nvidia-smi -L 2>&1 | head -8 || echo "no nvidia-smi"

PIDS=()
for w in $(seq 0 $((N_WORKERS - 1))); do
    g=$((w % N_GPUS))
    LOG_FILE="${LOG_DIR}/worker${w}_gpu${g}.log"
    echo "worker ${w} -> GPU ${g}, log=${LOG_FILE}"
    (
        cd "$REPO"
        CUDA_VISIBLE_DEVICES="$g" \
            $PY -m M5p1.phase25_trial2_v_alpha_test \
                --files "$FILE" \
                --out-dir "$OUT_DIR" \
                --max-events-per-file 0 \
                --n-outer-passes "$N_OUTER_PASSES" \
                --device-policy auto \
                --prefetch-depth "$PREFETCH_DEPTH" \
                --event-stride "$N_WORKERS" \
                --event-offset "$w" \
                --verbose \
                $EXTRA_ARGS
    ) > "$LOG_FILE" 2>&1 &
    PIDS+=("$!")
done
echo "launched ${#PIDS[@]} parallel workers; PIDs=${PIDS[*]}"
echo "tail logs with: tail -f ${LOG_DIR}/worker*.log"

FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAIL=$((FAIL + 1))
    fi
done
echo "all workers done; failures=${FAIL}"
ls "$OUT_DIR"/*.json 2>/dev/null | wc -l | xargs -I{} echo "{} JSONs in $OUT_DIR"

if [[ "$SKIP_AGGREGATE" == "1" ]]; then
    echo "SKIP_AGGREGATE=1 set; not running aggregator."
else
    echo "----- aggregating per-event NPZ shards into per-file .pt -----"
    AGG_ARGS=("--shard-dir" "$OUT_DIR" "--output-dir" "$PT_DIR")
    if [[ "$AGGREGATE_OVERWRITE" == "1" ]]; then
        AGG_ARGS+=("--overwrite")
    fi
    $PY "${HERE}/scripts/aggregate_to_pt.py" "${AGG_ARGS[@]}" 2>&1 | tee -a "${LOG_DIR}/aggregate.log"
    echo "----- aggregator done -----"
    ls -la "$PT_DIR"/*.v_alpha_test.pt 2>&1 | head -5
fi

exit $FAIL
