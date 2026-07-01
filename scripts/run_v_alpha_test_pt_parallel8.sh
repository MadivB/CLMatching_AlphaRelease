#!/bin/bash
# v_alpha_test launcher: 8 workers (2 per GPU) on a single 4-GPU node, then
# aggregate the per-event NPZ shards into per-file .pt outputs in the
# vBeta3-compatible schema (see ../config.yaml).
#
# Usage:
#   bash run_v_alpha_test_pt_parallel8.sh                         # default 10-file test set
#   bash run_v_alpha_test_pt_parallel8.sh /path/to/file*.hdf5     # explicit files

set -euo pipefail

# Auto-detect the v_alpha_test repo root from the script's own location.
HERE=${HERE:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
REPO=${REPO:-"$HERE"}
PY=${PY:-/global/common/software/nersc/pe/conda-envs/26.1.0/python-3.13/nersc-python/bin/python}

# Default DATA_DIR comes from paths.yaml (input_data.default_data_dir).
if [[ -z "${DATA_DIR:-}" ]]; then
    DATA_DIR=$($PY -c "
import sys; sys.path.insert(0, '$REPO')
try:
    from M5p1.first_stage_matching.asset_resolver import resolve_input_data_dir
    print(resolve_input_data_dir(default='') or '')
except Exception:
    print('')
" 2>/dev/null)
fi
DATA_DIR=${DATA_DIR:-/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000}

OUT_DIR=${OUT_DIR:-${HERE}/output/test10_v_alpha_test}
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

if [[ $# -eq 0 ]]; then
    FILES=()
    for i in 1 2 3 4 5 6 7 8 9 10; do
        n=$(printf "%07d" "$i")
        FILES+=("${DATA_DIR}/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.${n}.FLOW.hdf5")
    done
else
    FILES=("$@")
fi

N_WORKERS=$((N_GPUS * N_WORKERS_PER_GPU))
echo "node=$(hostname); ${#FILES[@]} files; ${N_WORKERS} workers (${N_WORKERS_PER_GPU} per GPU x ${N_GPUS} GPUs)"
echo "shards -> ${OUT_DIR}"
echo "pt out -> ${PT_DIR}"
nvidia-smi -L 2>&1 | head -8 || echo "no nvidia-smi"
echo "files:"
for f in "${FILES[@]}"; do echo "  ${f##*/}"; done

# ---- Spawn N_WORKERS python workers, round-robin on the global event list ----
PIDS=()
for w in $(seq 0 $((N_WORKERS - 1))); do
    g=$((w % N_GPUS))
    LOG_FILE="${LOG_DIR}/worker${w}_gpu${g}.log"
    echo "worker ${w} -> GPU ${g}, log=${LOG_FILE}"
    (
        cd "$REPO"
        CUDA_VISIBLE_DEVICES="$g" \
            $PY -m M5p1.phase25_trial2_v_alpha_test \
                --files "${FILES[@]}" \
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

# ---- Wait for all workers ----
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAIL=$((FAIL + 1))
    fi
done
echo "all workers done; failures=${FAIL}"
ls "$OUT_DIR"/*.json 2>/dev/null | wc -l | xargs -I{} echo "{} JSONs in $OUT_DIR"

# ---- Aggregate per-event shards into per-file .pt outputs ----
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
    # Mode A files do not produce a .pt; suppress the failure so we do not
    # trip `set -o pipefail` and mislead callers into thinking the run failed.
    ls -la "$PT_DIR"/*.v_alpha_test.pt 2>/dev/null | head -20 || true
fi

exit $FAIL
