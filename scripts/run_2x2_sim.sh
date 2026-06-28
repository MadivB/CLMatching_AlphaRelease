#!/bin/bash
# ============================================================================
# 2x2 charge-light matching — SIMULATION workflow.
# 8 workers (2 per GPU x 4 GPUs) process the events of one FLOW file, then
# auto-aggregate per-event NPZ shards into one per-file .pt
# (<basename>.qlmatch2x2.pt) with the same schema as the ND release.
#
# Algorithm version (env VERSION):
#   v1.0  (DEFAULT) = error-matrix small-cluster association (greedy, unit-var).
#   v2.0            = region-grow + learned-variance tiebreaker.
#
# Run on a 4-GPU interactive node, e.g.:
#   salloc -A dune -q interactive -C gpu --gpus-per-node=4 -N 1 -t 30 \
#     srun -N1 -n1 --gpus-per-node=4 bash scripts/run_2x2_sim.sh
#
# Override file / version:
#   FILE=/path/to.FLOW.hdf5 VERSION=v2.0 bash scripts/run_2x2_sim.sh
#   bash scripts/run_2x2_sim.sh /path/to.FLOW.hdf5
# ============================================================================
set -euo pipefail

HERE=${HERE:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
PY=${PY:-/global/common/software/nersc/pe/conda-envs/26.1.0/python-3.13/nersc-python/bin/python}

MODE=sim
VERSION=${VERSION:-v1.0}
DATA_DIR=${DATA_DIR:-/global/cfs/cdirs/dune/www/data/2x2/simulation/productions/MiniRun6.4_1E19_RHC/MiniRun6.4_1E19_RHC.flow/FLOW/0000000}
FILE=${FILE:-${1:-${DATA_DIR}/MiniRun6.4_1E19_RHC.flow.0000000.FLOW.hdf5}}

OUT_DIR=${OUT_DIR:-${HERE}/output/2x2_sim_${VERSION}}
PT_DIR=${PT_DIR:-${OUT_DIR}/pt_outputs}
LOG_DIR=${LOG_DIR:-${OUT_DIR}/parallel8_logs}
N_GPUS=${N_GPUS:-4}
N_WORKERS_PER_GPU=${N_WORKERS_PER_GPU:-2}
SKIP_AGGREGATE=${SKIP_AGGREGATE:-0}

mkdir -p "$OUT_DIR" "$LOG_DIR" "$PT_DIR"
if [[ ! -f "$FILE" ]]; then echo "ERROR: file not found: $FILE" >&2; exit 2; fi

N_WORKERS=$((N_GPUS * N_WORKERS_PER_GPU))
echo "node=$(hostname); 2x2 SIM ${VERSION}; ${N_WORKERS} workers (${N_WORKERS_PER_GPU}/GPU x ${N_GPUS})"
echo "file:   ${FILE}"
echo "shards: ${OUT_DIR}"
echo "pt out: ${PT_DIR}"
nvidia-smi -L 2>&1 | head -8 || echo "no nvidia-smi"

PIDS=()
for w in $(seq 0 $((N_WORKERS - 1))); do
    g=$((w % N_GPUS))
    LOG_FILE="${LOG_DIR}/worker${w}_gpu${g}.log"
    (
        cd "$HERE"
        CUDA_VISIBLE_DEVICES="$g" \
            $PY TwoByTwo/run_2x2_worker.py \
                --files "$FILE" \
                --out-dir "$OUT_DIR" \
                --mode "$MODE" \
                --version "$VERSION" \
                --event-stride "$N_WORKERS" \
                --event-offset "$w" \
                --verbose
    ) > "$LOG_FILE" 2>&1 &
    PIDS+=("$!")
    echo "worker ${w} -> GPU ${g}, log=${LOG_FILE}"
done
echo "launched ${#PIDS[@]} workers; tail -f ${LOG_DIR}/worker*.log"

FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=$((FAIL + 1)); done
echo "all workers done; failures=${FAIL}"

if [[ "$SKIP_AGGREGATE" == "1" ]]; then
    echo "SKIP_AGGREGATE=1; not aggregating."
else
    echo "----- aggregating NPZ shards -> per-file .pt -----"
    $PY "${HERE}/TwoByTwo/aggregate_2x2_to_pt.py" \
        --shard-dir "$OUT_DIR" --output-dir "$PT_DIR" --overwrite \
        --algorithm "2x2 ${VERSION} ($([[ $VERSION == v2.0 ]] && echo 'region-grow + tiebreaker' || echo 'error-matrix'))" \
        2>&1 | tee -a "${LOG_DIR}/aggregate.log"
    ls -la "$PT_DIR"/*.qlmatch2x2.pt 2>&1 | head -5
fi
exit $FAIL
