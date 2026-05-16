#!/bin/bash
# Quick at-a-glance status for the production chain.

set -uo pipefail

PROD_DIR=${PROD_DIR:-/pscratch/sd/y/yuxuan/CLMatching_AlphaRelease_prod/0000000}
DATA_DIR=${DATA_DIR:-/global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000}

N_TOTAL=$(ls "$DATA_DIR"/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.*.FLOW.hdf5 2>/dev/null | wc -l)
N_DONE=$(ls "$PROD_DIR"/pt_outputs/*.v_alpha_test.pt 2>/dev/null | wc -l)
N_INFLIGHT=$(ls "$PROD_DIR"/work 2>/dev/null | wc -l)
N_LOGS=$(ls "$PROD_DIR"/logs/job_*.out 2>/dev/null | wc -l)

echo "==== Production status: $(date) ===="
echo "data dir : $DATA_DIR"
echo "out dir  : $PROD_DIR"
echo "progress : $N_DONE / $N_TOTAL files done  ($((100*N_DONE/N_TOTAL))% if total>0)"
echo "in-flight (current work dirs)  : $N_INFLIGHT"
echo "total log files (jobs spawned) : $N_LOGS"
echo
echo "==== Recent (last 3) finished files ===="
ls -lt "$PROD_DIR"/pt_outputs/*.v_alpha_test.pt 2>/dev/null | head -3 | awk '{print "  " $6, $7, $8, $9}'
echo
echo "==== Active jobs ===="
squeue --me -o "%.10i %.9P %.20j %.2t %.10M %.20S %R" 2>/dev/null | head -10
echo
echo "==== Latest log tail (10 lines) ===="
LATEST=$(ls -t "$PROD_DIR"/logs/job_*.out 2>/dev/null | head -1)
if [[ -n "$LATEST" ]]; then
    echo "($LATEST)"
    tail -10 "$LATEST"
else
    echo "(no logs yet)"
fi
echo
echo "==== Disk usage ===="
du -sh "$PROD_DIR" 2>/dev/null || true
du -sh "$PROD_DIR"/pt_outputs 2>/dev/null || true
du -sh "$PROD_DIR"/work 2>/dev/null || true
