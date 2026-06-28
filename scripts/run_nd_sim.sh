#!/bin/bash
# ============================================================================
# ND-LAr charge-light matching — SIMULATION workflow.
#
# This is the uniform-named entry point for the ND simulation pipeline (one of
# the four sim/data x ND/2x2 workflows).  It simply forwards to the canonical
# ND launcher, which runs 8 workers (2 per GPU x 4 GPUs) over one FLOW file and
# aggregates per-event NPZ shards into one <basename>.v_alpha_test.pt.
#
# Run on a 4-GPU interactive node, e.g.:
#   salloc -A dune -q interactive -C gpu --gpus-per-node=4 -N 1 -t 30 \
#     srun -N1 -n1 --gpus-per-node=4 bash scripts/run_nd_sim.sh
#
# All env overrides of the underlying launcher apply (FILE=..., DATA_DIR=...,
# OUT_DIR=..., N_GPUS=...).  See scripts/run_v_alpha_test_pt_one_file.sh and
# the README "Three run modes" section for the full ND interface.
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${HERE}/run_v_alpha_test_pt_one_file.sh" "$@"
