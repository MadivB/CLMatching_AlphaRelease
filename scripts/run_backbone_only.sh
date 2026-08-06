#!/bin/bash
# v0.1 "backbone only" quick-test launcher.
#
# Runs the ND-LAr pipeline through front stage + Phase 2 (large-cluster scan)
# + V2 light rescue, then STOPS -- the Phase 3 small-cluster error-matrix
# association (and any v0.1 post-pass) is skipped.  To cut GPU time further,
# perceiver image prediction is skipped for clusters that could only ever be
# matched by Phase 3: single-TPC, non-backbone clusters with E <= 50 MeV.
# Override the threshold with PREDICT_MIN_E_MEV -- it must stay <= 50 (the
# CLI enforces this): Phase 2 treats single-TPC clusters with E > 50 MeV as
# primaries, so a larger cut would silently drop genuine Phase-2 primaries.
#
# The per-event NPZ shards keep the full v_alpha_test schema, with
# hit_timestamps_post_phase3 == hit_timestamps_post_v2, so all inspection /
# aggregation tooling keeps working.  Aggregation is OFF by default because
# aggregate_to_pt.py auto-detects Mode-A flow files and writes t_0 /
# t_cluster_id back into the SOURCE HDF5 in-place -- do not let a quick test
# do that.  Opt back in with SKIP_AGGREGATE=0 (Mode-B files only).
#
# Completed events are skipped via their ok-JSONs (the driver's skip-existing
# filter), and the shard tag does not encode the flags -- so each threshold
# gets its own default OUT_DIR (output/backbone_only_e50, _e30, _eall, ...).
# If you override OUT_DIR yourself, use a fresh directory per configuration,
# or force recomputation with EXTRA_ARGS="--no-skip-existing".
#
# Usage (same interface as run_v_alpha_test_pt_one_file.sh):
#   bash scripts/run_backbone_only.sh                      # default test file
#   bash scripts/run_backbone_only.sh /path/to/file.hdf5   # positional file
#   FILE=... N_GPUS=1 N_WORKERS_PER_GPU=1 bash scripts/run_backbone_only.sh
#   PREDICT_MIN_E_MEV=30 bash scripts/run_backbone_only.sh # lower cut
#   PREDICT_MIN_E_MEV="" bash scripts/run_backbone_only.sh # predict ALL clusters

set -euo pipefail

HERE=${HERE:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
export HERE

PREDICT_MIN_E_MEV=${PREDICT_MIN_E_MEV-50}
REFINE_INTERSECTIONS=${REFINE_INTERSECTIONS:-0}

BACKBONE_ARGS="--skip-phase3 --postpass none"
if [[ -n "$PREDICT_MIN_E_MEV" ]]; then
    BACKBONE_ARGS="$BACKBONE_ARGS --predict-min-energy-mev $PREDICT_MIN_E_MEV"
fi
if [[ "$REFINE_INTERSECTIONS" == "1" ]]; then
    BACKBONE_ARGS="$BACKBONE_ARGS --refine-intersections"
fi

# Backbone args first, user EXTRA_ARGS last: argparse last-wins, so explicit
# user flags override the defaults (and invalid combos fail loudly via the
# CLI guards instead of being silently clobbered).
export EXTRA_ARGS="$BACKBONE_ARGS ${EXTRA_ARGS:-}"
export SKIP_AGGREGATE=${SKIP_AGGREGATE:-1}
IR_SUFFIX=""
if [[ "$REFINE_INTERSECTIONS" == "1" ]]; then IR_SUFFIX="_ir"; fi
export OUT_DIR=${OUT_DIR:-"${HERE}/output/backbone_only_e${PREDICT_MIN_E_MEV:-all}${IR_SUFFIX}"}

echo "backbone-only mode: EXTRA_ARGS=$EXTRA_ARGS"
echo "outputs -> $OUT_DIR (SKIP_AGGREGATE=$SKIP_AGGREGATE)"

exec bash "${HERE}/scripts/run_v_alpha_test_pt_one_file.sh" "$@"
