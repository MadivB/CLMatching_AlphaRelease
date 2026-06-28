#!/bin/bash
# ============================================================================
# ND-LAr charge-light matching — real DATA workflow.
#
# INTENTIONALLY EMPTY.
#
# There is no ND-LAr *data* charge-light-matching workflow yet: the ND data
# light-prediction model / pipeline does not exist at this release.  This file
# is a deliberate placeholder so the repo carries the full 2x2 grid of four
# workflows (sim/data x ND/2x2).  Of the four, three are runnable today:
#
#   scripts/run_nd_sim.sh     ND simulation         (runnable)
#   scripts/run_nd_data.sh    ND data               (THIS FILE — not available)
#   scripts/run_2x2_sim.sh    2x2 simulation        (runnable)
#   scripts/run_2x2_data.sh   2x2 data              (runnable)
#
# When an ND data workflow is developed, implement it here.
# ============================================================================
echo "[run_nd_data] No ND-LAr data charge-light-matching workflow exists yet."
echo "[run_nd_data] Runnable workflows: run_nd_sim.sh, run_2x2_sim.sh, run_2x2_data.sh"
exit 0
