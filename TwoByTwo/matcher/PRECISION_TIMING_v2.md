# 2x2 per-hit precision timing — v2.0_beta

Point-wise timing chain built on the matcher's family association,
reaching ~1 ns per-piece / per-hit times. Locked as **v2.0_beta**
(beta: the in-deposition velocity fit `v_in` is still being validated).

## Production timing model (locked)
* light propagation velocity **c_LAr FIXED** (data 7.90 cm/ns from the
  radon-alpha walk fit, `step1_constants_radon_walk.json`; sim studies
  use 8.0) — floating it is degenerate with walk and unstable.
* per-channel constants **k_j** (fit jointly, mean-zero gauge per TPC).
* empirical **walk ~ amplitude** (linear in kADC, MicroBooNE form).
  `walk~1/amp` is degenerate with the velocity term — never use it.
* particle propagation **v_in = c** within strong depositions;
  per-hit render `t_hit = t_start + s/c`.

Validated on sim vs truth: per-hit sigma 0.50-0.58 ns (flat vs distance
from the entrance over 128 cm); muon velocity refit gives c (29.9-30.7).

## Step 6 — family disintegration (`disintegrate_2x2.py`)
Splits one family into span-bounded PIECES without re-running the GPU:
* atomic unit = existing matcher cluster; a piece is a union of whole
  clusters, so its predicted light = SUM of member images.
* KD-tree local adjacency (LINK 10 cm) -> connected groups; isolated
  groups >= 30 cm from the main = DISPLACED (neutron candidates).
* span cap D_MAX 25 cm via O(1) bbox tests; oversized clusters split by
  PCA slices (thin tracks, aspect>=3 & transverse RMS<3 cm -> TRACK-SEG)
  or spatial k-means (fat showers -> SHOWER-SEG); split-cluster light is
  charge-apportioned (GPU re-prediction per segment is the accurate
  option and the only place GPU is re-needed).
* ~1-2 ms per family (median); strictly local in space AND time.

## Step 6.5 — piece timing + corrector (`piece_timing_2x2.py`)
* vertex time t_vtx from main-dominated channels; every MAIN/SATELLITE
  hit gets t_vtx + |hit-vtx|/c.
* each DISPLACED piece gets ONE free time shift (== v_out / TOF proxy),
  closed-form: weighted median of (onset_j - d_j/c_LAr) over
  piece-dominated channels. Validity: shift in [-160, +650] ns of the
  family flash and robust loss <= 4 ns.
* **corrector** (`--corrector`): a failed piece is re-fit against every
  OTHER candidate flash (reconstructed families + per-TPC flash SEEDS —
  seeds are essential: charge-poor pile-up never forms a family) via
  bounded waveform re-extraction. Frame convention: waveform sample =
  matching tick + 102 (pulse peak at +105, `geometry_2x2` docstring).
  A piece that fits another flash is re-assigned (family-membership
  audit by timing).

## Truth-scored metrics (80 sim events, 44,724 hits, MiniRun6.4)
| arm | acc(±1 tick) | precision (robust) |
|---|---|---|
| stage-F baseline | 0.982 | 2.64 ns |
| 6.5 no corrector | 0.985 | 1.09 ns |
| 6.5 + corrector  | 0.987 | 1.09 ns |
DISPLACED-only accuracy 0.614 -> 0.727 with the corrector (pile-up
impostor reassigned to its true flash, -462 ticks away).

Known beta limitations: (1) a mis-associated piece can evade the
corrector via a false low-loss in-window fit — the planned "trading"
criterion (compare in-window vs alternative-candidate loss for EVERY
piece) addresses this; (2) dim pieces (<3 usable channels) cannot be
fit and keep the family time, flagged; (3) v_in currently fixed at c —
fitting it per track-like chain is the v2.0 validation step.

Analysis scripts/figures: /pscratch/sd/y/yuxuan/2x2QLMatching/StartTimeFit/
