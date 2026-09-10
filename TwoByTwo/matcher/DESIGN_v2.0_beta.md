# 2x2 Point-Wise Precision Timing — Design & Intuition (v2.0_beta, FINAL)

This document describes every design newly built for the per-hit
precision-timing chain ("point-wise fit", Steps 1–7), and — importantly —
the *intuition* behind each choice. v2.0_beta is the final locked version
(tag `v2.0_beta`, branch `2x2-per-hit-precision-assignment`).

The goal of the whole chain: starting from the matcher's charge–light
family association (~16 ns DAQ ticks), assign **every charge hit a time
good to ~1 ns**, so that displaced/invisible propagation (neutron TOF),
beam microstructure, and pile-up mis-associations all become measurable.

---

## 0. The one governing idea

Scintillation light travels at a fixed, calibratable speed
(`c_LAr`, the group velocity of 128 nm light in LAr), and charged
particles travel at essentially `c`. Therefore every photon's arrival
time at a SiPM is a *deterministic geometric function* of (a) when and
where it was emitted and (b) one per-channel electronics constant. If
the geometry is known from the charge image — which the TPC gives us in
mm detail — then the only unknowns are a handful of *times*. Fitting
those times against the ~50-channel light system over-constrains them
massively: that is where the sub-tick (sub-16 ns → ~1 ns) precision
comes from. Every step below is one instance of this idea applied to a
progressively harder topology (point source → track → arbitrary
interaction).

---

## 1. Step 1 — Calibration on point sources: `t = d/c_LAr + k_j`

**Design.** Fit the channel-onset time of isolated point-like flashes as
a linear function of the flash→SiPM distance `d`, with one global slope
(`1/c_LAr`), one constant `k_j` per channel (384 total), and one free
emission time per source. Sources: beam flashes first, then **radon
BiPo-212 alphas** as the precision sample. Robust iterated weighted
least squares (MAD trimming + ridge); constant-fraction (25 %-of-peak)
onsets with a mean-of-first-75-samples baseline; an **empirical walk
term linear in pulse amplitude** added as one extra column.

**Intuition.**
* A *point* source is the cleanest possible clock: every channel sees
  light from the *same* emission instant, so any spread in arrival times
  is purely propagation + electronics. A track cannot give you this
  (different parts emit at different times); an alpha (mm range) can.
* Radon alphas are ideal because they are **α-triggered** (drift-x known
  exactly) and they illuminate *both* walls simultaneously — the light
  crosses the drift volume at `c_LAr` in ~8 ns, so both walls constrain
  the same emission time. The distance lever arm (5–85 cm) is what
  determines `c_LAr`; the per-channel residual determines `k_j`.
* `k_j` exists because each SiPM+ASIC chain has its own cable length,
  shaping and threshold behavior. ArcLight vs LCM modules showed
  *opposite-sign* offsets in different TPCs — so a per-**type** constant
  is not enough; it must be per-**channel**. (Gauge: `k_j` mean-zero per
  TPC, since an overall constant is absorbed by the source time.)
* **Walk** is the classical discriminator effect: a dim pulse crosses a
  fixed fraction of its peak *later* than a bright one with the same
  start. Fitting residual-vs-amplitude at FIXED distance separates walk
  from geometry (amplitude and distance are correlated for fixed-energy
  alphas — the exclusive-bin scan taught us that ring-restricted fits
  are degenerate).
* Two negative results that shaped the design: (i) `t(d)` is **linear**
  — no quadratic term, i.e. no field-dependent photon "acceleration";
  the earlier "velocity varies with distance" appearance was a
  per-slice-constant degeneracy artifact. (ii) Floating `c_LAr`
  *together with* a `walk ~ 1/amp` term is degenerate (the fit ran to
  c_LAr ≈ 728 cm/ns) — hence the production rules below.

**Production rules locked here:** `c_LAr` is **FIXED** (7.90 cm/ns from
the radon walk fit on data; sim uses its own effective value), walk is
**`~amp` (linear in kADC)** and never `~1/amp`.

**MicroBooNE provenance of the walk term (and why c_LAr = 7.9, not
13.4):** the walk column is our adaptation of MicroBooNE's empirical
correction `T_Emp = a1*(propagation) + b1*N_Ph` (arXiv:2304.02076). We
carry the photon-count piece explicitly (amplitude as the N_Ph proxy);
the `a1*prop` piece is exactly degenerate with `1/c_LAr` once velocity
is fixed, so it is ABSORBED into the fitted effective `c_LAr` — which
is why our calibrated 7.90 cm/ns is an *effective* propagation constant
and differs from the textbook group velocity (~13.4 cm/ns). Mapping to
their formalism: T_os <-> k_j; b1*N_Ph <-> walk*amp; a1*prop <-> folded
into effective c_LAr; median-over-PMTs <-> robust weighted median over
channels.

**Two calibration paths for the constants (important):**
* **DATA:** radon alphas are the primary source (true point sources);
  beam muons provide the independent cross-check (k_j corr 0.60).
* **SIMULATION:** no radon alphas exist. There the constants are
  **self-calibrated on the muon sample itself**: the same joint solve,
  but each muon contributes ONE free `t_start` (nuisance) and `v = c`
  ties all its segments to that single clock — a muon is a point source
  with a known internal schedule. One muon constrains ~20-48 channels
  while adding one unknown, so shared k_j + walk stay over-determined.
  Fitted this way, sim walk is ~0.03 ns/kADC (essentially zero — the
  sim optical/electronics chain has no discriminator walk); the
  2.0 ns/kADC walk is a real-data phenomenon. Constants must be re-fit
  per dataset, never copied from sim to data or vice versa.

---

## 2. Step 3 — The point-wise muon fit (moving point source)

**Design.** Break each clean muon track into `N ≈ 10` charge-weighted
segments. Model the onset at channel `j` as

```
t_j = t_start + min over segments s of [ L1(s)/c + d(s,j)/c_LAr ] + k_j (+ walk_j)
```

with the muon speed fixed at `c`. Joint robust linear solve for
(`t_start` per muon, `k_j` per channel), iterating the segment
assignment (the argmin) to convergence.

**Intuition.**
* A muon is a **moving point source**: segment `s` emits at
  `t_start + L1(s)/c`. Each channel's *first* light is whichever segment
  minimizes total travel time `L1/c + L2/c_LAr` — a Fermat/least-time
  principle. Using the earliest arrival (rather than, say, the mean)
  matches what a constant-fraction onset physically measures: the
  leading edge is dominated by the least-time path.
* Fixing `v = c` is not an approximation to apologize for: beam/cosmic
  muons are ultra-relativistic, and every free parameter you *don't* fit
  is variance you don't pay. (Step 7 later quantifies this.)
* Segments, not hits: ten charge-weighted emission points capture the
  geometry to ~2 cm while keeping the argmin cheap and stable.
* The same joint fit re-derives `k_j` from muons alone; its correlation
  with the radon `k_j` (independent source class) is the closure test
  that the constants are real physics, not fit artifacts.

Split-half precision (fit on even channels vs odd channels): ~1.0–1.7 ns
per muon on real data — the first proof that ~1 ns event timing is
reachable with this detector.

---

## 3. Step 4 — Render every hit: `t_hit = t_start + s/c`

**Design.** Propagate the fitted start time along the track at `c`:
every charge hit at arclength `s` from the entry gets
`t_hit = t_start + s/c`.

**Intuition.**
* The charge image tells you *where* every deposit is; the light fit
  tells you *when the track started*; kinematics (`v=c`) connects them.
  There is nothing more to measure per hit — the whole track is timed by
  one number plus geometry.
* Validation against per-hit truth: σ = 0.50–0.58 ns, and — the key
  check — **flat versus distance from the entrance** over 128 cm.
  Intuition for the flatness: every hit inherits the same `t_start`
  error (the floor), and the *additional* error from propagation is only
  the track-direction/geometry error, which is tiny for a
  well-constrained straight line. If σ had grown with `s`, the v=c
  propagation model would have been wrong.
* Method ladder on the same truth sample (per-hit σ): flash group-t0
  1.44 ns → entry-time-only 0.93 ns → v=c render 0.58 ns. Each rung is
  "add one piece of correct physics".

---

## 4. Step 5 — Fit the muon's velocity (machinery test)

**Design.** Freeze *all* calibration (c_LAr, k_j, walk). Per muon, leave
exactly one free velocity `v` (plus the unavoidable `t_start`) and
minimize the robust onset loss over a `v` grid with parabolic
refinement.

**Intuition.**
* This is the dress rehearsal for neutrons: can the light system measure
  *how fast something moved* between deposits? For muons the answer is
  known (`c`), so it is a closed-book exam.
* With calibration frozen, the only gradient left in the channel onsets
  along the track is the particle's flight time `L1/v`. The fit recovers
  the median **29.9 cm/ns vs c = 29.98** — and interestingly beats the
  naive truth-slope estimate (26.7), because the least-time light model
  accounts for the propagation geometry that a plain dt/ds fit smears.
* Per-muon spread (±7–10 cm/ns at 60–130 cm spans) taught us the
  *lever-arm law*: velocity precision is set by (time across the chain)
  vs (onset scatter). This later explains everything about Step 7.

---

## 5. Step 6 — Family disintegration (arbitrary topologies)

**Design.** Decompose one matched family into span-bounded **pieces**:
1. **Atomic unit = existing matcher cluster.** A piece is a union of
   whole clusters, so its predicted light is the **sum of the member
   clusters' already-predicted images** — zero new GPU inference.
2. **Local connectivity** (one KD-tree neighbor query over family hits,
   gap < 10 cm) → connected groups; union-find components.
3. **Main group** = highest-energy component; a *separate* component
   ≥ 30 cm from the main centroid is tagged **DISPLACED** (neutron
   candidate); near ones are SATELLITE.
4. **Span cap** `D_MAX = 25 cm` per piece, enforced with O(1)
   bounding-box tests during greedy whole-cluster agglomeration.
5. **Oversized single clusters are split**: thin ones (aspect ≥ 3,
   transverse RMS < 3 cm) by PCA-axis slices (**TRACK-SEG**), fat ones
   by spatial k-means (**SHOWER-SEG**); the split light is
   charge-apportioned as a proxy, with per-segment GPU re-prediction as
   the accurate option — the *only* place GPU is ever re-needed.
6. Energy floors drop noise clusters (keep ≥ 3 % of family energy or
   ≥ 8 units; always keep the most energetic).

**Intuition.**
* Why pieces at all: the timing unit must be something with **one
  well-defined emission time**. A whole neutrino event does not have
  one; a ≤ 25 cm piece does to within `25 cm / c ≈ 0.8 ns` — matched to
  our resolution. The span cap *is* the timing coherence requirement.
* Why cluster-atomic: the perceiver's light predictions are per-cluster.
  Summing images of whole clusters is *exact* (superposition of light);
  splitting a cluster is the only operation that invalidates its image —
  so pieces never cross cluster boundaries except for tracks, where the
  cost is accepted and bounded (a handful of GPU segments per family,
  batched).
* Why connectivity for displaced tagging: a neutron travels invisibly —
  by construction its deposit is *spatially disconnected* from the
  vertex. The 30 cm displacement threshold is the minimum TOF baseline
  worth measuring (shorter baselines give hopeless velocity precision,
  and sub-30 cm satellites are usually vertex EM spray).
* Why track/shower discrimination by *thickness* not just aspect: an
  elongated EM cascade looks track-like in aspect ratio but is fat
  transversely; slicing it along a PCA axis would mix unrelated cascade
  branches into "segments". The truth audit (particles-per-piece) is
  what forced this: purity is the wrong metric for showers (a shower is
  legitimately 40 particles *at one time*), so pieces are judged by
  **time coherence**, which the audit confirmed at 0.2–0.5 ns even for
  mixed shower pieces.
* Compute discipline (ND-ready): everything is **local in space** (a
  cluster only ever sees hits within the linking radius; ~1–2 ms per
  family) and — crucially — **local in time**: every time operation
  is anchored to the family's flash window; nothing ever searches the
  full 16 µs axis.

---

## 6. Step 6.5 — Piece timing + the corrector

**Design.**
* **Vertex clock:** `t_vtx` fit from main-dominated channels using the
  same least-time arrival model; every MAIN/SATELLITE hit gets
  `t_vtx + |hit − vtx|/c`.
* **Displaced pieces get ONE free time shift each.** Closed form: the
  optimal shift is the weighted median of `(onset_j − d_j/c_LAr)` over
  piece-dominated channels — no scan needed. The shift *is* the TOF,
  and `v_out = baseline / shift` is just a re-parameterization.
* **Validity window** from physics: shift ∈ [−160, +650] ns of the
  family flash (neutron flight over ≤ ~1 m), robust loss ≤ 4 ns,
  ≥ 3 channels.
* **Corrector** (flag `--corrector`): a piece that fails the in-window
  fit is re-fit against every *other* candidate flash time — the
  reconstructed families **plus the per-TPC light flash seeds** — by
  bounded re-extraction of the piece's top predicted channels in
  `[T′−10, T′+45]` samples. If a consistent time is found, the piece is
  **re-assigned** (its hits re-timed; its family membership corrected).

**Intuition.**
* The free shift is the user-level insight that "fitting v_out is the
  same as allowing a time shift on that cluster": for a compact piece
  the geometry is frozen, so exactly one time parameter exists. Fit the
  shift; *report* it as TOF or velocity. The sign is free information:
  positive shift = deposit after the vertex (neutron flew out); negative
  = the "displaced" piece is actually the start.
* The corrector exists because **timing is an independent auditor of the
  spatial association**. The matcher assigns clusters to families by
  geometry and predicted light; pile-up that lands nearby fools it. But
  a mis-associated piece *cannot* fake its light-arrival time: the ev28
  case (deposit whose true time was −462 ticks / −7.4 µs from its
  assigned family) fit nothing in-window, matched a light-only flash
  seed exactly, and was reclaimed. Displaced-hit accuracy: 0.614 →
  0.727.
* Why flash **seeds** and not just families: charge-poor pile-up never
  forms a charge family, but its *light* is still found by the seed
  finder. The candidate list must come from the light, not the charge.
* Why "failure to fit in-window IS the flag": we never scan the time
  axis looking for where a piece belongs. We test a physical window; if
  nothing consistent is there, the piece is by definition foreign, and
  the candidate lookup is a fixed handful of discrete times. O(1), not
  O(axis).
* One hard-won mechanical fact: **waveform sample = matching tick
  + 102** (template peak at +105; CF onset ~3 ticks earlier). Any
  re-extraction that ignores this frame offset searches an empty window
  — this single constant was the difference between the corrector
  reassigning nothing and working.

**Truth-scored metrics (80 sim events, 44,724 hits):**

| arm | accuracy (±1 tick) | robust σ (correct hits) |
|---|---|---|
| stage-F baseline | 0.982 | 2.64 ns |
| 6.5, no corrector | 0.985 | 1.09 ns |
| 6.5 + corrector | 0.987 | 1.09 ns |

(±5-tick and ±1-tick accuracies are identical — errors are bimodal:
either sub-tick correct or pile-up-wrong by hundreds of ticks. There is
no "slightly wrong" population, which is itself a statement that the
model is right.)

---

## 7. Step 7 — Fitting v_in, and what the A/B taught us

**Design.** Group TRACK-SEG pieces of one parent cluster into a chain,
anchor its start at the end nearer the vertex (ν-vertex tracks travel
outward), and fit one free `v_in` per chain with the Step-5 machinery.
Then A/B-score per-hit renders: (α) vertex clock + v=c, (β) chain-local
t₀ + v=c, (γ) chain-local t₀ + fitted v.

**Results and intuition.**
* μ-tagged chains fit `v_in` median **28.3 cm/ns ≈ c** — the beta-exit
  criterion. Their truth slopes are all 30.0–31.0: genuinely
  relativistic, correctly measured.
* A large population **rails at v → ∞ — and that is physics**: EM
  fragments deposit *near-simultaneously* (photon transport at c in all
  directions), not sequentially along an axis. A muon's deposits are
  ordered in time along its path; a shower's are not. So the v_in fit is
  a **sequential-vs-simultaneous discriminant** (track vs EM) for free.
  (Refinement: fit slowness 1/v so the simultaneous limit is a finite
  point instead of a rail.)
* The A/B verdict (49,342 chain hits, β≈1 subset):
  α 0.96 ns → **β 0.64 ns** → γ 0.75 ns (robust σ; accuracy 100 %
  everywhere).
  - **Chain-local t₀ is the real gain** (0.96 → 0.64): it self-corrects
    the vertex-centroid position error instead of propagating it into
    every hit along the track.
  - **Fitting v hurts** (0.64 → 0.75) for relativistic tracks:
    measuring a quantity you already know exactly can only add its
    fit variance. `v = c` is *information*, not an assumption.
* Production recipe therefore: **render with chain-local t₀ + v = c;
  use the v_in fit only as a classifier** (v≈c sequential → track;
  rail → EM; v<c → slow-particle candidate, the one class where fitted-v
  rendering would beat v=c).

---

## 8. What the beam-data campaign established (context)

The full July-10 campaign (255 files, 11,972 horizontal muons, 4-GPU
node) reached per-muon σ = 1.62 ns on real data and found a genuine
64.9 % forward (beam rock-muon) excess — but **no 18.831 ns comb**:
the v10 reflow carries no RWM/RF-phase reference, so the readout-window
phase relative to the accelerator RF is unknown at the 16 ns tick level
and the comb averages flat. The timing resolution needed to see the
bunches exists; the *reference signal* does not (MicroBooNE's RWM is
exactly this ingredient). Sim (MiniRun6.4) cannot test it either: the
spill envelope is simulated but intra-spill interactions are
time-collapsed with no bunch comb.

---

## 9. Known limitations of v2.0_beta and exit items

1. **False in-window fits** can shield a mis-associated piece from the
   corrector (ev28 E47: 3 channels of leaked family light fit at loss
   0.36). Fix: the **trading criterion** — compare the in-window loss vs
   the best alternative-candidate loss for *every* displaced piece, not
   only failed ones, and swap when the alternative is decisively better.
2. **Dim pieces** (< 3 usable predicted channels) cannot be fit; they
   keep the family time, flagged.
3. **pdg truth-tagging** is currently corrupted by per-vertex traj-id
   collisions (use segments' `file_traj_id`); until fixed, the predicted
   slow-proton branch (v_in ≈ 15–22 cm/ns) is untested.
4. **Per-chain v_in precision** (±10 cm/ns at ≤ 62 cm span) needs
   cross-TPC chain stitching to double the lever arm.
5. The neutron **+TOF spectrum** on a neutron-enriched sample is the
   remaining physics demonstration: displaced pieces from real neutrons
   should populate distinctly positive shifts (tens of ns), separable
   from the prompt γ/EM satellites measured at −5…+2 ns.

## Constants of record

| quantity | value | provenance |
|---|---|---|
| c_LAr (data) | 7.90 ± 0.05 cm/ns | radon-α walk fit, `step1_constants_radon_walk.json` |
| walk | ~2.0 ns·kADC (data); `~amp` form only | radon fit; degeneracy study |
| k_j | per-channel, mean-zero/TPC | radon + muon joint fits (corr 0.60) |
| v_in (render) | c = 29.9792458 cm/ns, fixed | Step 5/7 validation |
| piece span cap | 25 cm (≈ 0.8 ns coherence) | Step 6 |
| displaced threshold | 30 cm | minimum useful TOF baseline |
| shift window | [−160, +650] ns vs family flash | neutron flight physics |
| frame offset | waveform sample = matching tick + 102 | geometry_2x2 convention + ev28 |

*Modules:* `disintegrate_2x2.py`, `piece_timing_2x2.py` (this repo);
analysis scripts and figures under
`/pscratch/sd/y/yuxuan/2x2QLMatching/StartTimeFit/`.
