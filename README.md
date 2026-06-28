# Charge Light Matching Alpha Release

The alpha release pipeline for DUNE charge-light matching, covering **two detectors** (ND-LAr and the 2x2 demonstrator) and **simulation + data**.

* **ND-LAr** — end-to-end charge-light matching: front-stage track/shower placement, Phase 2 large-cluster scan, V2 light rescue (the `phase25_trial2_v_alpha_test` module), Phase 3 small-cluster matrix association. Perceiver weights (~490 MB) ship as a GitHub Release asset; variance prediction is optional (constant-std fallback). This is the version integrated into flow.
* **2x2** — a port of the same algorithm to the 2x2 geometry/light system, living under [`TwoByTwo/`](TwoByTwo/), with its own perceiver (sim **and** data) and matcher. All 2x2 models are **small (~3.5–6 MB) and committed in-repo**, so the 2x2 workflows are grab-and-run with no separate download.

Both detectors emit the **same per-file `.pt` schema** (`calib_hit_t0_reco` etc., documented in [`config.yaml`](config.yaml)).

## Four workflows (sim/data × ND/2x2)

There are four `scripts/run_<det>_<kind>.sh` entry points. Three are runnable today; ND-data has no algorithm yet and is an intentionally-empty placeholder.

| workflow | script | status | models |
|---|---|---|---|
| ND simulation | `scripts/run_nd_sim.sh` | ✅ runnable | perceiver = Release asset (download once) |
| ND data | `scripts/run_nd_data.sh` | ⬜ empty placeholder (no ND-data pipeline yet) | — |
| 2x2 simulation | `scripts/run_2x2_sim.sh` | ✅ runnable | bundled in-repo |
| 2x2 data | `scripts/run_2x2_data.sh` | ✅ runnable | bundled in-repo |

### 2x2 algorithm versions (`VERSION=`)

The **default is the error-matrix formulation** — the validated, conservative baseline. A newer region-grow method also ships, selectable per-run, but is **not** the default:

| `VERSION` | name | what it does |
|---|---|---|
| `v1.0` *(default)* | **error-matrix** | greedy per-TPC brightest-first small-cluster association, unit-variance χ² (the ND vAlpha formulation). |
| `v2.0` | **region-grow + tiebreaker** | adds cluster-guided spatial region-growing (confident light-matched clusters propagate t0 to neighbours; tuned conf_cos=0.55, light_margin=0.04) plus a learned-variance tiebreaker for ambiguous t0 candidates. Higher efficiency on the sim validation set (~+0.7 pp overall); the variance tiebreaker itself is roughly neutral. |

```bash
bash scripts/run_2x2_sim.sh                 # v1.0 error-matrix (default)
VERSION=v2.0 bash scripts/run_2x2_sim.sh    # opt into region-grow
```

## 2x2 quick start (grab-and-run)

```bash
git clone https://github.com/MadivB/CLMatching_AlphaRelease.git
cd CLMatching_AlphaRelease
python scripts/check_install.py             # 2x2 assets are bundled -> all OK, no download

# On a 4-GPU interactive node (8 workers, auto-aggregates per-event -> per-file .pt):
salloc -A dune -q interactive -C gpu --gpus-per-node=4 -N 1 -t 30 \
  srun -N1 -n1 --gpus-per-node=4 bash scripts/run_2x2_sim.sh      # or run_2x2_data.sh

# Result (per input FLOW file):
#   output/2x2_sim_v1.0/pt_outputs/<basename>.qlmatch2x2.pt
```

Process a specific file (sim or data) by passing it positionally or via `FILE=`:

```bash
bash scripts/run_2x2_sim.sh  /path/to/MiniRun6.4_1E19_RHC.flow.0000123.FLOW.hdf5
FILE=/path/to/packet-XXXX.FLOW.hdf5 bash scripts/run_2x2_data.sh
```

The 2x2 layout (matcher + perceiver + bundled models + driver) lives under [`TwoByTwo/`](TwoByTwo/); see [`TwoByTwo/README_2x2.md`](TwoByTwo/README_2x2.md) for the package internals.

---

## ND-LAr (the original alpha release)

Everything below documents the ND-LAr workflow. Its perceiver weights (~490 MB) are too big to commit to git and ship instead as a GitHub Release asset. The variance-prediction model is optional (the pipeline runs with a constant-std fallback when absent).

## Quick start (any machine, any path)

```bash
# 1. Pick where you want the install to live, then clone.
INSTALL_DIR=/path/to/where/you/want/it
mkdir -p "$INSTALL_DIR" && cd "$INSTALL_DIR"
git clone https://github.com/MadivB/CLMatching_AlphaRelease.git
cd v_alpha_test

# 2. Preflight check (no GPU needed): tells you exactly what's missing
#    and how to fix it.
python scripts/check_install.py
#    Expected on a fresh clone: perceiver MISSING (required), pulse OK,
#                               variance optional.  Exit code 1.

# 3. Download the perceiver weights (~490 MB; ~30 s on a fast network)
#    into the path paths.yaml expects:
mkdir -p NewMLSection/runs/ndfull_run_distributed
curl -L -o NewMLSection/runs/ndfull_run_distributed/checkpoint.pt \
  https://github.com/MadivB/CLMatching_AlphaRelease/releases/download/v0.1.0/checkpoint.pt

# 4. Verify the SHA matches release.yaml (paranoia check; recommended).
sha256sum NewMLSection/runs/ndfull_run_distributed/checkpoint.pt
#    Expected:
#    38655cca2b50f2caa643ef572fb80c77332611eafd3a831215cbe0f117473ac5  ...

# 5. Re-verify the install (should now be all green).
python scripts/check_install.py
#    Expected: "All required assets are present.", exit code 0.

# 6. Optional: edit paths.yaml if any of your assets/data live somewhere
#    other than the defaults.  paths.yaml is the single source of truth.

# 7. Run the single-file smoke test on a GPU node (assuming that you are on nersc) (~6-10 min wall clock,
#    8 workers across 4 GPUs, auto-aggregates per-event NPZ -> per-file .pt).
salloc -A dune -q interactive -C gpu --gpus-per-node=4 -N 1 -t 30 \
  srun -N1 -n1 --gpus-per-node=4 \
    bash scripts/run_v_alpha_test_pt_one_file.sh 

# 8. Inspect the result.
python scripts/inspect_pt.py output/test_one_file/pt_outputs/*.v_alpha_test.pt
```

The launcher writes outputs to `output/test_one_file/` inside your clone (override with `OUT_DIR=...`).
Expected coverage on the default test file: ~98.7% of prompt hits get a finite t0 in `calib_hit_t0_reco`.

If `check_install.py` reports a missing required asset, it prints the exact path it tried, the download URL, the copy-pasteable download command, and the `paths.yaml` key to edit.

### Watch progress (separate terminal)

After the `salloc` lands, in another login shell:
```bash
cd "$INSTALL_DIR"/v_alpha_test    # same install dir as above
tail -f output/test_one_file/parallel8_logs/worker*.log
```

### Alternative: 10-file production run (~30-50 min wall)

```bash
salloc -A dune -q interactive -C gpu --gpus-per-node=4 -N 1 -t 90 \
  srun -N1 -n1 --gpus-per-node=4 \
    bash scripts/run_v_alpha_test_pt_parallel8.sh
```

## Three run modes

The pipeline can be driven three ways, all sharing the same engine and output schema:

| # | mode | script | when to use |
|---|---|---|---|
| 1 | **batch submission** | `scripts/submit_production_robust.sh [N]` | mass production; launches N preemption-robust SLURM chains that self-resubmit and cooperate via atomic file claims |
| 2 | **interactive folder** | `scripts/run_interactive_forward_0000000.sh` | run inside an existing `salloc` GPU node; processes a whole folder forward, cooperating with any batch chains |
| 3 | **single file** | `scripts/process_one_flow_file.sh <flow.hdf5> [out_dir]` | run inside an existing `salloc` GPU node; process exactly one FLOW file |

All three use 8 workers (2 per GPU × 4 GPUs) and auto-aggregate per-event NPZ shards into one per-file `.pt`.

Example for mode 3 (already on an interactive GPU node):

```bash
bash scripts/process_one_flow_file.sh \
  /global/cfs/cdirs/dunepro/people/abooth/nd-production/output/MiniProdN5/run-ndlar-flow/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift/FLOW/0000000/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000123.FLOW.hdf5
# -> output/single/<basename>/pt_outputs/<basename>.v_alpha_test.pt
```

## paths.yaml — single source of truth for external paths

Every external file the pipeline loads is listed in [`paths.yaml`](paths.yaml):

| asset | required? | default location |
|---|---|---|
| `perceiver_charge_light_relation` | **yes** | `NewMLSection/runs/ndfull_run_distributed/checkpoint.pt` (download from GitHub Release) |
| `pulse_template` | **yes** | `assets/avg_pulse.npy` (bundled, ~4 KB) |
| `variance_prediction` | optional | `NewMLSection/var_prediction/runs/.../best_model.pt` (constant-std fallback if missing) |
| `input_data.default_data_dir` | optional | NERSC default; override via CLI or paths.yaml |

Each `path:` can be absolute or repo-relative. `path_candidates:` lets you list multiple fallbacks.

You can also point the resolver at a different YAML via `V_ALPHA_TEST_PATHS_YAML=/path/to/your.yaml`.

## Layout

```
v_alpha_test/
├── README.md                                    # this file
├── paths.yaml                                   # USER-EDITABLE asset paths (perceiver, pulse, variance)
├── config.yaml                                  # per-file .pt output schema + field provenance
├── release.yaml                                 # release manifest (sha256s, asset URLs, distribution)
├── assets/
│   └── avg_pulse.npy                            # bundled pulse template (4 KB)
├── M5p1/                                        # M5p1 python package (front stage + V2 + Phase 3 + resolver)
│   └── first_stage_matching/
│       └── asset_resolver.py                    # reads paths.yaml, validates, friendly errors
├── NewMLSection/                                # perceiver model code (weights downloaded separately)
└── scripts/
    ├── check_install.py                         # validates paths.yaml; exits 1 on missing required assets
    ├── aggregate_to_pt.py                       # per-event NPZ shards -> per-file .pt
    ├── inspect_pt.py                            # peek at a per-file .pt
    ├── run_v_alpha_test_pt_one_file.sh          # 8-worker single-file launcher (auto-aggregates)
    └── run_v_alpha_test_pt_parallel8.sh         # 8-worker 10-file launcher (auto-aggregates)
```

The launcher scripts auto-detect the repo location from their own path — they work from any clone, no editing needed.

## Output: per-file `.pt` schema (vBeta3-compatible + new field)

See [`config.yaml`](config.yaml) for the full schema. Highlights:

**Per-prompt-hit fields** (size `n_calib_hits`):

| field | dtype | filled by |
|---|---|---|
| `calib_hit_t0_reco` | float32 | full pipeline (Front + Phase 2 + V2 + Phase 3); `hit_timestamps_post_phase3` scattered via `event.hit_refs` |
| `prompt_hit_t_cluster_id` | int16 | front-stage `labels_global` re-labeled by every V2 spatial+light move (each move yields a brand-new id past the original cluster count) |

**Per-merged-hit fields** (size `n_calib_final_hits`, vBeta3-compatible):

| field | dtype | filled by |
|---|---|---|
| `calib_final_hit_t0_reco` | float32 | aggregator: `calib_hit_t0_reco[prompt_idx[i]]` where `prompt_idx = charge/calib_prompt_hits/ref/charge/calib_final_hits/ref[:, 0]` |
| `calib_final_hit_cluster_id` | int16 | aggregator: same prompt-index lookup against `prompt_hit_t_cluster_id` |
| `calib_final_hit_prompt_index` | int64 | aggregator: the column-0 ref above |

**Counts + metadata:**

| field | type | filled by |
|---|---|---|
| `n_calib_hits`, `n_assigned`, `n_unassigned` | int | aggregator |
| `n_calib_final_hits`, `n_calib_final_assigned`, `n_calib_final_unassigned` | int | aggregator |
| `processed_event_ids`, `all_event_ids` | int64 | aggregator |
| `event_summaries`, `failed_events` | list[dict] | aggregator |
| `version`, `algorithm`, `input_file`, `calib_final_hit_source` | str | aggregator |

**Sentinels:** unassigned prompt and merged hits have `*_t0_reco = -1.0` and `*_cluster_id = -1`.

## Inspect a result

```bash
python scripts/inspect_pt.py output/test_one_file/pt_outputs/*.v_alpha_test.pt
```

## Manual aggregation

If you ran the batch but didn't auto-aggregate, run the aggregator separately:

```bash
python scripts/aggregate_to_pt.py \
    --shard-dir output/test_one_file \
    --output-dir output/test_one_file/pt_outputs
```

## On NERSC

The default paths in [`paths.yaml`](paths.yaml) and the launchers are NERSC-friendly out of the box. Run on a 4-GPU GPU-node interactive allocation:

```bash
salloc -A dune -q interactive -C gpu --gpus-per-node=4 -N 1 -t 30 \
  srun -N1 -n1 --gpus-per-node=4 \
    bash scripts/run_v_alpha_test_pt_one_file.sh
```

For 10 files / ~130 events in ~30 min:

```bash
salloc -A dune -q interactive -C gpu --gpus-per-node=4 -N 1 -t 90 \
  srun -N1 -n1 --gpus-per-node=4 \
    bash scripts/run_v_alpha_test_pt_parallel8.sh
```

## Excluded from this folder (too large for git)

- The perceiver charge-light relation weights (`checkpoint.pt`, ~490 MB) — GitHub Release asset
- The variance-prediction `.pt` (when produced) — GitHub Release asset, optional

Both are loaded by `M5p1.first_stage_matching.load_first_stage_models` using the paths from `paths.yaml`. Missing required assets trigger a friendly error with download instructions.
