# Charge Light Matching Alpha Release

The alpha release pipeline. **End-to-end NDLAr charge-light matching**: front-stage track/shower placement, Phase 2 large-cluster scan, V2 light rescue (the `phase25_trial2_v_alpha_test` module), Phase 3 small-cluster matrix association — and a per-file `.pt` output with the schema documented in [`config.yaml`](config.yaml).

This is intended to be the same version integrated into flow. Everything needed to reproduce a run lives in this folder **except** the perceiver charge-light-relation weights (~490 MB), which are too big to commit to git and ship instead as a GitHub Release asset. The variance-prediction model is optional (the pipeline runs with a constant-std fallback when absent).

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

| field | dtype | shape | filled by |
|---|---|---|---|
| `calib_hit_t0_reco` | float32 | `(n_calib_hits,)` | full pipeline (Front + Phase 2 + V2 + Phase 3); `hit_timestamps_post_phase3` scattered via `event.hit_refs` |
| **`prompt_hit_t_cluster_id`** | **int16** | **`(n_calib_hits,)`** | front-stage `labels_global` re-labeled by every V2 spatial+light move (each move yields a brand-new id past the original cluster count) |
| `n_calib_hits`, `n_assigned`, `n_unassigned` | int | scalar | aggregator |
| `processed_event_ids`, `all_event_ids` | int64 | varies | aggregator |
| `event_summaries`, `failed_events` | list[dict] | varies | aggregator |
| `version`, `algorithm`, `input_file` | str | scalar | aggregator |

**Sentinels:** unassigned prompt hits have `calib_hit_t0_reco = -1.0` and `prompt_hit_t_cluster_id = -1`.

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
