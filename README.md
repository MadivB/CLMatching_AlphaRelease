# v_alpha_test

The v_alpha "test" pipeline. **End-to-end NDLAr charge-light matching**: front-stage track/shower placement, Phase 2 large-cluster scan, V2 light rescue (the `phase25_trial2_v_alpha_test` module), Phase 3 small-cluster matrix association — and a per-file `.pt` output with the schema documented in [`config.yaml`](config.yaml).

This is intended to be the long-lived alpha entrypoint. Everything needed to reproduce a run lives in this folder *except* the two large model weight files (the perceiver charge-light relation and the variance prediction `.pt`), which are loaded from the existing repository locations via `M5p1.first_stage_matching.load_first_stage_models`.

## Layout

```
v_alpha_test/
├── README.md                  # this file
├── config.yaml                # paths + per-file .pt schema + field provenance
├── M5p1/                      # symlinks/copies of the M5p1 modules used by the pipeline
├── scripts/
│   ├── aggregate_to_pt.py     # per-event NPZ shards -> per-file .pt
│   ├── run_v_alpha_test_pt_parallel8.sh   # 8-worker GPU launcher (auto-aggregates)
│   └── inspect_pt.py          # quick CLI to peek at a .pt
└── output/                    # default landing pad for aggregated .pt files
```

## Quick start (4 GPUs, ≤90 min, 10 files)

```bash
salloc -A dune -q interactive -C gpu --gpus-per-node=4 -N 1 -t 90 \
  srun -N1 -n1 --gpus-per-node=4 \
    /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/v_alpha_test/scripts/run_v_alpha_test_pt_parallel8.sh
```

The launcher:
1. Spawns 8 python workers (2 per GPU) running `M5p1.phase25_trial2_v_alpha_test`.
2. Each worker writes per-event `.npz` + `.json` shards.
3. After all workers finish, `aggregate_to_pt.py` merges the shards into one `.pt` per source HDF5 with the vBeta3-style schema (see `config.yaml`).

Outputs:
- per-event shards: `$SHARDS_DIR/<basename>__ev<NNNN>.{json,npz}`
- per-file pt:      `$PT_DIR/<basename>.v_alpha_test.pt`
- summaries:        `v_alpha_test_summary.json`, `v_alpha_test_aggregator_summary.json`

## Per-file .pt schema (vBeta3-compatible + new field)

See [`config.yaml`](config.yaml) for the full schema. Highlights:

| field | dtype | shape | filled by |
|---|---|---|---|
| `calib_hit_t0_reco` | float32 | `(n_calib_hits,)` | full pipeline (Front + Phase 2 + V2 + Phase 3); `hit_timestamps_post_phase3` scattered via `event.hit_refs` |
| **`prompt_hit_t_cluster_id`** | **int16** | **`(n_calib_hits,)`** | **front-stage clustering label `labels_global` scattered via `event.hit_refs`** *(temporary placeholder for the eventual charge-light cluster id)* |
| `n_calib_hits`, `n_assigned`, `n_unassigned` | int | scalar | aggregator |
| `processed_event_ids`, `all_event_ids` | int64 | varies | aggregator |
| `event_summaries`, `failed_events` | list[dict] | varies | aggregator |
| `version`, `algorithm`, `input_file` | str | scalar | aggregator |

**Sentinels:** unassigned prompt hits have `calib_hit_t0_reco = -1.0` and `prompt_hit_t_cluster_id = -1`.

## Manual aggregation

If you ran the batch but didn't auto-aggregate (e.g. you killed the launcher early), run the aggregator separately:

```bash
PY=/global/common/software/nersc/pe/conda-envs/26.1.0/python-3.13/nersc-python/bin/python
$PY /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/v_alpha_test/scripts/aggregate_to_pt.py \
    --shard-dir /pscratch/sd/y/yuxuan/light_rescue_test/valpha_runs/test10_v_alpha_test \
    --output-dir /pscratch/sd/y/yuxuan/light_rescue_test/valpha_runs/test10_v_alpha_test/pt_outputs
```

## Inspect one .pt

```bash
$PY /global/cfs/cdirs/dune/users/yuxuan/NDLAr-full/v_alpha_test/scripts/inspect_pt.py \
    /pscratch/sd/y/yuxuan/light_rescue_test/valpha_runs/test10_v_alpha_test/pt_outputs/MiniProdN5p1_NDComplex_FHC.flow.full.sanddrift.0000001.FLOW.v_alpha_test.pt
```

## Excluded from this folder (too large for github)

- The perceiver charge-light relation model weights
- The variance-prediction `.pt`

Both are loaded by `M5p1.first_stage_matching.load_first_stage_models` from their existing in-repo locations. The path is read from the front-stage config and is not modified by v_alpha_test.

## What changed vs valpha

- vBeta3-compatible per-file `.pt` output schema (one .pt per source HDF5).
- New `prompt_hit_t_cluster_id` field (int16) — currently filled with the front-stage `labels_global`; provenance documented in `config.yaml` so it's easy to swap in the charge-light cluster id later.
- Aggregator reads per-event NPZ shards independently of the GPU run, so re-aggregation is cheap and re-runnable.
