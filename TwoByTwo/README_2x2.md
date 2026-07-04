# 2x2 charge-light matching (`TwoByTwo/`)

A port of the ND-LAr alpha-release algorithm to the DUNE 2x2 demonstrator. The
2x2 has a different geometry (32×128×64 voxels, 8 TPCs) and light system (48
channels/TPC, ordered `[(0,0..23),(1,0..23)]`) and much less pile-up than ND, so
the small/low-energy clusters are the focus.

## Layout

```
TwoByTwo/
├── matcher/                 # the QLMatching2x2 package (flat imports; added to sys.path)
│   ├── pipeline_2x2.py      #   run_pipeline_for_event(h5, ev, light_model=..., **cfg)
│   ├── matching_2x2.py      #   greedy placement, region-grow, matched-filter kernels
│   ├── data_2x2.py          #   FLOW -> hits/light; hit_refs index calib_prompt_hits
│   ├── light_model_2x2.py   #   perceiver wrapper (load_light_model('sim'|'data'))
│   ├── truth_2x2.py         #   ND-style segment-t0 truth (for sim validation)
│   ├── eval_efficiency.py   #   energy-weighted ±160 ns efficiency (ND save_t0_hist)
│   └── ...                  #   clustering, RANSAC tracks, phase25, geometry, lut
├── perceiver/               # = C2L_DIR; ML_2x2_perceiver.py + perceiver3d.py
│   └── runs/
│       ├── 2x2_run/best_model.pt        # SIM perceiver  (3.5 MB, committed)
│       └── 2x2_data_run/best_model.pt   # DATA perceiver (3.5 MB, committed)
├── var_model/
│   ├── var_model.py
│   └── var_model_2x2.pt     # variance predictor (6 MB, committed; v2.0 only)
├── assets/avg_pulse_2x2.npy # pulse template (length 1000)
├── run_2x2_worker.py        # one GPU worker: events -> per-event NPZ shards
├── aggregate_2x2_to_pt.py   # NPZ shards -> <basename>.qlmatch2x2.pt
└── README_2x2.md            # this file
```

The matcher uses **flat imports** (`import pipeline_2x2`), so `run_2x2_worker.py`
inserts `matcher/`, `perceiver/`, `var_model/` onto `sys.path` and sets
`C2L_DIR` to `perceiver/` before importing. Nothing here depends on absolute
dev paths; the worker resolves every model from in-repo locations.

## Running

Use the launchers (8 workers / 4 GPUs, auto-aggregate). From the repo root:

```bash
bash scripts/run_2x2_sim.sh                 # v1.0 error-matrix (default)
VERSION=v2.0 bash scripts/run_2x2_sim.sh    # region-grow + tiebreaker
bash scripts/run_2x2_data.sh                # real beam data, data perceiver
```

Or drive one worker directly (e.g. a CPU smoke test of a single event):

```bash
python TwoByTwo/run_2x2_worker.py \
    --files /path/to.FLOW.hdf5 --out-dir output/smoke \
    --mode sim --version v1.0 \
    --event-stride 9999 --event-offset 0 --max-events-per-file 1 --device cpu
python TwoByTwo/aggregate_2x2_to_pt.py --shard-dir output/smoke --overwrite
```

## Algorithm versions

| `--version` | config passed to `run_pipeline_for_event` | notes |
|---|---|---|
| `v1.0` (default) | `enable_region_grow=False` | **error-matrix**: greedy per-TPC brightest-first small-cluster association, unit-variance χ². The ND vAlpha formulation. |
| `v0.1` (alias `v2.0`) | `enable_region_grow=True`, `tiebreak_variance_model=<2x2 var>`, `tie_frac=0.10` | **region-grow + tiebreaker** — the development line toward the first serious release. |
| `v0.1-fx` (experimental) | greedy + `family_expand_association` post-pass | **χ² family-expand** ([family_expand_2x2.py](matcher/family_expand_2x2.py)): cosine-free agglomerative spatial families arbitrated by the error matrix. |

All use `unit_variance=True` for the core matching — the established best for 2x2
t0 χ² (the bright channels carry the flash-discriminating signal; down-weighting
them with a learned variance hurts the match).

### Why v1.0 is the default

The error-matrix formulation is the validated, conservative baseline. The v0.1
line is opt-in until further review:

* **region-grow (v0.1)** raises 2x2 sim efficiency ~+0.8 pp overall (low-energy-
  safe — every per-cluster energy bin improves). The learned-variance tiebreaker
  inside it is roughly neutral in aggregate (95.845% vs 95.830% on a 3-file A/B);
  kept because it targets genuinely ambiguous t0 ties.
* **family-expand (v0.1-fx)** beats region-grow on the 2x2 sim aggregate
  (96.3% vs 95.7% vs 90.4% baseline on the hard-sample benchmark) but this 2x2
  prototype scores at base=0, a known multi-flash caveat. The ND port
  (`M5p1/postpass_v01.py`) fixes it with remove-and-rescore residual scoring;
  measured on ND (63 events): baseline 92.63% → family-expand 92.83%
  (worst event −0.10 pp) / region-grow 92.88% (worst event −0.62 pp).
* Cross-detector picture: family-expand is the *safer* variant (magnitude-aware —
  a faint displaced fragment cannot latch onto a bright flash on pattern alone);
  region-grow is marginally higher in ND aggregate. Both ship in v0.1.

## Output

`<basename>.qlmatch2x2.pt` — the same schema as the ND `v_alpha_test` release
(`calib_hit_t0_reco`, `prompt_hit_t_cluster_id`, `calib_final_hit_t0_reco`,
`calib_final_hit_cluster_id`, counts, `event_summaries`), plus `detector="2x2"`
and `version="qlmatch2x2.1"`. t0 is in 16 ns matching ticks; unassigned hits are
`-1`. See [`../config.yaml`](../config.yaml).

## Validating on simulation

`matcher/eval_efficiency.py` computes the ND energy-weighted ±160 ns efficiency
against the segment-t0 truth (`matcher/truth_2x2.py`). On the sim validation
sample the matcher reaches ~95–96% overall (tracks ~99.5%, blobs ~86–95%),
beating the ND 93% thanks to the lower 2x2 pile-up.
