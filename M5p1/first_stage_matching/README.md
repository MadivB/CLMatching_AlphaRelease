# First Stage Charge-Light Matching

This package runs the stable front half of `v13_4` as an importable code path:

1. Load one charge/light event.
2. Run global track/shower clustering.
3. Predict per-label light images with the GPU Perceiver.
4. Predict phase-2 variance with the variance model.
5. Build support-channel and saturation-veto caches.
6. Place track/shower labels with unit-std loss.
7. Run optional track rescan, overlap swap, and fine half-tick t0 correction.

It intentionally stops before large/small non-track cluster association.

## Streaming Image Prediction

The default image prediction mode is streaming:

```python
config.prediction.image_prediction_mode = "streaming"
config.prediction.image_batch_size = 8
config.prediction.image_voxelize_device = "auto"
```

This still predicts every `(cluster_id, charge_tpc_id)` light image, but it
does not build the full dense `(n_groups, 1, 50, 300, 100)` voxel tensor at
once. It groups hits once, precomputes voxel indices once, then voxelizes and
runs inference one batch of groups at a time.

`image_voxelize_device="auto"` uses CUDA `scatter_add_` when the light model is
on GPU; set it to `"cpu"` to force CPU `np.add.at` voxelization. The returned
`imageMaps` remain full `(120, 1000)` arrays for downstream compatibility.

For more speed, test mixed precision explicitly:

```python
config.prediction.image_use_mixed_precision = True
config.prediction.image_amp_dtype = "bf16"
```

This changes numerical precision slightly, so compare the downstream result
before making it the release default.

To benchmark image settings without rerunning the whole pipeline:

```python
from M5p1.first_stage_matching import benchmark_image_prediction_from_result

bench = benchmark_image_prediction_from_result(
    first_stage_result,
    FIRST_STAGE_CONFIG,
    batch_sizes=(8, 16, 24, 32),
    mixed_precision_options=(False, True),
    voxelize_devices=("auto",),
)
```

## Minimal Use

```python
from M5p1.first_stage_matching import (
    FirstStageConfig,
    run_first_stage_charge_light_matching,
)

result = run_first_stage_charge_light_matching(
    data_file="/path/to/file.FLOW.hdf5",
    event_id=1,
    config=FirstStageConfig(),
    verbose=True,
)

hit_t0s = result.hit_t0
current_light_prediction = result.current_light_prediction
modified_flash_table = result.modified_flash_table_by_tpc
```

`hit_t0s` has one entry per charge hit. Unplaced hits are exported as `-1`.

`current_light_prediction` is the accumulated predicted light image after the
track/shower first stage. Shape is `(n_charge_tpc, 120, 1000)`.

`modified_flash_table_by_tpc` is the per-TPC t0 candidate table after the
track/shower stage and track corrections.

## Fast vs Exact Track Scan

The default first-pass track/shower scan is:

```python
config.track_stage.scan_mode = "correlation"
```

This uses a fast unit-std correlation objective. For notebook-equivalent exact
loop behavior:

```python
config.track_stage.scan_mode = "exact"
```

## Optional Stages

```python
config.track_stage.enable_second_pass_rescan = True
config.track_stage.enable_overlap_swap = True
config.track_stage.enable_fine_correction = True
```

The fine correction scans:

```python
[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
```

around the current assigned track t0 and uses unit waveform std.
