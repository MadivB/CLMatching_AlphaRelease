"""v_alpha_test: speed-optimized batch driver.

Functionally identical to ``phase25_trial2_valpha_batch`` (front stage +
Phase 2 + v2 light rescue), but layers in PyTorch-side speedups on the
prediction model:

- ``torch.compile(light_model, dynamic=True, mode="reduce-overhead")``
  -- compiles the perceiver forward pass.  First batch eats the compile
  cost, then steady-state inference is materially faster on A100.
- Enables TF32 on cuDNN + matmul (default-on in pytorch 2.x but we set
  it explicitly to be safe).
- Sets ``cudnn.benchmark = True`` so cuDNN can autotune conv kernels
  for the actual shapes used.
- Wraps the *whole* per-event pipeline in ``torch.inference_mode``
  rather than ``no_grad`` (slightly faster, identical numerics).
- Prefetches the next event's hits asynchronously on a worker thread
  while the GPU is busy with the current event.  Hides ~1-3s/event.

We deliberately do NOT change ``image_batch_size`` -- the variance
prediction step re-uses the same batch sizing and bumping it caused
issues in the past.

Same CLI as the valpha module::

    python -m M5p1.phase25_trial2_v_alpha_test \\
        --files <glob> --max-files 10 --out-dir <dir>

Per-event outputs are byte-compatible with the valpha module (same
JSON schema, same NPZ keys).
"""

from __future__ import annotations

import argparse
import gc
import glob as glob_mod
import json
import os
import queue
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _configure_paths() -> str:
    here = Path(__file__).resolve()
    base_dir = here.parent.parent
    candidates = [
        str(base_dir),
        str(base_dir / "M5p1"),
        str(base_dir / "NewMLSection"),
        str(base_dir.parent / "2x2CLMatching"),
    ]
    for path in candidates:
        if path and Path(path).exists() and path not in sys.path:
            sys.path.insert(0, path)
    return str(base_dir)


_BASE_DIR = _configure_paths()


import torch  # noqa: E402

from M5p1.first_stage_matching import (  # noqa: E402
    FirstStageConfig,
    load_first_stage_models,
    run_first_stage_charge_light_matching,
)
from M5p1 import phase25_trial2_v2_light_rescue as v2  # noqa: E402

# Re-use the valpha helpers verbatim so logic stays in one place.
from M5p1.phase25_trial2_valpha_batch import (  # noqa: E402
    _build_namespace,
    _list_events_in_file,
    _run_phase2,
    _serialize_light_moves,
    _summarize_v2_result,
    EventReport,
    build_default_first_stage_config,
    build_default_v2_config,
)
from M5p1.phase25_trial2_valpha_batch import _expand_files  # noqa: E402

from v11_phased_matching import (  # noqa: E402
    run_small_cluster_matrix_phase_v11,
    snapshot_backbone_hits_v11,
    verify_backbone_hits_unchanged_v11,
)


# Phase 3 constants (mirror notebook cell 35).
_V11_ENERGY_BAND_FRACTION = 0.20
_V11_POSITIVE_ROW_MARGIN = -1e-4
_V11_MATRIX_WORSEN_TOLERANCE_NORM = 0.15
_V11_FULL_SCAN_ASSIGN_EPS = -20000
_V11_BACKWARD_PEAK_ALIGN_TICKS = 5
_PHASE3_SEARCH_RANGE = 800
_ADC_CLIP = 60780.0


def _run_phase3(ns: dict[str, Any], *, verbose: bool = False) -> dict[str, Any]:
    """Mirror of notebook cell 35 (Phase 3 small-cluster matrix association).

    Runs after Phase 2 + the v2 light rescue.  Sweeps up the long tail of
    small clusters that the front stage left undecided.  Updates baseImage,
    hit_timestamps, t0Candidates, assignment_info, and unassigned_by_tpc
    in-place via the returned tuple.
    """
    t0 = time.perf_counter()
    (
        baseImage,
        hit_timestamps,
        t0Candidates,
        assignment_info,
        unassigned_by_tpc,
        small_cluster_assignment_log,
        _small_cluster_scan_updates,
        v11_small_phase_stats,
    ) = run_small_cluster_matrix_phase_v11(
        active_cluster_tpcs=ns["v11_active_cluster_tpcs"],
        iterative_single_tpc=ns["v11_iterative_single_tpc"],
        pruned_iterative_clusters=ns["v11_pruned_iterative_clusters"],
        image_maps=ns["imageMaps"],
        base_image=ns["baseImage"],
        full_light_waveform=ns["fullLightWaveform"],
        full_light_std=ns["fullLightStd"],
        channel_support_cache=ns["cluster_channel_support_cache"],
        labels_global=ns["labels_global"],
        hit_timestamps=ns["hit_timestamps"],
        t0_candidates=ns["t0Candidates"],
        assignment_info=ns["assignment_info"],
        unassigned_by_tpc=ns["unassigned_by_tpc"],
        cluster_energies=ns["cluster_energies"],
        energy_band_fraction=_V11_ENERGY_BAND_FRACTION,
        positive_row_margin=_V11_POSITIVE_ROW_MARGIN,
        matrix_worsen_tolerance_norm=_V11_MATRIX_WORSEN_TOLERANCE_NORM,
        search_range=_PHASE3_SEARCH_RANGE,
        adc_clip=_ADC_CLIP,
        collect_scan_losses=False,  # off -> faster, no extra storage
        full_scan_assign_eps=_V11_FULL_SCAN_ASSIGN_EPS,
        backward_peak_align_ticks=_V11_BACKWARD_PEAK_ALIGN_TICKS,
        leftover_absorption_context=ns.get("v11_1_leftover_absorption_context"),
        saturated_channel_cache=ns["saturated_channel_cache"],
    )
    ns["baseImage"] = baseImage
    ns["hit_timestamps"] = hit_timestamps
    ns["t0Candidates"] = t0Candidates
    ns["assignment_info"] = assignment_info
    ns["unassigned_by_tpc"] = unassigned_by_tpc

    n_assigned = sum(1 for row in small_cluster_assignment_log if row.get("assigned"))
    n_remaining = sum(len(v) for v in unassigned_by_tpc.values())
    elapsed = float(time.perf_counter() - t0)
    if verbose:
        print(
            f"  Phase 3 done in {elapsed:.1f}s | "
            f"assigned={n_assigned}/{len(small_cluster_assignment_log)} | "
            f"remaining={n_remaining}",
            flush=True,
        )
    return {
        "elapsed_s": elapsed,
        "n_proposed": int(len(small_cluster_assignment_log)),
        "n_assigned": int(n_assigned),
        "n_remaining_unassigned": int(n_remaining),
        "phase_stats": dict(v11_small_phase_stats),
    }


# ---------------------------------------------------------------------------
# Optimization helpers


def _enable_global_speedups(verbose: bool = False) -> dict[str, Any]:
    """Set cuDNN/matmul flags that should be safe on A100 + FP32 path."""
    info: dict[str, Any] = {}
    info["torch_version"] = torch.__version__
    info["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
        # TF32 on for matmul + cudnn (default-on in 2.x but be explicit).
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Autotune conv kernels for the shapes we hit.  Safe for variable
        # shapes; cuDNN remembers tuned plans per shape.
        torch.backends.cudnn.benchmark = True
        info["matmul_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
        info["cudnn_tf32"] = bool(torch.backends.cudnn.allow_tf32)
        info["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    if verbose:
        for k, v in info.items():
            print(f"  speedup.{k} = {v}", flush=True)
    return info


class _CompiledWithEagerFallback(torch.nn.Module):
    """Run ``compiled_model``; on *any* runtime error, retry once on the
    eager model and remember to use eager from then on.

    This insulates the batch from a single bad-shape compile event
    bringing down the entire run.  Each per-event recovery costs at
    most one eager forward.
    """

    def __init__(self, eager_model: torch.nn.Module, compiled_model):
        super().__init__()
        self.eager_model = eager_model
        self.compiled_model = compiled_model
        self.use_compiled = True
        self._n_compile_failures = 0

    def forward(self, *args, **kwargs):
        if self.use_compiled:
            try:
                return self.compiled_model(*args, **kwargs)
            except Exception as exc:
                self._n_compile_failures += 1
                print(
                    "  WARN: torch.compile forward failed "
                    f"({type(exc).__name__}); reverting to eager for the rest of this worker",
                    flush=True,
                )
                self.use_compiled = False
        return self.eager_model(*args, **kwargs)

    # Forward attribute access through to the eager model so anything
    # the front stage may probe (e.g. ``.training``, ``.parameters()``,
    # ``.to(device)``) keeps working.
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.eager_model, name)


def _try_compile_model(model, *, mode: str | None = "default", verbose: bool = False):
    """Wrap the perceiver model with ``torch.compile`` if available.

    Uses ``mode="default"`` (inductor compile, **no** CUDA-graph capture)
    -- ``mode="reduce-overhead"`` triggered ``CUBLAS_STATUS_INVALID_VALUE``
    in the cudagraph_trees path on variable-shape inputs.

    Wraps the result in ``_CompiledWithEagerFallback`` so a runtime
    failure on any single batch reverts to eager instead of failing
    the event.
    """
    if not hasattr(torch, "compile"):
        if verbose:
            print("  torch.compile not available; using eager model", flush=True)
        return model, False
    try:
        kwargs: dict[str, Any] = {"dynamic": True}
        if mode is not None:
            kwargs["mode"] = mode
        compiled = torch.compile(model, **kwargs)
    except Exception as exc:  # pragma: no cover - best effort
        print(f"  WARN: torch.compile setup failed ({exc!r}); using eager", flush=True)
        return model, False
    wrapped = _CompiledWithEagerFallback(model, compiled)
    if verbose:
        print(f"  torch.compile applied (mode={mode!r}, dynamic=True, fallback=eager)", flush=True)
    return wrapped, True


def load_first_stage_models_optimized(
    fs_config: FirstStageConfig,
    *,
    verbose: bool = False,
    enable_compile: bool = True,
    compile_mode: str | None = "default",
):
    """Same as ``load_first_stage_models`` plus optional torch.compile."""
    models = load_first_stage_models(fs_config)
    if not enable_compile:
        if verbose:
            print("  torch.compile disabled by config; using eager model", flush=True)
        return models
    if hasattr(models, "light_model") and models.light_model is not None:
        compiled, ok = _try_compile_model(
            models.light_model, mode=compile_mode, verbose=verbose
        )
        if ok:
            models.light_model = compiled
    # The variance model stays eager -- it runs per event (not per cluster)
    # and its compile cost is not amortized across calls.
    return models


# ---------------------------------------------------------------------------
# Async h5 prefetch


@dataclass
class _PrefetchedEvent:
    file_path: str
    event_id: int
    h5_open: h5py.File   # open handle owned by this record
    payload: dict[str, Any]


class _EventPrefetcher:
    """Reads upcoming (file, event_id) pairs in a worker thread.

    The h5 file handles are opened once and kept alive for the lifetime
    of the prefetcher; each result carries a reference so the consumer
    can use the open handle.  When ``next()`` is called and the queue
    is empty the consumer is blocked until the worker produces.
    """

    def __init__(self, items: list[tuple[str, int]], maxsize: int = 2):
        self.items = list(items)
        self.queue: "queue.Queue[_PrefetchedEvent | Exception | None]" = queue.Queue(maxsize=maxsize)
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        open_handles: dict[str, h5py.File] = {}
        try:
            for file_path, event_id in self.items:
                if self._stopped.is_set():
                    break
                h5 = open_handles.get(file_path)
                if h5 is None:
                    h5 = h5py.File(file_path, "r")
                    open_handles[file_path] = h5
                payload: dict[str, Any] = {"event_id": int(event_id)}
                # Touch the metadata once -- the heavy IO happens in the
                # front stage but we warm the page cache here.
                _ = h5["charge/events/data"]["id"][int(event_id)]
                self.queue.put(_PrefetchedEvent(
                    file_path=file_path,
                    event_id=int(event_id),
                    h5_open=h5,
                    payload=payload,
                ))
        except Exception as exc:  # pragma: no cover - defensive
            self.queue.put(exc)
        finally:
            self.queue.put(None)
            # Note: handles stay open until close() is called by the consumer.
            self._open_handles = open_handles

    def next(self) -> _PrefetchedEvent | None:
        item = self.queue.get()
        if item is None:
            return None
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        self._stopped.set()
        # Drain the queue.
        try:
            while True:
                self.queue.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=2.0)
        # Close any handles we created on the worker side.
        handles = getattr(self, "_open_handles", {}) or {}
        for h in handles.values():
            try:
                h.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Per-event pipeline (same logic as valpha, wrapped in inference_mode)


@contextmanager
def _inference_block():
    if hasattr(torch, "inference_mode"):
        with torch.inference_mode():
            yield
    else:
        with torch.no_grad():
            yield


def process_one_event_optimized(
    *,
    data_file: str,
    event_id: int,
    h5: h5py.File,
    fs_config: FirstStageConfig,
    fs_models,
    v2_config: v2.Trial2V2LightConfig,
    out_dir: Path,
    verbose: bool = False,
    save_arrays: bool = True,
    postpass: str = "none",
) -> EventReport:
    t_start = time.perf_counter()
    file_basename = Path(data_file).stem
    tag = f"{file_basename}__ev{int(event_id):04d}"
    npz_path = out_dir / f"{tag}.npz"
    json_path = out_dir / f"{tag}.json"

    try:
        with _inference_block():
            fs_result = run_first_stage_charge_light_matching(
                h5=h5,
                event_id=int(event_id),
                config=fs_config,
                models=fs_models,
                verbose=bool(verbose),
            )
            ns = _build_namespace(fs_result, h5, fs_config)
            hit_ts_pre_phase2 = np.asarray(ns["hit_timestamps"], dtype=np.float32).copy()
            base_image_pre_phase2 = np.asarray(ns["baseImage"], dtype=np.float32).copy()

            backbone_snapshot = snapshot_backbone_hits_v11(
                hit_timestamps=ns["hit_timestamps"],
                labels_global=ns["labels_global"],
                v8_absorbed_hit_parent=ns.get("v8_absorbed_hit_parent"),
                track_shower_labels=ns["track_shower_labels"],
            )
            _run_phase2(ns, verbose=verbose)
            verify_backbone_hits_unchanged_v11(
                backbone_snapshot,
                hit_timestamps=ns["hit_timestamps"],
                stage_name="Phase 2 (large-cluster scan)",
            )
            hit_ts_pre_v2 = np.asarray(ns["hit_timestamps"], dtype=np.float32).copy()

            result = v2.run_trial2_v2_light_rescue_from_namespace(
                ns,
                config=v2_config,
                commit=True,
            )
            hit_ts_post_v2 = np.asarray(ns["hit_timestamps"], dtype=np.float32).copy()
            base_image_post_v2 = np.asarray(ns["baseImage"], dtype=np.float32).copy()

            # ---- Re-label moved hits with fresh cluster ids ----
            # Each spatial or light move in V2 is a "movement of a group of
            # hits to a new t0".  Per spec: every such moved group gets a
            # brand-new cluster id starting at (max existing label + 1) and
            # incrementing.  The remaining hits of the original cluster keep
            # their old id.  Spatial moves come first (they happen first in
            # the V2 pipeline), then light moves.
            t_cluster_id = np.asarray(ns["labels_global"], dtype=np.int64).copy()
            valid_mask = t_cluster_id >= 0
            if np.any(valid_mask):
                next_id = int(t_cluster_id[valid_mask].max()) + 1
            else:
                next_id = 0

            v2_relabel_log: list[dict[str, Any]] = []

            def _safe_int(value: Any, default: int = -1) -> int:
                """Coerce to int; some move dicts store arrays for old_t0/new_t0."""
                if value is None:
                    return int(default)
                try:
                    arr = np.asarray(value).ravel()
                    if arr.size == 0:
                        return int(default)
                    return int(arr[0])
                except Exception:
                    return int(default)

            def _relabel_one_move(
                idx: np.ndarray,
                kind: str,
                move_meta: dict[str, Any],
            ) -> None:
                nonlocal next_id
                idx = np.asarray(idx, dtype=np.int64)
                if idx.size == 0:
                    return
                # Drop any out-of-range or duplicated indices defensively.
                in_range = (idx >= 0) & (idx < t_cluster_id.shape[0])
                idx = np.unique(idx[in_range])
                if idx.size == 0:
                    return
                old_ids = t_cluster_id[idx].copy()
                t_cluster_id[idx] = int(next_id)
                v2_relabel_log.append({
                    "kind": kind,
                    "new_cluster_id": int(next_id),
                    "n_hits": int(idx.size),
                    "TPCid": _safe_int(move_meta.get("TPCid"), -1),
                    "old_t0": _safe_int(move_meta.get("old_t0"), -1),
                    "new_t0": _safe_int(move_meta.get("new_t0"), -1),
                    "old_cluster_ids": [int(v) for v in np.unique(old_ids).tolist()][:8],
                })
                next_id += 1

            for move in result.get("spatial_moves", []):
                _relabel_one_move(move.get("moved_idx", []), "spatial", move)
            for move in result.get("light_moves", []):
                _relabel_one_move(move.get("hit_indices", []), "light", move)

            # Stash on the namespace + the result for downstream reporting.
            ns["t_cluster_id"] = t_cluster_id
            result["v2_cluster_relabel_log"] = v2_relabel_log
            result["t_cluster_id_max"] = int(next_id - 1) if v2_relabel_log else (
                int(t_cluster_id.max()) if (t_cluster_id >= 0).any() else -1
            )

            # ---- Phase 3: small-cluster matrix association (notebook cell 35) ----
            phase3_stats = _run_phase3(ns, verbose=verbose)
            verify_backbone_hits_unchanged_v11(
                backbone_snapshot,
                hit_timestamps=ns["hit_timestamps"],
                stage_name="Phase 3 (small-cluster matrix)",
            )

            # ---- optional v0.1 post-pass: spatial-guided family assignment ----
            # Runs before the final snapshot so hit_timestamps_post_phase3 (and
            # the whole NPZ -> .pt chain) transparently reflects it.
            postpass_stats = None
            if postpass and postpass != "none":
                import postpass_v01 as _pp01
                _t_pp = time.perf_counter()
                _hit_ts_live = np.asarray(ns["hit_timestamps"])
                if postpass in ("v0.1", "v0.1-fx"):
                    _pp_log = _pp01.family_expand_nd(ns, _hit_ts_live)
                    _n_moves = sum(
                        1 for r in _pp_log
                        if r["event"] in ("absorb", "seed_move")
                        or (r["event"] == "contact" and r.get("moved"))
                    )
                elif postpass == "v0.1-rg":
                    _pp_log = _pp01.region_grow_nd(ns, _hit_ts_live)
                    _n_moves = sum(1 for r in _pp_log if r["event"] == "grow")
                else:
                    raise ValueError(f"unknown postpass {postpass!r}")
                verify_backbone_hits_unchanged_v11(
                    backbone_snapshot,
                    hit_timestamps=ns["hit_timestamps"],
                    stage_name=f"v0.1 post-pass ({postpass})",
                )
                postpass_stats = {
                    "mode": postpass,
                    "n_moves": int(_n_moves),
                    "n_log_rows": int(len(_pp_log)),
                    "elapsed_s": float(time.perf_counter() - _t_pp),
                }
        hit_ts_post_phase3 = np.asarray(ns["hit_timestamps"], dtype=np.float32).copy()
        base_image_post_phase3 = np.asarray(ns["baseImage"], dtype=np.float32).copy()

        # Per-stage coverage (energy-weighted finite-t0 fraction).
        finite_after_v2 = np.isfinite(hit_ts_post_v2).sum() / max(hit_ts_post_v2.size, 1)
        finite_after_p3 = np.isfinite(hit_ts_post_phase3).sum() / max(hit_ts_post_phase3.size, 1)
        coverage = {
            "n_hits": int(hit_ts_post_v2.size),
            "frac_finite_pre_phase2": float(np.isfinite(hit_ts_pre_phase2).mean()),
            "frac_finite_pre_v2": float(np.isfinite(hit_ts_pre_v2).mean()),
            "frac_finite_post_v2": float(finite_after_v2),
            "frac_finite_post_phase3": float(finite_after_p3),
            "phase3_assigned_hits": int(
                np.isfinite(hit_ts_post_phase3).sum() - np.isfinite(hit_ts_post_v2).sum()
            ),
        }

        summary = _summarize_v2_result(result)
        summary["phase3"] = phase3_stats
        if postpass_stats is not None:
            summary["postpass_v01"] = postpass_stats
        summary["coverage"] = coverage
        summary["v2_cluster_relabel"] = {
            "n_new_cluster_ids": int(len(result.get("v2_cluster_relabel_log", []))),
            "max_t_cluster_id": int(result.get("t_cluster_id_max", -1)),
            "n_orig_clusters": int(int(np.asarray(ns["labels_global"]).max()) + 1)
                                if (np.asarray(ns["labels_global"]) >= 0).any() else 0,
        }
        light_moves_audit = _serialize_light_moves(list(result.get("light_moves", [])))

        report = {
            "ok": True,
            "file": str(data_file),
            "file_basename": file_basename,
            "event_id": int(event_id),
            "n_hits": int(np.asarray(ns["xset"]).size),
            "summary": summary,
            "light_moves": light_moves_audit,
            "v2_pass_logs": list(result.get("v2_pass_logs", [])),
            "elapsed_s": float(time.perf_counter() - t_start),
            "v_alpha_test": True,
        }
        with open(json_path, "w") as f:
            json.dump(report, f, indent=1, default=float)

        if save_arrays:
            # Per-event NPZ schema:
            #   hit_refs        -- indices into the file-global calib_prompt_hits
            #                      table (used to scatter per-event values into
            #                      the per-file output array).
            #   labels_global   -- ORIGINAL front-stage clustering label per
            #                      event hit (kept for debug/audit).
            #   t_cluster_id    -- POST-V2 cluster id per event hit.  Hits
            #                      moved during V2 (spatial or light moves)
            #                      are re-labeled with brand-new ids starting
            #                      at (max original label) + 1.  This is the
            #                      array the aggregator scatters into the
            #                      per-file prompt_hit_t_cluster_id field.
            np.savez_compressed(
                npz_path,
                hit_timestamps_pre_phase2=hit_ts_pre_phase2,
                hit_timestamps_pre_v2=hit_ts_pre_v2,
                hit_timestamps_post_v2=hit_ts_post_v2,
                hit_timestamps_post_phase3=hit_ts_post_phase3,
                baseImage_delta_v2=(base_image_post_v2 - base_image_pre_phase2).astype(np.float32),
                baseImage_delta_phase3=(base_image_post_phase3 - base_image_post_v2).astype(np.float32),
                hitTPCid=np.asarray(ns["hitTPCid"], dtype=np.int32),
                hit_refs=np.asarray(ns["hit_refs"], dtype=np.int64),
                labels_global=np.asarray(ns["labels_global"], dtype=np.int64),
                t_cluster_id=np.asarray(ns["t_cluster_id"], dtype=np.int64),
            )

        return EventReport(
            file=str(data_file),
            event_id=int(event_id),
            ok=True,
            summary=summary,
            elapsed_s=report["elapsed_s"],
        )
    except Exception as exc:
        err = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        try:
            with open(json_path, "w") as f:
                json.dump(
                    {
                        "file": str(data_file),
                        "event_id": int(event_id),
                        "ok": False,
                        "error": str(exc),
                        "traceback": err,
                        "v_alpha_test": True,
                    },
                    f,
                    indent=1,
                )
        except Exception:
            pass
        return EventReport(
            file=str(data_file),
            event_id=int(event_id),
            ok=False,
            error=err,
            elapsed_s=float(time.perf_counter() - t_start),
        )


# ---------------------------------------------------------------------------
# Batch driver with prefetch


def run_batch_optimized(
    *,
    file_paths: list[str],
    out_dir: str,
    fs_config: FirstStageConfig | None = None,
    v2_config: v2.Trial2V2LightConfig | None = None,
    max_events_per_file: int = 0,
    skip_existing: bool = True,
    skip_only_ok: bool = True,
    verbose: bool = False,
    prefetch_depth: int = 2,
    enable_compile: bool = True,
    compile_mode: str | None = "default",
    event_stride: int = 1,
    event_offset: int = 0,
    postpass: str = "none",
) -> dict[str, Any]:
    fs_config = fs_config or build_default_first_stage_config()
    v2_config = v2_config or build_default_v2_config()
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    print(f"v_alpha_test batch: {len(file_paths)} files -> {out_dir_p}", flush=True)
    print("enabling speedups ...", flush=True)
    _enable_global_speedups(verbose=verbose)
    print(f"loading first-stage models (one-time; compile={enable_compile}, mode={compile_mode!r}) ...", flush=True)
    t0 = time.perf_counter()
    fs_models = load_first_stage_models_optimized(
        fs_config,
        verbose=True,
        enable_compile=bool(enable_compile),
        compile_mode=compile_mode,
    )
    print(f"  models loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    # Build the global (file, event_id) work list and apply event-stride
    # *before* skip-existing so the stride splits the same total list
    # regardless of which events have already been done.
    stride = max(1, int(event_stride))
    offset = max(0, int(event_offset)) % stride
    full_list: list[tuple[str, int]] = []
    for file_path in file_paths:
        if not Path(file_path).exists():
            print(f"SKIP missing: {file_path}", flush=True)
            continue
        try:
            event_ids = _list_events_in_file(file_path)
        except Exception as exc:
            print(f"SKIP unreadable: {file_path}: {exc}", flush=True)
            continue
        if int(max_events_per_file) > 0:
            event_ids = event_ids[: int(max_events_per_file)]
        for ev_id in event_ids:
            full_list.append((file_path, int(ev_id)))

    # Stride/offset slice (round-robin over the global list).
    my_slice = full_list[offset::stride]
    if stride > 1:
        print(
            f"event partition: stride={stride} offset={offset} -> "
            f"{len(my_slice)} of {len(full_list)} events for this worker",
            flush=True,
        )

    # Apply skip-existing filter to my slice.
    work_list: list[tuple[str, int]] = []
    for file_path, ev_id in my_slice:
        tag = f"{Path(file_path).stem}__ev{int(ev_id):04d}"
        json_path = out_dir_p / f"{tag}.json"
        if skip_existing and json_path.exists():
            if skip_only_ok:
                try:
                    with open(json_path) as f:
                        existing = json.load(f)
                    if bool(existing.get("ok", False)):
                        continue
                except Exception:
                    pass
            else:
                continue
        work_list.append((file_path, int(ev_id)))

    print(f"work items after skip filter: {len(work_list)}", flush=True)
    if not work_list:
        print("nothing to do.", flush=True)
        return {"n_events_attempted": 0, "n_ok": 0, "n_err": 0,
                "summary_path": str(out_dir_p / "v_alpha_test_summary.json")}

    prefetcher = _EventPrefetcher(work_list, maxsize=int(prefetch_depth))
    t_batch = time.perf_counter()
    reports: list[dict[str, Any]] = []
    n_ok = 0
    n_err = 0
    total = 0
    try:
        while True:
            item = prefetcher.next()
            if item is None:
                break
            total += 1
            t_ev = time.perf_counter()
            print(
                f"[{total}/{len(work_list)}] {Path(item.file_path).name} ev{item.event_id:04d}",
                flush=True,
            )
            report = process_one_event_optimized(
                data_file=item.file_path,
                event_id=item.event_id,
                h5=item.h5_open,
                fs_config=fs_config,
                fs_models=fs_models,
                v2_config=v2_config,
                out_dir=out_dir_p,
                verbose=verbose,
                postpass=postpass,
            )
            if report.ok:
                n_ok += 1
                s = report.summary
                cov = s.get("coverage", {}) or {}
                p3 = s.get("phase3", {}) or {}
                print(
                    f"  OK light_moves={s.get('n_light_moves', 0)}"
                    f" rshfl={s.get('n_reshuffle_moves', 0)}"
                    f" forceOR={s.get('n_force_overrides', 0)}"
                    f" phase3_assigned={p3.get('n_assigned', 0)}/{p3.get('n_proposed', 0)}"
                    f" cover_post_v2={cov.get('frac_finite_post_v2', 0):.3f}"
                    f" cover_post_p3={cov.get('frac_finite_post_phase3', 0):.3f}"
                    f" elapsed={time.perf_counter() - t_ev:.1f}s",
                    flush=True,
                )
            else:
                n_err += 1
                short = (report.error or "").splitlines()[-1][:200]
                print(
                    f"  ERR after {time.perf_counter() - t_ev:.1f}s: {short}",
                    flush=True,
                )
            reports.append({
                "file": report.file,
                "event_id": report.event_id,
                "ok": report.ok,
                "elapsed_s": report.elapsed_s,
                "summary": report.summary if report.ok else {"error": report.error},
            })
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        prefetcher.close()

    summary_path = out_dir_p / "v_alpha_test_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "n_files": int(len(file_paths)),
                "n_events_attempted": int(total),
                "n_ok": int(n_ok),
                "n_err": int(n_err),
                "wall_s": float(time.perf_counter() - t_batch),
                "v_alpha_test": True,
                "per_event": reports,
            },
            f,
            indent=1,
            default=float,
        )
    print(
        f"v_alpha_test batch complete: events={total} ok={n_ok} err={n_err}"
        f" wall={time.perf_counter() - t_batch:.1f}s",
        flush=True,
    )
    return {
        "n_events_attempted": total,
        "n_ok": n_ok,
        "n_err": n_err,
        "summary_path": str(summary_path),
    }


# ---------------------------------------------------------------------------
# CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="v_alpha_test: speed-optimized batch driver "
                    "(torch.compile + TF32 + cudnn.benchmark + h5 prefetch).",
    )
    parser.add_argument("--files", nargs="+", required=True,
                        help="Explicit paths or glob patterns.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-events-per-file", type=int, default=0)
    parser.add_argument("--n-outer-passes", type=int, default=1)
    parser.add_argument("--max-total-moves", type=int, default=24)
    parser.add_argument("--max-moves-per-tpc", type=int, default=3)
    parser.add_argument("--device-policy", default="auto",
                        choices=("auto", "force_cuda", "force_cpu"))
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--prefetch-depth", type=int, default=2,
                        help="how many events to prefetch ahead (default 2)")
    parser.add_argument("--no-compile", action="store_true",
                        help="disable torch.compile (use eager model only)")
    parser.add_argument("--compile-mode", default="default",
                        choices=("default", "reduce-overhead", "max-autotune"),
                        help="torch.compile mode (default 'default'; "
                             "'reduce-overhead' may break on variable shapes)")
    parser.add_argument("--event-stride", type=int, default=1,
                        help="Round-robin partition: take every Nth event from "
                             "the global (file, event) list.  Use with "
                             "--event-offset to spread N workers across the "
                             "same set of files.")
    parser.add_argument("--event-offset", type=int, default=0,
                        help="Offset 0..stride-1 for round-robin partitioning.")
    parser.add_argument("--postpass", default="none",
                        choices=["none", "v0.1", "v0.1-fx", "v0.1-rg"],
                        help="optional v0.1 post-pass after Phase 3: "
                             "'v0.1'/'v0.1-fx' = chi2 family-expand (spatial-"
                             "guided, remove-and-rescore base); 'v0.1-rg' = "
                             "cosine region-grow. Default 'none' = baseline "
                             "vAlpha, bit-identical to previous releases.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    files = _expand_files(args.files)
    if int(args.max_files) > 0:
        files = files[: int(args.max_files)]
    if not files:
        print("ERROR: no input files.", file=sys.stderr)
        return 2

    fs_cfg = build_default_first_stage_config()
    v2_cfg = build_default_v2_config(
        n_outer_passes=int(args.n_outer_passes),
        light_max_total_moves=int(args.max_total_moves),
        light_max_moves_per_tpc=int(args.max_moves_per_tpc),
        device_policy=str(args.device_policy),
        verbose=bool(args.verbose),
    )

    res = run_batch_optimized(
        file_paths=files,
        out_dir=str(args.out_dir),
        fs_config=fs_cfg,
        v2_config=v2_cfg,
        max_events_per_file=int(args.max_events_per_file),
        skip_existing=not bool(args.no_skip_existing),
        verbose=bool(args.verbose),
        prefetch_depth=int(args.prefetch_depth),
        enable_compile=not bool(args.no_compile),
        compile_mode=str(args.compile_mode),
        event_stride=int(args.event_stride),
        event_offset=int(args.event_offset),
        postpass=str(args.postpass),
    )
    return 0 if res["n_err"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
