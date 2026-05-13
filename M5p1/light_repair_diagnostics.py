from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go

try:
    from plottingTools import VALID_GROUP_COLORS
except ModuleNotFoundError:
    from M5p1.plottingTools import VALID_GROUP_COLORS


def _as_array(value: Any, dtype: Any | None = None) -> np.ndarray:
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _weighted_pca_for_indices(
    indices: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energy: np.ndarray,
) -> dict[str, Any]:
    idx = _as_array(indices, np.int64)
    pts = np.column_stack((x[idx], y[idx], z[idx])).astype(np.float64)
    weights = np.clip(_as_array(energy[idx], np.float64), 1e-9, None)
    weights = weights / max(float(np.sum(weights)), 1e-12)

    centroid = np.sum(pts * weights[:, None], axis=0)

    if pts.shape[0] < 3:
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        evals = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        evecs = np.eye(3, dtype=np.float64)
        proj = np.zeros(pts.shape[0], dtype=np.float64)
        perp_dist = np.zeros(pts.shape[0], dtype=np.float64)
        linearity = 0.0
        transverse_rms = 0.0
    else:
        centered = pts - centroid
        cov = (centered * weights[:, None]).T @ centered
        try:
            evals, evecs = np.linalg.eigh(cov)
            order = np.argsort(evals)[::-1]
            evals = evals[order]
            evecs = evecs[:, order]
        except np.linalg.LinAlgError:
            evals = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            evecs = np.eye(3, dtype=np.float64)

        direction = evecs[:, 0]
        direction = direction / max(float(np.linalg.norm(direction)), 1e-12)

        centered = pts - centroid
        proj = centered @ direction
        perp = centered - proj[:, None] * direction[None, :]
        perp_dist = np.sqrt(np.sum(perp * perp, axis=1))

        linearity = float(evals[0] / max(float(np.sum(evals)), 1e-12))
        transverse_rms = float(np.sqrt(max(float(evals[1] + evals[2]), 0.0)))

    proj_min = float(np.min(proj)) if proj.size else 0.0
    proj_max = float(np.max(proj)) if proj.size else 0.0
    length = float(proj_max - proj_min)
    width68 = float(np.quantile(perp_dist, 0.68)) if perp_dist.size else 0.0
    width90 = float(np.quantile(perp_dist, 0.90)) if perp_dist.size else 0.0

    return {
        "points": pts,
        "centroid": centroid,
        "evals": evals,
        "evecs": evecs,
        "direction": direction,
        "proj": proj,
        "proj_min": proj_min,
        "proj_max": proj_max,
        "length": length,
        "perp_dist": perp_dist,
        "width68": width68,
        "width90": width90,
        "linearity": linearity,
        "transverse_rms": transverse_rms,
    }


def _label_summary(indices: np.ndarray, labels: np.ndarray, *, max_items: int = 6) -> str:
    idx = _as_array(indices, np.int64)
    if idx.size == 0:
        return "[]"

    vals, counts = np.unique(labels[idx].astype(np.int32), return_counts=True)
    order = np.argsort(counts)[::-1]
    return "[" + ", ".join(f"{int(vals[k])}:{int(counts[k])}" for k in order[:max_items]) + "]"


def summarize_light_repair_donor_components(
    light_repair_result: dict[str, Any],
    *,
    labels_global: np.ndarray,
    xset: np.ndarray,
    yset: np.ndarray,
    zset: np.ndarray,
    Eset: np.ndarray,
) -> list[dict[str, Any]]:
    """
    Return PCA/shape summaries for donor components identified by the light repair prepass.
    """
    donor_components = list(light_repair_result.get("donor_components", []))
    labels = _as_array(labels_global, np.int32)
    x = _as_array(xset, np.float64)
    y = _as_array(yset, np.float64)
    z = _as_array(zset, np.float64)
    energy = _as_array(Eset, np.float64)

    rows: list[dict[str, Any]] = []
    for comp in donor_components:
        idx = _as_array(comp["hit_indices"], np.int64)
        if idx.size == 0:
            continue

        pca = _weighted_pca_for_indices(idx, x=x, y=y, z=z, energy=energy)
        label_summary = _label_summary(idx, labels)
        comp_energy = float(np.sum(energy[idx]))

        rows.append(
            {
                "component_id": int(comp["component_id"]),
                "n_hits": int(idx.size),
                "energy_mev": comp_energy,
                "parent_label": int(comp.get("parent_label", -1)),
                "labels": label_summary,
                "linearity": float(pca["linearity"]),
                "length_cm": float(pca["length"]),
                "width68_cm": float(pca["width68"]),
                "width90_cm": float(pca["width90"]),
                "transverse_rms_cm": float(pca["transverse_rms"]),
                "eval0": float(pca["evals"][0]),
                "eval1": float(pca["evals"][1]),
                "eval2": float(pca["evals"][2]),
                "centroid_x": float(pca["centroid"][0]),
                "centroid_y": float(pca["centroid"][1]),
                "centroid_z": float(pca["centroid"][2]),
                "pca": pca,
                "hit_indices": idx,
            }
        )

    return sorted(rows, key=lambda r: (-r["energy_mev"], -r["n_hits"], r["component_id"]))


def print_component_pca_summary(pca_rows: list[dict[str, Any]]) -> None:
    if len(pca_rows) == 0:
        print("No donor components to summarize.")
        return

    header = (
        f"{'comp':>6} {'hits':>7} {'E[MeV]':>9} {'lin':>7} "
        f"{'len':>8} {'w68':>8} {'w90':>8} {'trRMS':>8} "
        f"{'parent':>8} {'labels':>24}"
    )
    print(header)
    print("-" * len(header))
    for row in pca_rows:
        print(
            f"{row['component_id']:6d} "
            f"{row['n_hits']:7d} "
            f"{row['energy_mev']:9.2f} "
            f"{row['linearity']:7.3f} "
            f"{row['length_cm']:8.2f} "
            f"{row['width68_cm']:8.2f} "
            f"{row['width90_cm']:8.2f} "
            f"{row['transverse_rms_cm']:8.2f} "
            f"{row['parent_label']:8d} "
            f"{str(row['labels']):>24}"
        )


def plot_light_repair_donor_components_pca(
    light_repair_result: dict[str, Any],
    *,
    hit_timestamps: np.ndarray,
    hitTPCid: np.ndarray,
    labels_global: np.ndarray,
    xset: np.ndarray,
    yset: np.ndarray,
    zset: np.ndarray,
    Eset: np.ndarray,
    t0_match_ticks: float = 10.0,
    save_path: str | None = None,
    show: bool = True,
    show_context: bool = True,
    show_traces: bool = True,
    show_infigure_labels: bool = True,
    show_centroids: bool = True,
    context_opacity: float = 0.08,
    marker_size: float = 4.0,
    axis_width: int = 7,
    centroid_size: float = 8.0,
    print_summary: bool = True,
) -> dict[str, Any]:
    """
    Plot donor components with optional PCA traces and in-figure labels.

    Set both `show_traces=False` and `show_infigure_labels=False` for a clean
    component-grouping-only view.
    """
    tpc = int(light_repair_result["TPCid"])
    old_t0 = int(light_repair_result["old_t0"])
    donor_components = list(light_repair_result.get("donor_components", []))
    if len(donor_components) == 0:
        raise RuntimeError("No donor_components found in light_repair_result.")

    if save_path is None:
        save_path = f"TPC{tpc}_t0_{old_t0}_donor_components_pca.html"

    hit_t0 = _as_array(hit_timestamps, np.float64)
    hit_tpc = _as_array(hitTPCid, np.int32)
    labels = _as_array(labels_global, np.int32)
    x = _as_array(xset, np.float64)
    y = _as_array(yset, np.float64)
    z = _as_array(zset, np.float64)
    energy = _as_array(Eset, np.float64)

    pca_rows = summarize_light_repair_donor_components(
        light_repair_result,
        labels_global=labels,
        xset=x,
        yset=y,
        zset=z,
        Eset=energy,
    )

    tpc_mask = hit_tpc == tpc
    old_t0_mask = (
        tpc_mask
        & np.isfinite(hit_t0)
        & (hit_t0 >= 0)
        & (np.abs(hit_t0 - float(old_t0)) <= float(t0_match_ticks))
    )

    component_hit_set = set()
    for row in pca_rows:
        component_hit_set.update(int(v) for v in _as_array(row["hit_indices"], np.int64).tolist())

    fig = go.Figure()

    if show_context:
        bg_idx = np.flatnonzero(old_t0_mask).astype(np.int64)
        bg_idx = np.asarray([i for i in bg_idx if int(i) not in component_hit_set], dtype=np.int64)
        if bg_idx.size > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=z[bg_idx],
                    y=y[bg_idx],
                    z=x[bg_idx],
                    mode="markers",
                    marker=dict(size=2.0, color="lightgray", opacity=float(context_opacity)),
                    name=f"other hits at t0~{old_t0} ({bg_idx.size})",
                    hoverinfo="skip",
                )
            )

    for i, row in enumerate(pca_rows):
        idx = _as_array(row["hit_indices"], np.int64)
        color_hex, color_name = VALID_GROUP_COLORS[i % len(VALID_GROUP_COLORS)]
        pca = row["pca"]

        hover = [
            (
                f"component={row['component_id']}<br>"
                f"hit={int(h)}<br>"
                f"TPC={tpc}<br>"
                f"old_t0={old_t0}<br>"
                f"label={int(labels[h])}<br>"
                f"E={energy[h]:.4f} MeV<br>"
                f"component hits={row['n_hits']}<br>"
                f"component E={row['energy_mev']:.2f} MeV<br>"
                f"parent label={row['parent_label']}<br>"
                f"labels={row['labels']}<br>"
                f"PCA linearity={row['linearity']:.3f}<br>"
                f"PCA length={row['length_cm']:.2f} cm<br>"
                f"PCA width68={row['width68_cm']:.2f} cm<br>"
                f"PCA width90={row['width90_cm']:.2f} cm<br>"
                f"transverse RMS={row['transverse_rms_cm']:.2f} cm"
            )
            for h in idx
        ]

        fig.add_trace(
            go.Scatter3d(
                x=z[idx],
                y=y[idx],
                z=x[idx],
                mode="markers",
                marker=dict(size=float(marker_size), color=color_hex, opacity=0.90),
                text=hover,
                hoverinfo="text+x+y+z",
                name=(
                    f"{color_name} comp {row['component_id']} | "
                    f"{row['n_hits']} hits | E={row['energy_mev']:.1f} | lin={row['linearity']:.2f}"
                ),
            )
        )

        if show_traces:
            p0 = pca["centroid"] + pca["direction"] * float(pca["proj_min"])
            p1 = pca["centroid"] + pca["direction"] * float(pca["proj_max"])
            fig.add_trace(
                go.Scatter3d(
                    x=[p0[2], p1[2]],
                    y=[p0[1], p1[1]],
                    z=[p0[0], p1[0]],
                    mode="lines",
                    line=dict(color=color_hex, width=int(axis_width)),
                    name=f"PCA axis comp {row['component_id']}",
                    hovertext=(
                        f"component={row['component_id']}<br>"
                        f"linearity={row['linearity']:.3f}<br>"
                        f"length={row['length_cm']:.2f} cm<br>"
                        f"width68={row['width68_cm']:.2f} cm<br>"
                        f"width90={row['width90_cm']:.2f} cm"
                    ),
                    hoverinfo="text",
                    showlegend=False,
                )
            )

            evec2 = pca["evecs"][:, 1]
            evec3 = pca["evecs"][:, 2]
            for vec in (evec2, evec3):
                if float(row["width68_cm"]) <= 0:
                    continue
                q0 = pca["centroid"] - vec * float(row["width68_cm"])
                q1 = pca["centroid"] + vec * float(row["width68_cm"])
                fig.add_trace(
                    go.Scatter3d(
                        x=[q0[2], q1[2]],
                        y=[q0[1], q1[1]],
                        z=[q0[0], q1[0]],
                        mode="lines",
                        line=dict(color=color_hex, width=max(2, int(axis_width) // 2)),
                        name=f"width comp {row['component_id']}",
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

        if show_centroids:
            centroid = pca["centroid"]
            mode = "markers+text" if show_infigure_labels else "markers"
            text = [f"C{row['component_id']}"] if show_infigure_labels else None
            fig.add_trace(
                go.Scatter3d(
                    x=[centroid[2]],
                    y=[centroid[1]],
                    z=[centroid[0]],
                    mode=mode,
                    marker=dict(size=float(centroid_size), color=color_hex, symbol="diamond", opacity=1.0),
                    text=text,
                    textposition="top center",
                    name=f"centroid comp {row['component_id']}",
                    hovertext=(
                        f"component={row['component_id']}<br>"
                        f"centroid x={centroid[0]:.2f}<br>"
                        f"centroid y={centroid[1]:.2f}<br>"
                        f"centroid z={centroid[2]:.2f}"
                    ),
                    hoverinfo="text+x+y+z",
                    showlegend=False,
                )
            )

    mode_note = "component grouping only" if not show_traces and not show_infigure_labels else "PCA diagnostic view"
    fig.update_layout(
        title=(
            f"TPC {tpc} donor components at overflowing t0={old_t0}"
            f"<br><sup>{mode_note}</sup>"
        ),
        height=980,
        margin=dict(l=0, r=0, b=0, t=85),
        scene=dict(xaxis_title="z", yaxis_title="y", zaxis_title="x"),
        showlegend=True,
    )

    fig.write_html(save_path)
    print(f"Saved html: {save_path}")

    if show:
        fig.show()

    if print_summary:
        print(f"TPC {tpc} | old_t0={old_t0} | donor component PCA summary")
        print_component_pca_summary(pca_rows)

    return {
        "TPCid": tpc,
        "old_t0": old_t0,
        "pca_rows": pca_rows,
        "donor_components": donor_components,
        "fig": fig,
        "save_path": save_path,
    }


def plot_light_repair_donor_components_pca_from_namespace(
    namespace: dict[str, Any],
    light_repair_result: dict[str, Any],
    *,
    t0_match_ticks: float = 10.0,
    save_path: str | None = None,
    show: bool = True,
    show_context: bool = True,
    show_traces: bool = True,
    show_infigure_labels: bool = True,
    show_centroids: bool = True,
    print_summary: bool = True,
) -> dict[str, Any]:
    """
    Notebook convenience wrapper.
    """
    return plot_light_repair_donor_components_pca(
        light_repair_result,
        hit_timestamps=namespace["hit_timestamps"],
        hitTPCid=namespace["hitTPCid"],
        labels_global=namespace["labels_global"],
        xset=namespace["xset"],
        yset=namespace["yset"],
        zset=namespace["zset"],
        Eset=namespace["Eset"],
        t0_match_ticks=t0_match_ticks,
        save_path=save_path,
        show=show,
        show_context=show_context,
        show_traces=show_traces,
        show_infigure_labels=show_infigure_labels,
        show_centroids=show_centroids,
        print_summary=print_summary,
    )


__all__ = [
    "summarize_light_repair_donor_components",
    "print_component_pca_summary",
    "plot_light_repair_donor_components_pca",
    "plot_light_repair_donor_components_pca_from_namespace",
]
