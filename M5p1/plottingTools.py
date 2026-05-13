"""
plottingTools.py
----------------
Plotting utilities for ND LAr waveforms and clustered charge hits.

Functions
---------
- shifted_ultimate_drawer(...): actual vs. shifted predicted waveforms in full TPC layout.
- labeled_ultimate_drawer(...): same as above but without time shift in the title (label-friendly).
- plot_3d_clusters_with_t0(...): 3D Plotly scatter of clusters with t0 shown in legend.
- plot_3d_clusters(...): 3D Plotly scatter of clusters.

Notes
-----
- Colormap uses "cmc.devon" from `cmcrameri`. Importing cmcrameri registers the colormap.
- Requires: numpy, matplotlib, cmasher, cmcrameri, plotly
"""

from __future__ import annotations

from typing import Optional, Dict, Tuple
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# Ensure cmcrameri colormaps are registered, then get a sub-cmap via cmasher helper
import cmcrameri.cm as cmc  # noqa: F401  # imported for side-effect of colormap registration
import cmasher as cmr

import plotly.graph_objects as go

VALID_GROUP_COLORS = [
    ("#1f77b4", "blue"),
    ("#d62728", "red"),
    ("#2ca02c", "green"),
    ("#ff7f0e", "orange"),
    ("#9467bd", "purple"),
    ("#8c564b", "brown"),
    ("#e377c2", "pink"),
    ("#bcbd22", "olive"),
    ("#17becf", "teal"),
    ("#003f5c", "navy"),
    ("#ef5675", "rose"),
    ("#ffa600", "gold"),
]


def _energy_ordered_labels(
    labels: np.ndarray,
    energies: Optional[np.ndarray] = None,
) -> tuple[list[int], dict[int, float]]:
    labels = np.asarray(labels, dtype=int)
    energy_map: dict[int, float] = {}

    valid_labels = [int(label) for label in np.unique(labels) if int(label) >= 0]
    if energies is not None:
        energies = np.asarray(energies, dtype=np.float64)
        if energies.shape[0] != labels.shape[0]:
            raise ValueError("`energies` must have the same length as `labels`.")
        for label in valid_labels:
            energy_map[int(label)] = float(np.nansum(energies[labels == int(label)]))
        ordered = sorted(valid_labels, key=lambda label: (-energy_map[label], label))
    else:
        ordered = sorted(valid_labels)

    if np.any(labels < 0):
        ordered.append(-1)
    return ordered, energy_map


def relabel_groups_by_time(
    labels: np.ndarray,
    t0s: np.ndarray,
    noise_label: int = -1,
) -> tuple[np.ndarray, dict[int, int]]:
    """
    Return a relabeled copy of `labels` where valid groups are renumbered in
    increasing mean-t0 order. The underlying grouping is unchanged.
    """
    labels = np.asarray(labels, dtype=int)
    t0s = np.asarray(t0s, dtype=np.float64)
    relabeled = np.full(labels.shape, int(noise_label), dtype=int)

    valid_labels = [int(label) for label in np.unique(labels) if int(label) != int(noise_label)]
    if len(valid_labels) == 0:
        return relabeled, {}

    mean_t0 = {}
    for label in valid_labels:
        mask = labels == int(label)
        vals = t0s[mask]
        finite_vals = vals[np.isfinite(vals)]
        mean_t0[int(label)] = float(np.nanmean(finite_vals)) if finite_vals.size > 0 else np.inf

    ordered = sorted(valid_labels, key=lambda label: (mean_t0[int(label)], int(label)))
    relabel_map = {int(old): int(new) for new, old in enumerate(ordered)}
    for old, new in relabel_map.items():
        relabeled[labels == int(old)] = int(new)
    return relabeled, relabel_map

def group_hits_by_time(t0_for_all_hits, time_window=5):
    """
    Assigns new labels to hits so that hits with t0 within `time_window` are grouped.
    Hits with non-finite t0 or t0 == -1 are treated as noise and labeled as -1.

    Returns
    -------
    time_labels: np.ndarray, same length as t0_for_all_hits, labels start from 0
    """
    t0s = np.asarray(t0_for_all_hits, dtype=np.float64)
    time_labels = np.full_like(t0s, -1, dtype=int)

    # Only consider finite, non-noise t0s. NaN timestamps should remain noise
    # in the plotting layer instead of being absorbed into the last real group.
    valid = np.isfinite(t0s) & (t0s != -1)
    t0_valid = t0s[valid]

    if t0_valid.size == 0:
        return time_labels  # all noise

    # Sort and cluster by time
    sorted_indices = np.argsort(t0_valid)
    sorted_t0 = t0_valid[sorted_indices]

    cluster_edges = [0]
    for i in range(1, len(sorted_t0)):
        # Split when there is a true gap in the sorted timestamps. Comparing to
        # the previous hit is more stable for broad physical peaks than
        # comparing to the first hit in the current cluster.
        if abs(sorted_t0[i] - sorted_t0[i - 1]) > time_window:
            cluster_edges.append(i)

    # Assign cluster labels
    current_label = 0
    for start, end in zip(cluster_edges, cluster_edges[1:] + [len(sorted_t0)]):
        idxs = sorted_indices[start:end]
        # Map back to original indices in t0s
        original_idxs = np.where(valid)[0][idxs]
        time_labels[original_idxs] = current_label
        current_label += 1

    return time_labels
def shifted_ultimate_drawer(
    actual: np.ndarray,
    predicted: np.ndarray,
    shift: int,
    TPCid: int = 2,
    ySet: Optional[np.ndarray] = None,
    zSet: Optional[np.ndarray] = None,
    xSet: Optional[np.ndarray] = None,
    lookup_table: Optional[Dict[Tuple[int, int, int], Tuple[int, int]]] = None,
    individualScaling: bool = False,
) -> None:
    """
    Plot actual vs shifted predicted waveform in full TPC layout with central charge image.

    Args:
        actual: Ground truth waveform (8, 64, 1000) or (48, 1000).
        predicted: Predicted waveform (8, 64, 1000) or (48, 1000).
        shift: Integer index shift to apply to predicted.
        TPCid: TPC number.
        ySet, zSet, xSet: y/z positions and a scalar per hit for color (e.g., time or amplitude).
        lookup_table: Mapping (TPC, side, vertical index) → (adc, ch).
        individualScaling: If True, use shared y-limits across all channels.
    """
    if lookup_table is None:
        raise ValueError("`lookup_table` is required.")

    cmap = cmr.get_sub_cmap('cmc.devon', 0.13, 0.95)
    is_left = TPCid in (2, 3, 6, 7)

    fig = plt.figure(figsize=(13, 10))
    outer_gs = gridspec.GridSpec(1, 7, width_ratios=[1, 0.7, 1, 0.02, 0.02, 0.1, 0.03], wspace=0.0)
    left_gs = gridspec.GridSpecFromSubplotSpec(24, 1, subplot_spec=outer_gs[0], hspace=0)
    center_ax = fig.add_subplot(outer_gs[1])
    right_gs = gridspec.GridSpecFromSubplotSpec(24, 1, subplot_spec=outer_gs[2], hspace=0)
    cbar_ax = fig.add_subplot(outer_gs[6])

    is_flattened = (actual.ndim == 2 and actual.shape[0] == 48)

    # Compute global y-limits if needed
    if individualScaling:
        all_vals = []
        for i in range(24):
            vert_pos = 23 - i
            adc_L, ch_L = lookup_table[(TPCid, 0, vert_pos)]
            adc_R, ch_R = lookup_table[(TPCid, 1, vert_pos)]
            if is_flattened:
                idx_L = 2 * i
                idx_R = 2 * i + 1
                actual_L = actual[idx_L, :]
                actual_R = actual[idx_R, :]
                pred_L = predicted[idx_L, :]
                pred_R = predicted[idx_R, :]
            else:
                actual_L = actual[adc_L, ch_L, :]
                actual_R = actual[adc_R, ch_R, :]
                pred_L = predicted[adc_L, ch_L, :]
                pred_R = predicted[adc_R, ch_R, :]

            if shift > 0:
                shifted_L = np.full_like(pred_L, np.nan, dtype=float)
                shifted_R = np.full_like(pred_R, np.nan, dtype=float)
                shifted_L[shift:] = pred_L[:-shift]
                shifted_R[shift:] = pred_R[:-shift]
            elif shift < 0:
                shifted_L = np.full_like(pred_L, np.nan, dtype=float)
                shifted_R = np.full_like(pred_R, np.nan, dtype=float)
                shifted_L[:shift] = pred_L[-shift:]
                shifted_R[:shift] = pred_R[-shift:]
            else:
                shifted_L = pred_L.astype(float)
                shifted_R = pred_R.astype(float)

            all_vals.extend([actual_L, actual_R, shifted_L, shifted_R])

        stacked = np.concatenate(all_vals)
        global_ymin = np.nanmin(stacked) - 3
        global_ymax = np.nanmax(stacked) + 10

    for i in range(24):
        vert_pos = 23 - i
        adc_L, ch_L = lookup_table[(TPCid, 0, vert_pos)]
        adc_R, ch_R = lookup_table[(TPCid, 1, vert_pos)]

        idx_L = 2 * i
        idx_R = 2 * i + 1

        if is_flattened:
            actual_L = actual[idx_L, :]
            actual_R = actual[idx_R, :]
            pred_L = predicted[idx_L, :]
            pred_R = predicted[idx_R, :]
        else:
            actual_L = actual[adc_L, ch_L, :]
            actual_R = actual[adc_R, ch_R, :]
            pred_L = predicted[adc_L, ch_L, :]
            pred_R = predicted[adc_R, ch_R, :]

        shifted_L = np.full_like(actual_L, np.nan, dtype=float)
        shifted_R = np.full_like(actual_R, np.nan, dtype=float)

        if shift > 0:
            shifted_L[shift:] = pred_L[:-shift]
            shifted_R[shift:] = pred_R[:-shift]
        elif shift < 0:
            shifted_L[:shift] = pred_L[-shift:]
            shifted_R[:shift] = pred_R[-shift:]
        else:
            shifted_L = pred_L.astype(float)
            shifted_R = pred_R.astype(float)

        if individualScaling:
            ymin, ymax = global_ymin, global_ymax
        else:
            ymin = min(np.nanmin(actual_L), np.nanmin(actual_R), np.nanmin(shifted_L), np.nanmin(shifted_R)) - 3
            ymax = max(np.nanmax(actual_L), np.nanmax(actual_R), np.nanmax(shifted_L), np.nanmax(shifted_R)) + 10

        ax_l = fig.add_subplot(left_gs[i])
        ax_l.plot(actual_L, color='#1f77b4')   # default blue
        ax_l.plot(shifted_L, color='#ff7f0e')  # default orange-yellow
        ax_l.set_xlim(0, 1000)
        ax_l.set_ylim(ymin, ymax)
        ax_l.set_yticks([])
        ax_l.set_xticks([] if i < 23 else [0, 250, 500, 750, 1000])
        ax_l.set_ylabel(f'{adc_L}:{ch_L}', rotation=0, labelpad=10, va='center')

        ax_r = fig.add_subplot(right_gs[i])
        ax_r.plot(actual_R, color='#1f77b4')   # default blue
        ax_r.plot(shifted_R, color='#ff7f0e')  # default orange-yellow
        ax_r.set_xlim(0, 1000)
        ax_r.set_ylim(ymin, ymax)
        ax_r.set_yticks([])
        ax_r.set_xticks([] if i < 23 else [0, 250, 500, 750, 1000])
        ax_r.yaxis.set_label_position("right")
        ax_r.set_ylabel(f'{adc_R}:{ch_R}', rotation=0, labelpad=15, va='center')

    if ySet is not None and zSet is not None and xSet is not None:
        if is_left:
            rect = mpatches.Rectangle((-64.5, -65), 64, 130, linewidth=1, edgecolor='blue', facecolor='black', alpha=1)
            center_ax.set_xlim([-64.5, -0.5])
            center_ax.set_xticks([-49, -38, -27, -16])
        else:
            rect = mpatches.Rectangle((0.5, -65), 64, 130, linewidth=1, edgecolor='blue', facecolor='black', alpha=1)
            center_ax.set_xlim([0.5, 64.5])
            center_ax.set_xticks([16, 27, 38, 49])

        center_ax.add_patch(rect)
        scatter = center_ax.scatter(zSet, ySet, c=xSet, cmap=cmap, s=2, marker='s')
        center_ax.set_ylim([-65, 65])
        center_ax.set_yticks([])
        center_ax.set_title(f'Module {TPCid // 2}, TPC {TPCid}', fontsize=14)

        cbar = plt.colorbar(scatter, cax=cbar_ax, orientation='vertical')
        cbar.set_label('xSet Value')

    fig.suptitle(f'Single TPC Event Display (Time Shift Applied = {shift*16}ns)', fontsize=16)
    fig.text(0.5, 0.935, 'Blue: Actual light waveforms from simulation', ha='center', color='#1f77b4', fontsize=12)
    fig.text(0.5, 0.915, 'Yellow: Predicted waveforms from charge readout', ha='center', color='#ff7f0e', fontsize=12)
    plt.show()

def extract_TPC_waveforms(image: np.ndarray, TPCid: int, lookup_table: Dict[Tuple[int,int,int], Tuple[int,int]]) -> np.ndarray:
    """
    Extract waveforms for a single TPC from a full (8, 64, 1000) image.
    Returns (48, 1000) in (L0, R0, L1, R1, ..., L23, R23) order.
    """
    waveforms = []
    for i in range(24):
        vert_pos = 23 - i
        adc_L, ch_L = lookup_table[(TPCid, 0, vert_pos)]
        adc_R, ch_R = lookup_table[(TPCid, 1, vert_pos)]
        waveforms.append(image[adc_L, ch_L, :])
        waveforms.append(image[adc_R, ch_R, :])
    return np.stack(waveforms).astype(np.float32)
def labeled_ultimate_drawer(
    actual: np.ndarray,
    predicted: np.ndarray,
    shift: int = 0,
    TPCid: int = 2,
    ySet: Optional[np.ndarray] = None,
    zSet: Optional[np.ndarray] = None,
    xSet: Optional[np.ndarray] = None,
    lookup_table: Optional[Dict[Tuple[int, int, int], Tuple[int, int]]] = None,
    individualScaling: bool = False,
    labelSets: Optional[object] = None,  # kept for signature compatibility
) -> None:
    """
    Plot actual vs shifted predicted waveform in full TPC layout with central charge image.

    Args:
        actual: Ground truth waveform (8, 64, 1000) or (48, 1000).
        predicted: Predicted waveform (8, 64, 1000) or (48, 1000).
        shift: Integer index shift to apply to predicted.
        TPCid: TPC number.
        ySet, zSet, xSet: y/z positions and a scalar per hit for color (e.g., time or amplitude).
        lookup_table: Mapping (TPC, side, vertical index) → (adc, ch).
        individualScaling: If True, use shared y-limits across all channels.
        labelSets: placeholder (not used).
    """
    if lookup_table is None:
        raise ValueError("`lookup_table` is required.")

    cmap = cmr.get_sub_cmap('cmc.devon', 0.13, 0.95)
    is_left = TPCid in (2, 3, 6, 7)

    fig = plt.figure(figsize=(13, 10))
    outer_gs = gridspec.GridSpec(1, 7, width_ratios=[1, 0.7, 1, 0.02, 0.02, 0.1, 0.03], wspace=0.0)
    left_gs = gridspec.GridSpecFromSubplotSpec(24, 1, subplot_spec=outer_gs[0], hspace=0)
    center_ax = fig.add_subplot(outer_gs[1])
    right_gs = gridspec.GridSpecFromSubplotSpec(24, 1, subplot_spec=outer_gs[2], hspace=0)
    cbar_ax = fig.add_subplot(outer_gs[6])

    is_flattened = (actual.ndim == 2 and actual.shape[0] == 48)

    # Compute global y-limits if needed
    if individualScaling:
        all_vals = []
        for i in range(24):
            vert_pos = 23 - i
            adc_L, ch_L = lookup_table[(TPCid, 0, vert_pos)]
            adc_R, ch_R = lookup_table[(TPCid, 1, vert_pos)]
            if is_flattened:
                idx_L = 2 * i
                idx_R = 2 * i + 1
                actual_L = actual[idx_L, :]
                actual_R = actual[idx_R, :]
                pred_L = predicted[idx_L, :]
                pred_R = predicted[idx_R, :]
            else:
                actual_L = actual[adc_L, ch_L, :]
                actual_R = actual[adc_R, ch_R, :]
                pred_L = predicted[adc_L, ch_L, :]
                pred_R = predicted[adc_R, ch_R, :]

            if shift > 0:
                shifted_L = np.full_like(pred_L, np.nan, dtype=float)
                shifted_R = np.full_like(pred_R, np.nan, dtype=float)
                shifted_L[shift:] = pred_L[:-shift]
                shifted_R[shift:] = pred_R[:-shift]
            elif shift < 0:
                shifted_L = np.full_like(pred_L, np.nan, dtype=float)
                shifted_R = np.full_like(pred_R, np.nan, dtype=float)
                shifted_L[:shift] = pred_L[-shift:]
                shifted_R[:shift] = pred_R[-shift:]
            else:
                shifted_L = pred_L.astype(float)
                shifted_R = pred_R.astype(float)

            all_vals.extend([actual_L, actual_R, shifted_L, shifted_R])

        stacked = np.concatenate(all_vals)
        global_ymin = np.nanmin(stacked) - 3
        global_ymax = np.nanmax(stacked) + 10

    for i in range(24):
        vert_pos = 23 - i
        adc_L, ch_L = lookup_table[(TPCid, 0, vert_pos)]
        adc_R, ch_R = lookup_table[(TPCid, 1, vert_pos)]

        idx_L = 2 * i
        idx_R = 2 * i + 1

        if is_flattened:
            actual_L = actual[idx_L, :]
            actual_R = actual[idx_R, :]
            pred_L = predicted[idx_L, :]
            pred_R = predicted[idx_R, :]
        else:
            actual_L = actual[adc_L, ch_L, :]
            actual_R = actual[adc_R, ch_R, :]
            pred_L = predicted[adc_L, ch_L, :]
            pred_R = predicted[adc_R, ch_R, :]

        shifted_L = np.full_like(actual_L, np.nan, dtype=float)
        shifted_R = np.full_like(actual_R, np.nan, dtype=float)

        if shift > 0:
            shifted_L[shift:] = pred_L[:-shift]
            shifted_R[shift:] = pred_R[:-shift]
        elif shift < 0:
            shifted_L[:shift] = pred_L[-shift:]
            shifted_R[:shift] = pred_R[-shift:]
        else:
            shifted_L = pred_L.astype(float)
            shifted_R = pred_R.astype(float)

        if individualScaling:
            ymin, ymax = global_ymin, global_ymax
        else:
            ymin = min(np.nanmin(actual_L), np.nanmin(actual_R), np.nanmin(shifted_L), np.nanmin(shifted_R)) - 3
            ymax = max(np.nanmax(actual_L), np.nanmax(actual_R), np.nanmax(shifted_L), np.nanmax(shifted_R)) + 10

        ax_l = fig.add_subplot(left_gs[i])
        ax_l.plot(actual_L, color='#1f77b4')
        ax_l.plot(shifted_L, color='#ff7f0e')
        ax_l.set_xlim(0, 1000)
        ax_l.set_ylim(ymin, ymax)
        ax_l.set_yticks([])
        ax_l.set_xticks([] if i < 23 else [0, 250, 500, 750, 1000])
        ax_l.set_ylabel(f'{adc_L}:{ch_L}', rotation=0, labelpad=10, va='center')

        ax_r = fig.add_subplot(right_gs[i])
        ax_r.plot(actual_R, color='#1f77b4')
        ax_r.plot(shifted_R, color='#ff7f0e')
        ax_r.set_xlim(0, 1000)
        ax_r.set_ylim(ymin, ymax)
        ax_r.set_yticks([])
        ax_r.set_xticks([] if i < 23 else [0, 250, 500, 750, 1000])
        ax_r.yaxis.set_label_position("right")
        ax_r.set_ylabel(f'{adc_R}:{ch_R}', rotation=0, labelpad=15, va='center')

    if ySet is not None and zSet is not None and xSet is not None:
        if is_left:
            rect = mpatches.Rectangle((-64.5, -65), 64, 130, linewidth=1, edgecolor='blue', facecolor='black', alpha=1)
            center_ax.set_xlim([-64.5, -0.5])
            center_ax.set_xticks([-49, -38, -27, -16])
        else:
            rect = mpatches.Rectangle((0.5, -65), 64, 130, linewidth=1, edgecolor='blue', facecolor='black', alpha=1)
            center_ax.set_xlim([0.5, 64.5])
            center_ax.set_xticks([16, 27, 38, 49])

        center_ax.add_patch(rect)
        scatter = center_ax.scatter(zSet, ySet, c=xSet, cmap=cmr.get_sub_cmap('cmc.devon', 0.13, 0.95), s=2, marker='s')
        center_ax.set_ylim([-65, 65])
        center_ax.set_yticks([])
        center_ax.set_title(f'Single TPC Event Display', fontsize=14)

        cbar = plt.colorbar(scatter, cax=cbar_ax, orientation='vertical')
        cbar.set_label('xSet Value')

    fig.suptitle(f'Single TPC Event Display', fontsize=16)
    fig.text(0.5, 0.935, 'Blue: Actual light waveforms from simulation', ha='center', color='#1f77b4', fontsize=12)
    fig.text(0.5, 0.915, 'Yellow: Predicted waveforms from charge readout', ha='center', color='#ff7f0e', fontsize=12)
    plt.show()


def plot_3d_clusters_with_t0(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    labels: np.ndarray,
    t0s: np.ndarray,
    energies: Optional[np.ndarray] = None,
    title: str = "3D Cluster Visualization",
    save_path: Optional[str] = None
) -> None:
    """
    Plot clustered charge hits in 3D with per-cluster t0 mean annotated in the legend.
    """
    labels = np.asarray(labels, dtype=int)
    t0s = np.asarray(t0s, dtype=np.float64)
    ordered_labels, energy_map = _energy_ordered_labels(labels, energies=energies)
    _, time_label_map = relabel_groups_by_time(labels, t0s)
    fig = go.Figure()

    color_idx = 0
    for label in ordered_labels:
        mask = labels == label
        if label >= 0:
            color_hex, color_name = VALID_GROUP_COLORS[color_idx % len(VALID_GROUP_COLORS)]
            color_idx += 1
            t0_vals = t0s[mask]
            t0_mean = np.nanmean(t0_vals) if np.any(np.isfinite(t0_vals)) else float('nan')
            energy_text = ""
            if label in energy_map:
                energy_text = f", E = {energy_map[label]:.2f}"
            display_label = int(time_label_map.get(int(label), int(label)))
            legend_label = f"{color_name} - cluster {display_label} (t₀ = {t0_mean:.2f}{energy_text})"
        else:
            color_hex = "gray"
            n_noise = int(np.count_nonzero(mask))
            legend_label = f"gray - noise ({n_noise} hits)"
        fig.add_trace(go.Scatter3d(
            x=z[mask], y=y[mask], z=x[mask],
            mode='markers',
            marker=dict(size=4, color=color_hex, opacity=0.85, line=dict(width=0)),
            name=legend_label
        ))

    fig.update_layout(
        scene=dict(xaxis_title='z', yaxis_title='y', zaxis_title='x'),
        legend=dict(title='Clusters', itemsizing='constant'),
        margin=dict(l=0, r=0, b=0, t=40),
        title=title,
        showlegend=True
    )

    if save_path:
        fig.write_html(save_path)
    fig.show()


def plot_3d_clusters(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    labels: np.ndarray,
    energies: Optional[np.ndarray] = None,
    save: bool = False,
    filename: str = "3d_clusters.html"
) -> None:
    """
    Plot clustered charge hits in 3D using Plotly with color-coded clusters.

    Args:
        x, y, z: Coordinates of charge hits.
        labels: Cluster labels for each hit.
        save: If True, saves the plot as an HTML file.
        filename: Filename to save the plot (only used if save is True).
    """
    fig = go.Figure()

    labels = np.asarray(labels, dtype=int)
    ordered_labels, energy_map = _energy_ordered_labels(labels, energies=energies)

    color_idx = 0
    for label in ordered_labels:
        mask = labels == label
        if label >= 0:
            color_hex, color_name = VALID_GROUP_COLORS[color_idx % len(VALID_GROUP_COLORS)]
            color_idx += 1
            energy_text = ""
            if label in energy_map:
                energy_text = f" | E={energy_map[label]:.2f}"
            name = f"{color_name} - Cluster {label}{energy_text}"
        else:
            color_hex = "gray"
            name = "Noise"
        fig.add_trace(go.Scatter3d(
            x=z[mask], y=y[mask], z=x[mask],  # z-horizontal, y-vertical, x-depth
            mode='markers',
            marker=dict(
                size=4,
                color=color_hex,
                opacity=0.8
            ),
            name=name
        ))

    fig.update_layout(
        scene=dict(xaxis_title='z', yaxis_title='y', zaxis_title='x'),
        margin=dict(l=0, r=0, b=0, t=40),
        title='3D Charge Cluster Visualization'
    )

    if save:
        fig.write_html(filename)
    else:
        fig.show()




from typing import Callable

def comparismPlotter(
    predictedImage: np.ndarray,
    actualImage: np.ndarray,
    ev_id: int,
    TPCid: int,
    *,
    hits_full: np.ndarray,
    hits_ref: np.ndarray,
    lookup_table: Dict[Tuple[int,int,int], Tuple[int,int]],
    individualScaling: bool = False,
    extractor: Optional[Callable[[np.ndarray, int, Dict[Tuple[int,int,int], Tuple[int,int]]], np.ndarray]] = None,
    title: Optional[str] = None,
) -> None:
    """
    Convenience wrapper: extract TPC waveforms for a given event/TPC, gather the
    charge-hit positions for that TPC, and render the combined event display.

    Parameters
    ----------
    predictedImage, actualImage : ndarray
        Full-detector waveforms from your pipeline (whatever `extract_TPC_waveforms`
        expects as its first argument). The extractor should return (48, 1000) for the
        requested TPC.
    ev_id : int
        Event id to visualize (matches hits_ref[:,0]).
    TPCid : int
        TPC number used both for waveform extraction and hit filtering.
    hits_full : structured ndarray
        HDF5 dataset for hits (e.g., h5['charge/<hits_dset>/data']).
    hits_ref : ndarray of shape (N,2)
        HDF5 event index mapping (e.g., h5['charge/events/ref/charge/<hits_dset>/ref']).
    lookup_table : dict
        Mapping (TPC, side, vertical index) -> (adc, ch) required by the drawers.
    individualScaling : bool
        If True, use shared y-limits across all channels (as in the drawers).
    extractor : callable, optional
        Function like `extract_TPC_waveforms(image, TPCid, lookup_table=...) -> (48,1000)`.
        If None, will try to find a global `extract_TPC_waveforms` and use that.
    title : Optional[str]
        If provided, overrides the default title in the center panel.

    Notes
    -----
    - This function does *not* mutate any inputs.
    - We call `labeled_ultimate_drawer` under the hood.
    """
    # Resolve extractor
    if extractor is None:
        # Try to use a global function if present
        g = globals()
        if 'extract_TPC_waveforms' in g and callable(g['extract_TPC_waveforms']):
            extractor = g['extract_TPC_waveforms']
        else:
            raise ValueError("No `extract_TPC_waveforms` provided. Pass `extractor=` or define it globally.")

    # 1) Pull waveforms for this TPC
    TPCactual    = extractor(actualImage,    TPCid, lookup_table=lookup_table)
    TPCpredicted = extractor(predictedImage, TPCid, lookup_table=lookup_table)

    # 2) Pull hits for this event
    hit_mask = (hits_ref[:, 0] == ev_id)
    hit_refs = hits_ref[hit_mask, 1]
    hits_evt = hits_full[hit_refs]

    # 3) Restrict to this charge TPC using io_group, not reconstructed x/y/z.
    io_tpc = (np.asarray(hits_evt['io_group'], dtype=np.int64) - 1) // 2
    tpc_mask = (io_tpc == int(TPCid))

    x_filtered = hits_evt['x'][tpc_mask]
    y_filtered = hits_evt['y'][tpc_mask]
    z_filtered = hits_evt['z'][tpc_mask]

    # 4) Plot
    labeled_ultimate_drawer(
        predicted=TPCpredicted,
        actual=TPCactual,
        ySet=y_filtered,
        zSet=z_filtered,
        xSet=x_filtered,
        shift=0,
        lookup_table=lookup_table,
        TPCid=TPCid,
        individualScaling=individualScaling,
    )

def comparismPlotter_simple(
    predictedImage: np.ndarray,
    actualImage: np.ndarray,
    TPCid: int,
    lookup_table: Dict[Tuple[int, int, int], Tuple[int, int]],
    *,
    xSet: Optional[np.ndarray] = None,
    ySet: Optional[np.ndarray] = None,
    zSet: Optional[np.ndarray] = None,
    hitTPCid: Optional[np.ndarray] = None,
    individualScaling: bool = False,
    shift: int = 0
) -> None:
    """
    Simplified plotter taking direct (48, 1000) waveforms and coordinates.
    skips the heavy extraction/filtering logic of comparismPlotter.
    """
    if hitTPCid is not None and xSet is not None and ySet is not None and zSet is not None:
        mask = (hitTPCid == TPCid)
        xSet = xSet[mask]
        ySet = ySet[mask]
        zSet = zSet[mask]

    labeled_ultimate_drawer(
        actual=actualImage,
        predicted=predictedImage,
        shift=shift,
        TPCid=TPCid,
        ySet=ySet,
        zSet=zSet,
        xSet=xSet,
        lookup_table=lookup_table,
        individualScaling=individualScaling,
    )

__all__ = [
    "comparismPlotter",
    "comparismPlotter_simple",
    "shifted_ultimate_drawer",
    "labeled_ultimate_drawer",
    "plot_3d_clusters_with_t0",
    "plot_3d_clusters",
    "group_hits_by_time"
]
