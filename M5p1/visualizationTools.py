import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as mpatches
try:
    import cmasher as cmr
    import cmasher.cm  # force colormap registration with matplotlib
except ImportError:
    cmr = None
import plotly.graph_objects as go

def plot_tpc_light_and_charge(ev_id, TPCid, hits_full_ev, actual, predicted=None, geom_map=None, individualScaling=False):
    """
    Display charge hits in the center and light waveforms (actual vs. predicted) on the sides for a given TPC.
    
    Layout:
      - Left:  channels 0..59 bottom->top (labels L0..L59)
      - Right: channels 60..119 bottom->top (labels R0..R59)
      - Center: charge scatter (z vs y) with x as color
      - Far right: colorbar
    """
    ROWS_PER_SIDE = 60
    try:
        cmap = cmr.get_sub_cmap('cmc.devon', 0.13, 0.95) if cmr else 'viridis'
    except (KeyError, AttributeError):
        cmap = 'viridis'

    fig = plt.figure(figsize=(14.5, 21))
    outer_gs = gridspec.GridSpec(1, 5, width_ratios=[1.0, 1.6, 1.0, 0.25, 0.06], wspace=0.0)

    left_gs   = gridspec.GridSpecFromSubplotSpec(ROWS_PER_SIDE, 1, subplot_spec=outer_gs[0], hspace=0.0)
    center_ax = fig.add_subplot(outer_gs[1])
    right_gs  = gridspec.GridSpecFromSubplotSpec(ROWS_PER_SIDE, 1, subplot_spec=outer_gs[2], hspace=0.0)
    cbar_ax   = fig.add_subplot(outer_gs[4])

    # Get hits for this TPC
    io_groups = [2 * TPCid + 1, 2 * TPCid + 2]
    hits_ev = hits_full_ev[np.isin(hits_full_ev['io_group'], io_groups)]
    if len(hits_ev) == 0:
        print(f"No hits found in TPC {TPCid}")
        return

    ySet = hits_ev['y']
    zSet = hits_ev['z']
    xSet = hits_ev['x']
    E = np.sum(hits_ev['E'])
    print(f'Total energy in TPC {TPCid}: {E:.2f}')

    if individualScaling:
        global_ymin = np.nanmin(actual)
        global_ymax = np.nanmax(actual) + 10
        if predicted is not None:
            global_ymin = min(global_ymin, float(np.nanmin(predicted))) - 3
            global_ymax = max(global_ymax, float(np.nanmax(predicted))) + 10

    NCH = int(actual.shape[0])
    for i in range(ROWS_PER_SIDE):
        ch_L = i
        ch_R = ROWS_PER_SIDE + i
        if max(ch_L, ch_R) >= NCH:
            break
        row_idx = ROWS_PER_SIDE - 1 - i
        actual_L = actual[ch_L, :]
        actual_R = actual[ch_R, :]
        
        pred_L = predicted[ch_L, :] if predicted is not None else None
        pred_R = predicted[ch_R, :] if predicted is not None else None

        if individualScaling:
            ymin, ymax = global_ymin, global_ymax
        else:
            vals = [actual_L, actual_R]
            if pred_L is not None: vals += [pred_L, pred_R]
            ymin = min(float(np.nanmin(v)) for v in vals) - 3
            ymax = max(float(np.nanmax(v)) for v in vals) + 10

        # Left
        ax_l = fig.add_subplot(left_gs[row_idx])
        ax_l.plot(actual_L, lw=0.8, color='#1f77b4')
        if pred_L is not None: ax_l.plot(pred_L, lw=0.8, color='#ff7f0e')
        ax_l.set_xlim(0, 1000); ax_l.set_ylim(ymin, ymax); ax_l.set_yticks([])
        ax_l.set_xticks([0, 250, 500, 750, 1000] if i == 0 else [])
        ax_l.set_ylabel(f'L{i}', rotation=0, labelpad=10, va='center')

        # Right
        ax_r = fig.add_subplot(right_gs[row_idx])
        ax_r.plot(actual_R, lw=0.8, color='#1f77b4')
        if pred_R is not None: ax_r.plot(pred_R, lw=0.8, color='#ff7f0e')
        ax_r.set_xlim(0, 1000); ax_r.set_ylim(ymin, ymax); ax_r.set_yticks([])
        ax_r.set_xticks([0, 250, 500, 750, 1000] if i == 0 else [])
        ax_r.yaxis.set_label_position("right")
        ax_r.set_ylabel(f'R{i}', rotation=0, labelpad=10, va='center', ha='left')

    # Center (Charge)
    y_fixed = (-215.5, 81.7)
    if geom_map and TPCid in geom_map:
        geom = geom_map[TPCid]
        zlim = (geom.z_min, geom.z_max)
    else:
        zlim = (np.min(zSet)-10, np.max(zSet)+10)

    center_ax.set_xlim(zlim); center_ax.set_ylim(y_fixed); center_ax.set_aspect('auto')
    center_ax.set_facecolor('black')
    border = mpatches.Rectangle((zlim[0], y_fixed[0]), zlim[1] - zlim[0], y_fixed[1] - y_fixed[0],
                                linewidth=1, edgecolor='royalblue', facecolor='none', alpha=0.8)
    center_ax.add_patch(border)
    scatter = center_ax.scatter(zSet, ySet, c=xSet, cmap=cmap, s=2, marker='s')
    center_ax.set_yticks([]); center_ax.set_xlabel("z")
    center_ax.set_title(f'Module {TPCid // 2}, TPC {TPCid}', fontsize=13)

    cbar = plt.colorbar(scatter, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Charge x Position', rotation=90)

    fig.suptitle('Single TPC Event Display', fontsize=16, y=0.93)
    fig.text(0.5, 0.91, 'Blue: Actual light waveforms', ha='center', color='#1f77b4', fontsize=12)
    if predicted is not None:
        fig.text(0.5, 0.90, 'Orange: Predicted waveforms (baseImage)', ha='center', color='#ff7f0e', fontsize=12)
    plt.show()

def plot_tpc_light_and_charge_with_overlay(
    ev_id,
    TPCid,
    hits_full_ev,
    actual,
    predicted=None,
    overlay=None,
    geom_map=None,
    individualScaling=False,
    overlay_label="Trial / repaired waveform",
):
    """
    Same display style as `plot_tpc_light_and_charge`, with a third waveform
    overlaid in red on every light channel.

    Colors:
      - Blue: actual light waveform
      - Orange: current predicted waveform
      - Red: overlay/trial waveform
    """
    ROWS_PER_SIDE = 60
    try:
        cmap = cmr.get_sub_cmap('cmc.devon', 0.13, 0.95) if cmr else 'viridis'
    except (KeyError, AttributeError):
        cmap = 'viridis'

    actual = np.asarray(actual)
    predicted = None if predicted is None else np.asarray(predicted)
    overlay = None if overlay is None else np.asarray(overlay)

    if predicted is not None and predicted.shape != actual.shape:
        raise ValueError(f"predicted shape {predicted.shape} does not match actual shape {actual.shape}")
    if overlay is not None and overlay.shape != actual.shape:
        raise ValueError(f"overlay shape {overlay.shape} does not match actual shape {actual.shape}")

    fig = plt.figure(figsize=(14.5, 21))
    outer_gs = gridspec.GridSpec(1, 5, width_ratios=[1.0, 1.6, 1.0, 0.25, 0.06], wspace=0.0)

    left_gs = gridspec.GridSpecFromSubplotSpec(ROWS_PER_SIDE, 1, subplot_spec=outer_gs[0], hspace=0.0)
    center_ax = fig.add_subplot(outer_gs[1])
    right_gs = gridspec.GridSpecFromSubplotSpec(ROWS_PER_SIDE, 1, subplot_spec=outer_gs[2], hspace=0.0)
    cbar_ax = fig.add_subplot(outer_gs[4])

    io_groups = [2 * TPCid + 1, 2 * TPCid + 2]
    hits_ev = hits_full_ev[np.isin(hits_full_ev['io_group'], io_groups)]
    if len(hits_ev) == 0:
        print(f"No hits found in TPC {TPCid}")
        return

    ySet = hits_ev['y']
    zSet = hits_ev['z']
    xSet = hits_ev['x']
    E = np.sum(hits_ev['E'])
    print(f'Total energy in TPC {TPCid}: {E:.2f}')

    if individualScaling:
        global_ymin = np.nanmin(actual)
        global_ymax = np.nanmax(actual) + 10
        for arr in (predicted, overlay):
            if arr is not None:
                global_ymin = min(global_ymin, float(np.nanmin(arr))) - 3
                global_ymax = max(global_ymax, float(np.nanmax(arr))) + 10

    NCH = int(actual.shape[0])
    for i in range(ROWS_PER_SIDE):
        ch_L = i
        ch_R = ROWS_PER_SIDE + i
        if max(ch_L, ch_R) >= NCH:
            break

        row_idx = ROWS_PER_SIDE - 1 - i
        actual_L = actual[ch_L, :]
        actual_R = actual[ch_R, :]
        pred_L = predicted[ch_L, :] if predicted is not None else None
        pred_R = predicted[ch_R, :] if predicted is not None else None
        over_L = overlay[ch_L, :] if overlay is not None else None
        over_R = overlay[ch_R, :] if overlay is not None else None

        if individualScaling:
            ymin, ymax = global_ymin, global_ymax
        else:
            vals = [actual_L, actual_R]
            if pred_L is not None:
                vals += [pred_L, pred_R]
            if over_L is not None:
                vals += [over_L, over_R]
            ymin = min(float(np.nanmin(v)) for v in vals) - 3
            ymax = max(float(np.nanmax(v)) for v in vals) + 10

        ax_l = fig.add_subplot(left_gs[row_idx])
        ax_l.plot(actual_L, lw=0.8, color='#1f77b4')
        if pred_L is not None:
            ax_l.plot(pred_L, lw=0.8, color='#ff7f0e')
        if over_L is not None:
            ax_l.plot(over_L, lw=0.8, color='#d62728')
        ax_l.set_xlim(0, 1000)
        ax_l.set_ylim(ymin, ymax)
        ax_l.set_yticks([])
        ax_l.set_xticks([0, 250, 500, 750, 1000] if i == 0 else [])
        ax_l.set_ylabel(f'L{i}', rotation=0, labelpad=10, va='center')

        ax_r = fig.add_subplot(right_gs[row_idx])
        ax_r.plot(actual_R, lw=0.8, color='#1f77b4')
        if pred_R is not None:
            ax_r.plot(pred_R, lw=0.8, color='#ff7f0e')
        if over_R is not None:
            ax_r.plot(over_R, lw=0.8, color='#d62728')
        ax_r.set_xlim(0, 1000)
        ax_r.set_ylim(ymin, ymax)
        ax_r.set_yticks([])
        ax_r.set_xticks([0, 250, 500, 750, 1000] if i == 0 else [])
        ax_r.yaxis.set_label_position("right")
        ax_r.set_ylabel(f'R{i}', rotation=0, labelpad=10, va='center', ha='left')

    y_fixed = (-215.5, 81.7)
    if geom_map and TPCid in geom_map:
        geom = geom_map[TPCid]
        zlim = (geom.z_min, geom.z_max)
    else:
        zlim = (np.min(zSet) - 10, np.max(zSet) + 10)

    center_ax.set_xlim(zlim)
    center_ax.set_ylim(y_fixed)
    center_ax.set_aspect('auto')
    center_ax.set_facecolor('black')
    border = mpatches.Rectangle(
        (zlim[0], y_fixed[0]),
        zlim[1] - zlim[0],
        y_fixed[1] - y_fixed[0],
        linewidth=1,
        edgecolor='royalblue',
        facecolor='none',
        alpha=0.8,
    )
    center_ax.add_patch(border)
    scatter = center_ax.scatter(zSet, ySet, c=xSet, cmap=cmap, s=2, marker='s')
    center_ax.set_yticks([])
    center_ax.set_xlabel("z")
    center_ax.set_title(f'Module {TPCid // 2}, TPC {TPCid}', fontsize=13)

    cbar = plt.colorbar(scatter, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Charge x Position', rotation=90)

    fig.suptitle('Single TPC Event Display', fontsize=16, y=0.93)
    fig.text(0.5, 0.91, 'Blue: Actual light waveforms', ha='center', color='#1f77b4', fontsize=12)
    if predicted is not None:
        fig.text(0.5, 0.90, 'Orange: Predicted waveforms (baseImage)', ha='center', color='#ff7f0e', fontsize=12)
    if overlay is not None:
        fig.text(0.5, 0.89, f'Red: {overlay_label}', ha='center', color='#d62728', fontsize=12)
    plt.show()

def plot_tpc_3d_charge(VIS_TPC_ID, hits_evt):
    """
    Interactive Plotly 3D scatter plot of charge hits for a chosen TPC.
    """
    print(f"Plotting 3D charge hits for TPC {VIS_TPC_ID}...")
    io_groups = [2 * VIS_TPC_ID + 1, 2 * VIS_TPC_ID + 2]
    hits_tpc = hits_evt[np.isin(hits_evt['io_group'], io_groups)]

    if len(hits_tpc) > 0:
        fig3d = go.Figure(data=[go.Scatter3d(
            x=hits_tpc['x'],
            y=hits_tpc['y'],
            z=hits_tpc['z'],
            mode='markers',
            marker=dict(
                size=3,
                color=hits_tpc['E'],
                colorscale='Viridis',
                colorbar=dict(title="Energy"),
                opacity=0.8
            ),
            text=[f"E: {e:.2f}<br>io: {io}" for e, io in zip(hits_tpc['E'], hits_tpc['io_group'])],
            hoverinfo='text'
        )])
        fig3d.update_layout(
            title=f"3D Charge Hits - TPC {VIS_TPC_ID}",
            scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        fig3d.show()
    else:
        print(f"No hits found to plot in 3D for TPC {VIS_TPC_ID}.")
