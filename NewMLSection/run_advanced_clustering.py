#!/usr/bin/env python3
"""
run_advanced_clustering.py
==========================
End-to-end evaluation of all four advanced clustering methods against
truth vertex information from the NDLAr FLOW HDF5 files.

For each selected event:
  1. Load charge hits (x, y, z, E, io_group)
  2. Load per-hit truth vertex_id via the MC backtrack
  3. Run: NHC, HCA, DVFS, SPEC  +  baseline DBSCAN (from global_track_clustering)
  4. Compute purity, efficiency, ARI, split-rate, merge-rate
  5. Save one HTML 3D plot per (event × method)
  6. Save a metrics table and summary dashboard per event
  7. Save a cross-event summary HTML

All outputs go into  ./advanced_clustering_results/
"""

import os, sys, argparse, warnings
import numpy as np
import h5py
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# ── add NewMLSection to path ──────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from advanced_clustering import (
    build_advanced_labels,
    plot_clustering_result,
    plot_truth_result,
    plot_metrics_table,
    _PALETTE,
)
from global_track_clustering import build_global_labels   # DBSCAN baseline

# ─────────────────────────────────────────────────────────────────────────────
#  Default HDF5 file
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_H5 = os.path.join(
    os.path.dirname(_HERE),
    "Tutorial.flow.0000000.FLOW.hdf5",
)

OUT_DIR = os.path.join(_HERE, "advanced_clustering_results")

METHODS = [
    ('nhc',  'NHC (Graph/Neigh.-Aware)',       {}),
    ('hca',  'HCA (Hier. Complete-Link)',       {}),
    ('dvfs', 'DVFS (DCA Vertex Seeding)',       {}),
    ('spec', 'SPEC (Spectral / SPINEX-like)',   {}),
]


# ─────────────────────────────────────────────────────────────────────────────
#  Truth loading
# ─────────────────────────────────────────────────────────────────────────────

def build_segment_to_vertex_map(h5):
    """
    Build a numpy array `seg2vtx` where seg2vtx[segment_row] = vertex_id.
    The backtrack stores segment_ids as row indices into segments/data.
    """
    segs = h5['mc_truth/segments/data']
    vertex_ids = segs['vertex_id'][:]          # shape (N_segs,)
    return vertex_ids                          # row i → vertex_id


def load_truth_vertex_ids(h5, hit_indices, seg2vtx):
    """
    Given the global row indices of hits in charge/calib_prompt_hits/data,
    return per-hit truth vertex_id (-1 = no truth).

    Path:
      backtrack/data[hit_idx]['segment_ids'][0]  → segment row index
      seg2vtx[segment_row]                        → vertex_id
    """
    bt_data = h5['mc_truth/calib_prompt_hit_backtrack/data']
    bt = bt_data[hit_indices]              # structured array (N_hits,)
    # Primary segment id (first slot that is >= 0)
    seg_ids_2d = bt['segment_ids']         # (N_hits, 20)

    n = len(hit_indices)
    vtx_ids = np.full(n, -1, dtype=np.int64)
    for k in range(seg_ids_2d.shape[1]):
        col = seg_ids_2d[:, k]
        valid = (col >= 0) & (vtx_ids < 0)
        if not np.any(valid):
            break
        ok = valid & (col < len(seg2vtx))
        vtx_ids[ok] = seg2vtx[col[ok]]

    # Compact vertex_ids to 0-based integers (keep -1 for no-truth)
    uniq = np.unique(vtx_ids[vtx_ids >= 0])
    if uniq.size > 0:
        v_map = {v: i for i, v in enumerate(uniq)}
        mapped = np.full(n, -1, dtype=int)
        for raw, compact in v_map.items():
            mapped[vtx_ids == raw] = compact
        return mapped
    return np.full(n, -1, dtype=int)


# ─────────────────────────────────────────────────────────────────────────────
#  Event loading
# ─────────────────────────────────────────────────────────────────────────────

def load_event_hits(h5, charge_event_id, hits_dset='calib_prompt_hits'):
    """Return (x, y, z, E, io_group, global_hit_indices) for a charge event."""
    hits_ref = h5[f'charge/events/ref/charge/{hits_dset}/ref'][:]
    mask     = hits_ref[:, 0] == charge_event_id
    if not np.any(mask):
        return None
    global_indices = hits_ref[mask, 1].astype(np.int64)
    hits_full = h5[f'charge/{hits_dset}/data']
    hits = hits_full[global_indices]
    x        = hits['x'].astype(np.float64)
    y        = hits['y'].astype(np.float64)
    z        = hits['z'].astype(np.float64)
    E        = hits['E'].astype(np.float64)
    io_group = hits['io_group'].astype(np.int32)
    return x, y, z, E, io_group, global_indices


def find_multi_vertex_events(h5, min_vertices=2, max_events=20,
                              hits_dset='calib_prompt_hits',
                              min_hits=80):
    """
    Scan through charge events and return IDs of events that have
    >= min_vertices distinct true vertex_ids and >= min_hits hits.
    These are the most interesting for clustering evaluation.
    """
    hits_ref   = h5[f'charge/events/ref/charge/{hits_dset}/ref'][:]
    bt_data    = h5['mc_truth/calib_prompt_hit_backtrack/data']
    seg_data   = h5['mc_truth/segments/data']
    seg2vtx    = seg_data['vertex_id'][:]

    unique_events = np.unique(hits_ref[:, 0])
    selected      = []

    print(f"[scan] Checking {min(500, len(unique_events))} events …")
    for cev in unique_events[:500]:
        row_mask  = hits_ref[:, 0] == cev
        gidx      = hits_ref[row_mask, 1].astype(np.int64)
        if len(gidx) < min_hits:
            continue
        bt        = bt_data[gidx]
        seg_ids   = bt['segment_ids'][:, 0]
        valid     = (seg_ids >= 0) & (seg_ids < len(seg2vtx))
        vtx_ids   = seg2vtx[seg_ids[valid]]
        n_vtx     = len(np.unique(vtx_ids[vtx_ids >= 0]))
        if n_vtx >= min_vertices:
            selected.append((cev, n_vtx, len(gidx)))
        if len(selected) >= max_events:
            break

    selected.sort(key=lambda t: -t[1])  # most vertices first
    print(f"[scan] Found {len(selected)} qualifying events.")
    return selected


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(reco_labels, truth_labels):
    """
    Compute clustering quality metrics.

    Returns a dict with:
      n_reco_clusters   – number of reco clusters (excluding noise)
      n_truth_vertices  – number of true vertices
      purity            – weighted average purity (Σ_c max_v overlap / N_assigned)
      efficiency        – fraction of true vertex hits captured in a dominant cluster
      split_rate        – fraction of true vertices split across ≥2 reco clusters
      merge_rate        – fraction of reco clusters containing ≥2 true vertices
      ARI               – Adjusted Rand Index (assigned hits only)
      noise_frac        – fraction of hits left as noise (-1)
    """
    from sklearn.metrics import adjusted_rand_score

    r = np.asarray(reco_labels,  int)
    t = np.asarray(truth_labels, int)

    # Work only on hits that have a truth assignment
    valid = t >= 0
    r_v   = r[valid]
    t_v   = t[valid]

    n_truth = len(np.unique(t_v[t_v >= 0]))
    reco_uniq = [l for l in np.unique(r_v) if l >= 0]
    n_reco    = len(reco_uniq)

    noise_frac = float(np.mean(r < 0))

    if n_reco == 0 or n_truth == 0:
        return dict(n_reco_clusters=n_reco, n_truth_vertices=n_truth,
                    purity=0, efficiency=0, split_rate=1, merge_rate=0,
                    ARI=0, noise_frac=noise_frac)

    # Purity
    total_assigned = np.sum(r_v >= 0)
    purity_sum = 0.0
    for rc in reco_uniq:
        m = r_v == rc
        if not np.any(m):
            continue
        tc, cnts = np.unique(t_v[m], return_counts=True)
        purity_sum += cnts.max()
    purity = purity_sum / max(total_assigned, 1)

    # Efficiency: for each true vertex, what fraction of its hits go to
    # the dominant reco cluster?
    eff_vals = []
    for tv in np.unique(t_v):
        if tv < 0:
            continue
        m = t_v == tv
        rc_in_m = r_v[m]
        assigned_here = rc_in_m[rc_in_m >= 0]
        if len(assigned_here) == 0:
            eff_vals.append(0.0)
            continue
        rc, cnts = np.unique(assigned_here, return_counts=True)
        eff_vals.append(cnts.max() / m.sum())
    efficiency = float(np.mean(eff_vals)) if eff_vals else 0.0

    # Split rate: fraction of true vertices that appear in ≥2 reco clusters
    split_count = 0
    for tv in np.unique(t_v):
        if tv < 0:
            continue
        m = t_v == tv
        rc_here = np.unique(r_v[m & (r_v >= 0)])
        if len(rc_here) >= 2:
            split_count += 1
    split_rate = split_count / max(n_truth, 1)

    # Merge rate: fraction of reco clusters spanning ≥2 true vertices
    merge_count = 0
    for rc in reco_uniq:
        m = r_v == rc
        tv_here = np.unique(t_v[m & (t_v >= 0)])
        if len(tv_here) >= 2:
            merge_count += 1
    merge_rate = merge_count / max(n_reco, 1)

    # ARI on assigned hits
    assigned = (r_v >= 0)
    ari = float(adjusted_rand_score(t_v[assigned], r_v[assigned])) if assigned.sum() > 1 else 0.0

    return dict(
        n_reco_clusters  = n_reco,
        n_truth_vertices = n_truth,
        purity           = round(purity,    4),
        efficiency       = round(efficiency, 4),
        split_rate       = round(split_rate, 4),
        merge_rate       = round(merge_rate, 4),
        ARI              = round(ari,        4),
        noise_frac       = round(noise_frac, 4),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Per-event HTML: 3D scatter + track fits
# ─────────────────────────────────────────────────────────────────────────────

def _color_for(lab):
    return _PALETTE[int(lab) % len(_PALETTE)]


def make_event_gallery_html(event_id, results_dict, truth_labels,
                             x, y, z, metrics_rows, plot_dir):
    """
    Write one self-contained HTML file per event that shows:
      • Truth panel
      • One panel per method (NHC, HCA, DVFS, SPEC, DBSCAN-baseline)
      • A metrics comparison table
    """
    method_names   = ['truth'] + [tag for tag, _, _, _ in results_dict.values()]
    method_labels  = [truth_labels] + [v[0] for v in results_dict.values()]
    method_titles  = ['Truth vertex IDs'] + [v[1] for v in results_dict.values()]

    # Build individual plots then combine into one HTML via iframes trick
    # (Plotly can't do true subplots with 3D scenes, so we write individual
    #  divs and combine in a styled wrapper.)
    plots_html = []
    for mlab, mlabels, mtitle in zip(method_names, method_labels, method_titles):
        if mlab == 'truth':
            fig = plot_truth_result(x, y, z, mlabels, title=mtitle)
        else:
            fig = plot_clustering_result(x, y, z, mlabels,
                                          title=f"Event {event_id} — {mtitle}")
        plots_html.append(fig.to_html(full_html=False, include_plotlyjs='cdn'
                                       if len(plots_html) == 0 else False))

    # Metrics table
    df = pd.DataFrame(metrics_rows)
    fig_tbl = plot_metrics_table(df, title=f"Event {event_id} — Method Comparison")
    tbl_html = fig_tbl.to_html(full_html=False, include_plotlyjs=False)

    # Compose full HTML
    cards = []
    for title, ph in zip(method_titles, plots_html):
        cards.append(f"""
        <div class="card">
          <h3>{title}</h3>
          <div class="plot-wrap">{ph}</div>
        </div>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>Event {event_id} — Advanced Clustering</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{background:#0d0d1a;color:#e0e0f0;font-family:'Segoe UI',sans-serif}}
    header{{padding:24px 32px;background:linear-gradient(135deg,#1a1a3e,#0f3460);
            border-bottom:2px solid #e94560}}
    header h1{{font-size:1.8rem;letter-spacing:.04em}}
    header p{{color:#a0a8c0;margin-top:6px;font-size:.95rem}}
    .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(680px,1fr));
           gap:20px;padding:24px 32px}}
    .card{{background:#12122a;border:1px solid #1e2a4a;border-radius:12px;
           overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,.4)}}
    .card h3{{padding:12px 18px;background:#1a1a3e;font-size:1rem;
              letter-spacing:.03em;border-bottom:1px solid #2a3a6a}}
    .plot-wrap{{padding:8px}}
    .metrics{{padding:24px 32px}}
    footer{{text-align:center;padding:18px;color:#5060a0;font-size:.8rem}}
  </style>
</head>
<body>
<header>
  <h1>🔬 NDLAr Advanced Clustering — Event {event_id}</h1>
  <p>Comparing NHC · HCA · DVFS · SPEC · DBSCAN-baseline against truth vertex IDs</p>
</header>

<div class="grid">{"".join(cards)}</div>

<div class="metrics">
  <h2 style="margin-bottom:12px;color:#e94560">📊 Metrics Comparison</h2>
  {tbl_html}
</div>

<footer>Generated by run_advanced_clustering.py</footer>
</body>
</html>"""

    fname = os.path.join(plot_dir, f"event_{event_id}_gallery.html")
    with open(fname, 'w') as f:
        f.write(html)
    print(f"  [html] {fname}")
    return fname


# ─────────────────────────────────────────────────────────────────────────────
#  Summary dashboard across all events
# ─────────────────────────────────────────────────────────────────────────────

def make_summary_dashboard(all_metrics, event_gallery_links, plot_dir):
    """
    Write a summary HTML that:
     - Shows a bar chart of avg metrics per method
     - Links to per-event galleries
    """
    df = pd.DataFrame(all_metrics)
    if df.empty:
        return

    method_col = 'method_tag'
    metric_cols = ['purity', 'efficiency', 'ARI', 'split_rate', 'merge_rate', 'noise_frac']

    avg = df.groupby(method_col)[metric_cols].mean().reset_index()

    # Bar chart
    colors = {'nhc': '#e94560', 'hca': '#0f9b8e', 'dvfs': '#f5a623',
              'spec': '#7b5ea7', 'dbscan': '#4a90d9'}

    fig = go.Figure()
    for _, row in avg.iterrows():
        tag = row[method_col]
        fig.add_bar(
            name=tag.upper(),
            x=metric_cols,
            y=[row[c] for c in metric_cols],
            marker_color=colors.get(tag, '#aaa'),
        )

    fig.update_layout(
        barmode='group',
        title='Average Metrics Across Events (higher purity/efficiency/ARI = better)',
        paper_bgcolor='#0d0d1a', plot_bgcolor='#12122a',
        font=dict(color='#e0e0f0'),
        legend=dict(bgcolor='#1a1a3e', bordercolor='#2a3a6a'),
        xaxis=dict(gridcolor='#1e2a4a'),
        yaxis=dict(gridcolor='#1e2a4a', range=[0, 1.05]),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    bar_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # Per-event table
    df_disp = df.copy()
    for c in metric_cols:
        df_disp[c] = df_disp[c].map(lambda v: f'{v:.3f}')

    rows_html = ''
    for _, row in df_disp.iterrows():
        ev       = row['event_id']
        ev_link  = event_gallery_links.get(ev, '#')
        tag_cell = f'<td><span class="tag tag-{row[method_col]}">{row[method_col].upper()}</span></td>'
        metric_cells = ''.join(f'<td>{row[c]}</td>' for c in metric_cols)
        rows_html += f'<tr><td><a href="{os.path.basename(ev_link)}">{ev}</a></td>{tag_cell}{metric_cells}</tr>\n'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>Advanced Clustering — Summary</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{background:#0d0d1a;color:#e0e0f0;font-family:'Segoe UI',sans-serif}}
    header{{padding:28px 36px;
            background:linear-gradient(135deg,#1a1a3e 0%,#0f3460 60%,#16213e 100%);
            border-bottom:2px solid #e94560}}
    header h1{{font-size:2rem;letter-spacing:.05em}}
    header p{{color:#a0a8c0;margin-top:8px}}
    section{{padding:24px 36px}}
    section h2{{color:#e94560;margin-bottom:16px;font-size:1.3rem}}
    table{{width:100%;border-collapse:collapse;background:#12122a;
           border-radius:10px;overflow:hidden}}
    th{{background:#1a1a3e;padding:10px 14px;text-align:left;
        font-size:.85rem;letter-spacing:.04em;color:#a0b0d0}}
    td{{padding:9px 14px;border-top:1px solid #1e2a4a;font-size:.88rem}}
    tr:hover td{{background:#1a2640}}
    a{{color:#4a90d9;text-decoration:none}}
    a:hover{{text-decoration:underline}}
    .tag{{padding:2px 8px;border-radius:4px;font-size:.8rem;font-weight:600}}
    .tag-nhc  {{background:#e9456030;color:#e94560}}
    .tag-hca  {{background:#0f9b8e30;color:#0f9b8e}}
    .tag-dvfs {{background:#f5a62330;color:#f5a623}}
    .tag-spec {{background:#7b5ea730;color:#9b7ed0}}
    .tag-dbscan{{background:#4a90d930;color:#4a90d9}}
    .note{{background:#1a1a3e;border-left:4px solid #e94560;padding:12px 16px;
           border-radius:0 6px 6px 0;margin:12px 0;font-size:.9rem;color:#c0d0f0}}
    footer{{text-align:center;padding:20px;color:#404870;font-size:.8rem}}
  </style>
</head>
<body>
<header>
  <h1>⚛ NDLAr Advanced Clustering — Summary Dashboard</h1>
  <p>Four advanced vertex-clustering methods evaluated against MC truth on {df['event_id'].nunique()} events</p>
</header>

<section>
  <h2>📈 Average Performance Across All Events</h2>
  <div class="note">
    <strong>Key design goal:</strong> For charge-light matching, <em>merge-rate</em> (incorrectly
    combining different vertices) is far more damaging than <em>split-rate</em> (a vertex
    fragmented into multiple clusters). Methods with low merge-rate are preferred even if
    split-rate is higher.
  </div>
  {bar_html}
</section>

<section>
  <h2>📋 Per-Event Detailed Results</h2>
  <table>
    <thead>
      <tr>
        <th>Event ID</th><th>Method</th>
        {''.join(f'<th>{c}</th>' for c in metric_cols)}
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
</section>

<footer>Generated by run_advanced_clustering.py · NDLAr-full clustering study</footer>
</body>
</html>"""

    fname = os.path.join(plot_dir, 'summary.html')
    with open(fname, 'w') as f:
        f.write(html)
    print(f"\n[summary] {fname}")
    return fname


# ─────────────────────────────────────────────────────────────────────────────
#  Per-method individual HTML (single event, rich layout)
# ─────────────────────────────────────────────────────────────────────────────

def save_method_html(event_id, method_tag, method_name,
                      x, y, z, reco_labels, truth_labels,
                      global_tracks, metrics, plot_dir):
    """Write a self-contained HTML for one (event, method) combination."""
    fig_reco  = plot_clustering_result(x, y, z, reco_labels, global_tracks,
                                        title=f'{method_name} — Event {event_id}',
                                        split_index=None)
    fig_truth = plot_truth_result(x, y, z, truth_labels,
                                   title=f'Truth — Event {event_id}')

    # Metric cards
    card_items = []
    highlights = {
        'merge_rate':   ('⚠ Merge Rate',  'lower is better (critical for CL-match)', True),
        'split_rate':   ('↕ Split Rate',   'lower is better (tolerable)',             False),
        'purity':       ('✔ Purity',       'higher is better',                         False),
        'efficiency':   ('⚡ Efficiency',  'higher is better',                         False),
        'ARI':          ('🎯 ARI',         'higher is better',                         False),
        'noise_frac':   ('💨 Noise Frac',  'lower is better',                          True),
    }
    for key, (label, desc, invert) in highlights.items():
        val  = metrics.get(key, 0)
        good = (val < 0.25) if invert else (val > 0.65)
        color = '#2ecc71' if good else ('#e74c3c' if (val > 0.5 if invert else val < 0.3) else '#f39c12')
        card_items.append(f"""
        <div class="mcard" style="border-top:3px solid {color}">
          <div class="mval" style="color:{color}">{val:.3f}</div>
          <div class="mlabel">{label}</div>
          <div class="mdesc">{desc}</div>
        </div>""")

    reco_html  = fig_reco.to_html(full_html=False,  include_plotlyjs='cdn')
    truth_html = fig_truth.to_html(full_html=False, include_plotlyjs=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>{method_name} — Event {event_id}</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{background:#0d0d1a;color:#e0e0f0;font-family:'Segoe UI',sans-serif}}
    header{{padding:20px 32px;
            background:linear-gradient(135deg,#1a1a3e,#12235a);
            border-bottom:2px solid #e94560;display:flex;align-items:center;gap:20px}}
    header h1{{font-size:1.6rem}}
    header p{{color:#8090c0;font-size:.9rem;margin-top:4px}}
    .metrics-row{{display:flex;gap:14px;flex-wrap:wrap;padding:18px 32px}}
    .mcard{{flex:1;min-width:140px;background:#12122a;border-radius:10px;
            padding:14px 16px;text-align:center}}
    .mval{{font-size:2rem;font-weight:700}}
    .mlabel{{font-size:.85rem;color:#a0b0d0;margin-top:4px;font-weight:600}}
    .mdesc{{font-size:.72rem;color:#5060a0;margin-top:3px}}
    .plots{{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:0 32px 24px}}
    .pane{{background:#12122a;border-radius:12px;overflow:hidden;
           border:1px solid #1e2a4a}}
    .pane h3{{padding:10px 16px;background:#1a1a3e;font-size:.95rem;
              border-bottom:1px solid #2a3a6a}}
    footer{{text-align:center;padding:16px;color:#404870;font-size:.78rem}}
    .badge{{background:#e94560;color:#fff;border-radius:4px;padding:2px 8px;
            font-size:.75rem;margin-left:8px;vertical-align:middle}}
  </style>
</head>
<body>
<header>
  <div>
    <h1>{method_name}<span class="badge">{method_tag.upper()}</span></h1>
    <p>Event {event_id} &nbsp;|&nbsp;
       {metrics.get('n_reco_clusters',0)} reco clusters &nbsp;|&nbsp;
       {metrics.get('n_truth_vertices',0)} true vertices</p>
  </div>
</header>

<div class="metrics-row">{"".join(card_items)}</div>

<div class="plots">
  <div class="pane">
    <h3>Reconstruction: {method_name}</h3>
    {reco_html}
  </div>
  <div class="pane">
    <h3>MC Truth Vertices</h3>
    {truth_html}
  </div>
</div>

<footer>NDLAr Advanced Clustering Study · run_advanced_clustering.py</footer>
</body>
</html>"""

    fname = os.path.join(plot_dir, f"event_{event_id}_{method_tag}.html")
    with open(fname, 'w') as f:
        f.write(html)
    print(f"    → {fname}")
    return fname


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def run(h5_path, n_events=4, out_dir=OUT_DIR, events_override=None,
        min_hits=80, min_vertices=2):

    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  NDLAr Advanced Clustering Evaluation")
    print(f"  HDF5 : {h5_path}")
    print(f"  Out  : {out_dir}")
    print(f"{'='*60}\n")

    h5 = h5py.File(h5_path, 'r')
    seg2vtx = build_segment_to_vertex_map(h5)

    # Select events
    if events_override:
        event_list = [(cev, -1, -1) for cev in events_override]
    else:
        event_list = find_multi_vertex_events(
            h5, min_vertices=min_vertices, max_events=n_events,
            min_hits=min_hits,
        )
        event_list = event_list[:n_events]

    if not event_list:
        print("[ERROR] No qualifying events found. Try lowering min_hits/min_vertices.")
        h5.close()
        return

    all_metrics        = []
    event_gallery_links = {}

    for cev, n_vtx, n_hits in event_list:
        print(f"\n{'─'*50}")
        print(f"  Event {cev}  ({n_vtx} true vertices, {n_hits} hits)")
        print(f"{'─'*50}")

        result = load_event_hits(h5, cev)
        if result is None:
            print(f"  [skip] No hits for event {cev}")
            continue
        x, y, z, E, io_group, gidx = result

        # Truth vertex IDs per hit
        truth_labels = load_truth_vertex_ids(h5, gidx, seg2vtx)
        n_true = len(np.unique(truth_labels[truth_labels >= 0]))
        print(f"  Hits: {len(x)}   True vertices: {n_true}")

        # Save truth plot
        fig_truth = plot_truth_result(x, y, z, truth_labels,
                                       title=f'Truth — Event {cev}')
        fig_truth.write_html(os.path.join(out_dir, f"event_{cev}_truth.html"))

        # ── DBSCAN baseline ──────────────────────────────────────────────
        print("  [baseline] DBSCAN …", end=' ', flush=True)
        try:
            db_labels, db_si = build_global_labels(
                x, y, z, io_group,
                plotting=False,
            )
            db_gt = []
            db_metrics = compute_metrics(db_labels, truth_labels)
            print(f"ARI={db_metrics['ARI']:.3f}  merge={db_metrics['merge_rate']:.3f}")
        except Exception as ex:
            print(f"FAILED ({ex})")
            db_labels  = np.full(len(x), -1, dtype=int)
            db_gt      = []
            db_metrics = dict(n_reco_clusters=0, n_truth_vertices=n_true,
                              purity=0, efficiency=0, split_rate=1,
                              merge_rate=0, ARI=0, noise_frac=1)

        save_method_html(cev, 'dbscan', 'DBSCAN Baseline',
                          x, y, z, db_labels, truth_labels, db_gt,
                          db_metrics, out_dir)
        all_metrics.append({'event_id': cev, 'method_tag': 'dbscan',
                             'method_name': 'DBSCAN Baseline', **db_metrics})

        # ── Advanced methods ─────────────────────────────────────────────
        gallery_results = {}
        gallery_results['dbscan'] = (db_labels, 'DBSCAN Baseline', db_gt, db_metrics)

        for m_tag, m_name, m_kwargs in METHODS:
            print(f"  [{m_tag}] {m_name} …", end=' ', flush=True)
            try:
                gt_out = []
                reco_labels, si = build_advanced_labels(
                    x, y, z, io_group,
                    method=m_tag,
                    method_kwargs=m_kwargs,
                    plotting=False,
                    global_tracks_out=gt_out,
                )
                metrics = compute_metrics(reco_labels, truth_labels)
                print(f"ARI={metrics['ARI']:.3f}  "
                      f"merge={metrics['merge_rate']:.3f}  "
                      f"split={metrics['split_rate']:.3f}  "
                      f"clusters={metrics['n_reco_clusters']}")
            except Exception as ex:
                import traceback
                print(f"FAILED ({ex})")
                traceback.print_exc()
                reco_labels = np.full(len(x), -1, dtype=int)
                gt_out      = []
                metrics     = dict(n_reco_clusters=0, n_truth_vertices=n_true,
                                   purity=0, efficiency=0, split_rate=1,
                                   merge_rate=0, ARI=0, noise_frac=1)

            save_method_html(cev, m_tag, m_name,
                              x, y, z, reco_labels, truth_labels, gt_out,
                              metrics, out_dir)
            all_metrics.append({'event_id': cev, 'method_tag': m_tag,
                                 'method_name': m_name, **metrics})
            gallery_results[m_tag] = (reco_labels, m_name, gt_out, metrics)

        # ── Per-event gallery ────────────────────────────────────────────
        metrics_rows = [
            {'Method': v[1], **{k: vv for k, vv in v[3].items()}}
            for v in gallery_results.values()
        ]
        gallery_link = make_event_gallery_html(
            cev,
            {k: (v[0], v[1], v[2], v[3]) for k, v in gallery_results.items()},
            truth_labels, x, y, z, metrics_rows, out_dir,
        )
        event_gallery_links[cev] = gallery_link

    # ── Summary dashboard ────────────────────────────────────────────────────
    h5.close()
    make_summary_dashboard(all_metrics, event_gallery_links, out_dir)

    # ── Print final table ────────────────────────────────────────────────────
    df = pd.DataFrame(all_metrics)
    if not df.empty:
        print("\n" + "="*70)
        print("SUMMARY (average over all events)")
        print("="*70)
        mc = ['purity', 'efficiency', 'ARI', 'split_rate', 'merge_rate']
        avg = df.groupby('method_tag')[mc].mean().round(4)
        print(avg.to_string())
    print(f"\n✓  All outputs in: {out_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='NDLAr Advanced Clustering Evaluation')
    ap.add_argument('--h5',       default=DEFAULT_H5,  help='Path to FLOW HDF5 file')
    ap.add_argument('--n-events', default=4, type=int,  help='Number of events to process')
    ap.add_argument('--out-dir',  default=OUT_DIR,      help='Output directory for HTML files')
    ap.add_argument('--min-hits', default=80, type=int, help='Minimum hits per event')
    ap.add_argument('--min-vtx',  default=2,  type=int, help='Minimum true vertices per event')
    ap.add_argument('--events',   default='', type=str,
                    help='Comma-separated list of specific charge event IDs to process')
    args = ap.parse_args()

    ev_override = None
    if args.events:
        ev_override = [int(e.strip()) for e in args.events.split(',') if e.strip()]

    run(
        h5_path        = args.h5,
        n_events       = args.n_events,
        out_dir        = args.out_dir,
        events_override= ev_override,
        min_hits       = args.min_hits,
        min_vertices   = args.min_vtx,
    )
