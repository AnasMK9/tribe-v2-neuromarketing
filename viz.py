"""
Visualizations for the Gradio app.

- brain_surface_png: render a static PNG of activation on fsaverage5 cortical mesh
                     (lateral + medial views, both hemispheres)
- timeline_figure:   interactive plotly timeline of per-ROI z-scored activation
                     with spike markers
- radar_figure:      plotly polar chart of the 4 marketing scores
- leaderboard_figure: horizontal bar chart for multi-creative comparison
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from scoring import CLUSTER_COLORS

if TYPE_CHECKING:
    from scoring import RoiTimeseries
    from spikes import Spike


DARK_BG = "#0D1117"
PANEL_BG = "#161B22"
TEXT = "#E6EDF3"
MUTED = "#8B949E"
GRID = "#21262D"


def brain_surface_png(predictions: np.ndarray, out_path: str | Path) -> Path:
    """Render summary brain activation as a PNG.

    Uses the top-20% peak activation per vertex (averaged over time) so the
    static image shows where the creative consistently activated the brain,
    not just noise.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nilearn import datasets, plotting

    n_t, n_v = predictions.shape
    per_hemi = n_v // 2
    top_k = max(1, n_t // 5)
    peak_vertex = np.sort(predictions, axis=0)[-top_k:].mean(axis=0)

    lh = peak_vertex[:per_hemi]
    rh = peak_vertex[per_hemi:per_hemi * 2]

    fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")

    fig = plt.figure(figsize=(12, 7), facecolor=DARK_BG)
    axes_config = [
        (fsaverage["infl_left"], lh, fsaverage["sulc_left"], "left",  "lateral", (0, 0)),
        (fsaverage["infl_left"], lh, fsaverage["sulc_left"], "left",  "medial",  (0, 1)),
        (fsaverage["infl_right"], rh, fsaverage["sulc_right"], "right", "lateral", (1, 0)),
        (fsaverage["infl_right"], rh, fsaverage["sulc_right"], "right", "medial",  (1, 1)),
    ]

    vmax = float(np.percentile(np.abs(peak_vertex), 98)) or 1.0

    for mesh, data, bg, hemi, view, (r, c) in axes_config:
        ax = fig.add_subplot(2, 2, r * 2 + c + 1, projection="3d", facecolor=DARK_BG)
        plotting.plot_surf_stat_map(
            mesh,
            data,
            hemi=hemi,
            view=view,
            bg_map=bg,
            colorbar=(r == 1 and c == 1),
            cmap="inferno",
            vmax=vmax,
            axes=ax,
            figure=fig,
        )
        ax.set_title(f"{hemi.title()} · {view}", color=TEXT, fontsize=10)

    fig.suptitle("Peak cortical activation (fsaverage5)", color=TEXT, fontsize=13, y=0.98)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    return out


def timeline_figure(rts: "RoiTimeseries", spikes: list["Spike"]) -> go.Figure:
    """Interactive plotly timeline of per-ROI activation with spike markers."""
    fig = go.Figure()

    for ci, cluster in enumerate(rts.cluster_names):
        fig.add_trace(go.Scatter(
            x=rts.t_seconds,
            y=rts.activation[:, ci],
            mode="lines",
            name=cluster,
            line=dict(color=CLUSTER_COLORS.get(cluster, "#888"), width=2),
            hovertemplate=f"<b>{cluster}</b><br>t=%{{x:.1f}}s<br>z=%{{y:.2f}}<extra></extra>",
        ))

    # Spike markers
    if spikes:
        for sp in spikes:
            fig.add_trace(go.Scatter(
                x=[sp.t_peak],
                y=[sp.peak_z],
                mode="markers",
                marker=dict(
                    size=12,
                    color=CLUSTER_COLORS.get(sp.cluster, "#fff"),
                    line=dict(color="#fff", width=1.5),
                    symbol="star",
                ),
                name=f"{sp.cluster} spike",
                showlegend=False,
                hovertemplate=(
                    f"<b>{sp.cluster} SPIKE</b><br>"
                    f"peak at %{{x:.1f}}s<br>z=%{{y:.2f}}<extra></extra>"
                ),
            ))

    fig.update_layout(
        title=dict(text="Per-region activation over video time", font=dict(color=TEXT)),
        xaxis=dict(title="Time (s)", color=MUTED, gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(title="z-score (vs video baseline)", color=MUTED, gridcolor=GRID, zerolinecolor=GRID),
        plot_bgcolor=PANEL_BG,
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT, family="monospace"),
        hovermode="x unified",
        legend=dict(bgcolor=PANEL_BG, bordercolor=GRID),
        margin=dict(l=60, r=30, t=60, b=50),
        height=420,
    )
    return fig


def radar_figure(marketing_scores: dict) -> go.Figure:
    """Polar chart of the 4 composite marketing scores (0-100)."""
    labels = ["Attention", "Engagement", "Comprehension", "Memorability"]
    keys = ["attention_score", "engagement_score", "comprehension_score", "memorability_score"]
    values = [marketing_scores.get(k, 0) for k in keys]
    # close the polygon
    values_closed = values + values[:1]
    labels_closed = labels + labels[:1]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(78, 205, 196, 0.25)",
        line=dict(color="#4ECDC4", width=2),
        name="Score",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=PANEL_BG,
            radialaxis=dict(range=[0, 100], color=MUTED, gridcolor=GRID, showticklabels=True),
            angularaxis=dict(color=TEXT, gridcolor=GRID),
        ),
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT, family="monospace"),
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        height=380,
    )
    return fig


def roi_bar_figure(roi_scores: dict) -> go.Figure:
    """Horizontal bar chart of per-ROI cluster scores."""
    names = list(roi_scores.keys())
    values = list(roi_scores.values())
    colors = [CLUSTER_COLORS.get(n, "#888") for n in names]

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker=dict(color=colors, line=dict(color=GRID, width=0.5)),
        text=[f"{v:.0f}" for v in values],
        textposition="outside",
        textfont=dict(color=TEXT),
    ))
    fig.update_layout(
        title=dict(text="ROI cluster activation (0-100)", font=dict(color=TEXT)),
        xaxis=dict(range=[0, 110], color=MUTED, gridcolor=GRID),
        yaxis=dict(color=TEXT, autorange="reversed"),
        plot_bgcolor=PANEL_BG,
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT, family="monospace"),
        margin=dict(l=150, r=30, t=50, b=40),
        height=340,
    )
    return fig


def leaderboard_figure(rows: list[dict]) -> go.Figure:
    """For the comparison tab: horizontal bar of overall score per creative."""
    rows = sorted(rows, key=lambda r: r["overall"])
    names = [r["name"] for r in rows]
    values = [r["overall"] for r in rows]
    norm = [v / 100 for v in values]
    # red→green gradient
    import matplotlib.cm as cm
    rgba = [cm.get_cmap("RdYlGn")(n) for n in norm]
    colors = [f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{a:.2f})" for r, g, b, a in rgba]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker=dict(color=colors),
        text=[f"{v:.1f}" for v in values], textposition="outside",
        textfont=dict(color=TEXT),
    ))
    fig.update_layout(
        title=dict(text="Overall Neuromarketing Score", font=dict(color=TEXT)),
        xaxis=dict(range=[0, 110], color=MUTED, gridcolor=GRID),
        yaxis=dict(color=TEXT),
        plot_bgcolor=PANEL_BG,
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT, family="monospace"),
        margin=dict(l=180, r=30, t=50, b=40),
        height=max(240, 60 * len(rows) + 80),
    )
    return fig
