"""
Vertex-level TRIBE v2 predictions → 7 neuromarketing ROI clusters → 4 composite scores.

TRIBE v2 outputs on the fsaverage5 cortical surface (~20,484 vertices: 10,242 per
hemisphere). We map each vertex to one of 7 Yeo-2011 functional networks, then
group networks into the neuromarketing clusters defined in the user's original
notebook.

Important limitation: fsaverage5 is a *cortical* surface — no subcortex. True
amygdala / NAcc / hippocampus signal is not directly captured. The Emotion &
Reward and Memory Encoding clusters here are cortical proxies (Limbic network +
Default network temporal/medial parts), not subcortical ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Neuromarketing ROI clusters, mapped to Yeo-2011 7-network labels.
# Networks: 1=Visual, 2=Somatomotor, 3=DorsAttn, 4=VentAttn, 5=Limbic,
#           6=Control/Frontoparietal, 7=Default
#
# Weights in each cluster sum to 1.0 — they express how strongly a given
# functional network contributes to that marketing-relevant cluster.
# Grounded in the original notebook's cluster→ROI mapping.
# ─────────────────────────────────────────────────────────────────────────────
ROI_CLUSTERS: dict[str, dict[int, float]] = {
    "Visual Attention":   {1: 0.6, 3: 0.4},
    "Object & Scene":     {1: 0.5, 7: 0.5},
    "Face & Social":      {4: 0.4, 7: 0.6},
    "Motion & Dynamics":  {1: 0.5, 2: 0.3, 3: 0.2},
    "Language & Meaning": {6: 0.5, 7: 0.5},
    "Emotion & Reward":   {5: 1.0},
    "Memory Encoding":    {5: 0.4, 7: 0.6},
}

CLUSTER_COLORS = {
    "Visual Attention":   "#4ECDC4",
    "Object & Scene":     "#45B7D1",
    "Face & Social":      "#F7DC6F",
    "Motion & Dynamics":  "#F0A500",
    "Language & Meaning": "#BB8FCE",
    "Emotion & Reward":   "#EC7063",
    "Memory Encoding":    "#58D68D",
}

# Weights for the 4 composite marketing scores (ported verbatim from the
# mock notebook, cell 3).
SCORE_WEIGHTS: dict[str, dict[str, float]] = {
    "attention_score": {
        "Visual Attention":   0.40,
        "Motion & Dynamics":  0.30,
        "Face & Social":      0.20,
        "Object & Scene":     0.10,
    },
    "engagement_score": {
        "Emotion & Reward":   0.35,
        "Face & Social":      0.25,
        "Motion & Dynamics":  0.20,
        "Memory Encoding":    0.20,
    },
    "comprehension_score": {
        "Language & Meaning": 0.50,
        "Object & Scene":     0.30,
        "Memory Encoding":    0.20,
    },
    "memorability_score": {
        "Memory Encoding":    0.45,
        "Emotion & Reward":   0.30,
        "Face & Social":      0.15,
        "Language & Meaning": 0.10,
    },
}


@dataclass
class RoiTimeseries:
    """Per-ROI activation over time.

    activation: shape (n_timesteps, n_clusters) — z-scored within each cluster
                across the full video duration
    t_seconds:  shape (n_timesteps,) — timestamp of each sample
    cluster_names: list of n_clusters names, ordering matches activation columns
    """
    activation: np.ndarray
    t_seconds: np.ndarray
    cluster_names: list[str]


@lru_cache(maxsize=1)
def _yeo_vertex_labels(n_vertices_per_hemi: int = 10242) -> np.ndarray:
    """
    Returns a (2 * n_vertices_per_hemi,) int array in {1..7} giving each
    fsaverage5 vertex's Yeo-2011 network assignment.

    Uses nilearn's fetch_atlas_yeo_2011 which provides a volumetric label image;
    we project to the fsaverage5 surface via nilearn.surface.vol_to_surf.

    Cached — only downloaded/projected once per process.
    """
    from nilearn import datasets, surface

    # API shifted between nilearn versions. Old API: dict with "thick_7" key
    # returning a NIfTI path. New API (0.11+): Atlas object with "maps" attr.
    # We request n_networks=7, thickness="thick" to be explicit either way.
    try:
        yeo = datasets.fetch_atlas_yeo_2011(n_networks=7, thickness="thick")
        atlas_path = yeo["maps"]
    except TypeError:
        yeo = datasets.fetch_atlas_yeo_2011()
        atlas_path = yeo.get("thick_7") or yeo["maps"]

    fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")

    labels_lh = np.asarray(surface.vol_to_surf(
        atlas_path,
        fsaverage["pial_left"],
        interpolation="nearest_most_frequent",
    )).squeeze().astype(int)
    labels_rh = np.asarray(surface.vol_to_surf(
        atlas_path,
        fsaverage["pial_right"],
        interpolation="nearest_most_frequent",
    )).squeeze().astype(int)

    labels = np.concatenate([labels_lh, labels_rh])

    # Unlabeled vertices (medial wall, etc.) get 0 — assign them to Default (7)
    # as a neutral fallback so they don't contribute extreme signal.
    labels[labels == 0] = 7

    if labels.shape[0] != 2 * n_vertices_per_hemi:
        # fsaverage5 should be exactly 20,484. If we got something else, pad/trim
        # defensively rather than crash.
        target = 2 * n_vertices_per_hemi
        if labels.shape[0] > target:
            labels = labels[:target]
        else:
            labels = np.pad(labels, (0, target - labels.shape[0]), constant_values=7)

    return labels


def cluster_masks(n_vertices: int) -> dict[str, np.ndarray]:
    """Vertex → boolean mask for each of the 7 neuromarketing clusters.

    A vertex belongs to a cluster if its Yeo network is in that cluster's
    definition (with positive weight). Vertices can belong to multiple clusters
    (clusters overlap, by design).
    """
    per_hemi = n_vertices // 2
    yeo_labels = _yeo_vertex_labels(per_hemi)

    if yeo_labels.shape[0] != n_vertices:
        # fall back: if projection size differs, assign networks round-robin
        yeo_labels = np.tile(np.arange(1, 8), n_vertices // 7 + 1)[:n_vertices]

    masks = {}
    for cluster, net_weights in ROI_CLUSTERS.items():
        mask = np.zeros(n_vertices, dtype=bool)
        for net_id in net_weights:
            mask |= (yeo_labels == net_id)
        masks[cluster] = mask
    return masks


def aggregate_to_rois(
    predictions: np.ndarray,
    t_seconds: np.ndarray,
) -> RoiTimeseries:
    """Collapse (n_timesteps, n_vertices) → (n_timesteps, n_clusters).

    Weighted mean per cluster using ROI_CLUSTERS network weights.
    Output columns are z-scored across time so every cluster is on the same
    scale regardless of absolute activation magnitude.
    """
    n_t, n_v = predictions.shape
    per_hemi = n_v // 2
    yeo_labels = _yeo_vertex_labels(per_hemi)
    if yeo_labels.shape[0] != n_v:
        yeo_labels = np.tile(np.arange(1, 8), n_v // 7 + 1)[:n_v]

    cluster_names = list(ROI_CLUSTERS.keys())
    out = np.zeros((n_t, len(cluster_names)), dtype=np.float32)

    for ci, cluster in enumerate(cluster_names):
        net_weights = ROI_CLUSTERS[cluster]
        total_weight = 0.0
        cluster_signal = np.zeros(n_t, dtype=np.float32)
        for net_id, w in net_weights.items():
            net_mask = (yeo_labels == net_id)
            if not net_mask.any():
                continue
            # mean activation across vertices of this network
            net_mean = predictions[:, net_mask].mean(axis=1)
            cluster_signal += w * net_mean
            total_weight += w
        if total_weight > 0:
            cluster_signal /= total_weight
        out[:, ci] = cluster_signal

    # z-score each cluster across time so comparisons are scale-free
    mu = out.mean(axis=0, keepdims=True)
    sd = out.std(axis=0, keepdims=True) + 1e-8
    z = (out - mu) / sd

    return RoiTimeseries(activation=z, t_seconds=t_seconds, cluster_names=cluster_names)


def summary_scores(rts: RoiTimeseries) -> dict:
    """Collapse time → per-cluster summary (0–100), then composite marketing scores.

    Returns:
      {
        "roi_scores":  {cluster: 0-100, ...}   # 7 keys
        "marketing":   {attention_score: 0-100, engagement_score: 0-100, ...}
        "overall":     0-100
      }
    """
    # Summary per cluster = mean of top-20% activations across time.
    # Rationale: a creative's "peak engagement moments" matter more than its
    # average. Using top-quintile captures spike-driven signal without being
    # dominated by a single outlier.
    n_t = rts.activation.shape[0]
    top_k = max(1, n_t // 5)
    sorted_act = np.sort(rts.activation, axis=0)  # ascending
    peak_mean = sorted_act[-top_k:].mean(axis=0)  # mean of top 20%

    # Normalize across clusters to 0–100 for display
    mn, mx = peak_mean.min(), peak_mean.max()
    if mx - mn < 1e-6:
        roi_0_100 = {name: 50.0 for name in rts.cluster_names}
    else:
        scaled = (peak_mean - mn) / (mx - mn) * 100
        roi_0_100 = {name: float(round(scaled[i], 1)) for i, name in enumerate(rts.cluster_names)}

    marketing = {}
    for score_name, weights in SCORE_WEIGHTS.items():
        val = sum(roi_0_100.get(cluster, 0.0) * w for cluster, w in weights.items())
        marketing[score_name] = round(val, 1)

    overall = round(sum(marketing.values()) / len(marketing), 1)

    return {"roi_scores": roi_0_100, "marketing": marketing, "overall": overall}


def tier(score: float) -> tuple[str, str]:
    """0-100 score → (tier_label, emoji)."""
    if score < 40:
        return "LOW", "🔴"
    if score < 65:
        return "MED", "🟡"
    return "HIGH", "🟢"
