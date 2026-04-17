"""
Spike detection on per-ROI activation timeseries.

A "spike" is a moment in the video where one ROI cluster's predicted neural
response rises significantly above that cluster's baseline — operationalized
as rolling z-score > 2 sustained for at least one sample.

Each spike returns:
  - ROI cluster name
  - start/end/peak timestamps (seconds)
  - peak z-score
  - path to an extracted frame thumbnail at the peak moment

The frame thumbnail is what makes this useful for a marketer: "at 4.2s,
Face & Social spiked — here's the exact frame that did it."
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from scoring import RoiTimeseries


@dataclass
class Spike:
    cluster: str
    t_start: float
    t_end: float
    t_peak: float
    peak_z: float
    frame_path: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def detect_spikes(
    rts: RoiTimeseries,
    z_threshold: float = 2.0,
    min_samples: int = 1,
    rolling_window_s: float = 6.0,
) -> list[Spike]:
    """Find moments where each ROI's rolling-baseline-adjusted signal exceeds z_threshold.

    rolling_window_s: the baseline window. Short window = more sensitive to
    fast changes; long window = captures broader rises. 6s is a reasonable
    default for ad creatives where scenes change every 1–3s.
    """
    spikes: list[Spike] = []
    t = rts.t_seconds
    if len(t) < 2:
        return spikes

    dt = float(np.median(np.diff(t)))
    win = max(3, int(round(rolling_window_s / max(dt, 1e-3))))

    for ci, cluster in enumerate(rts.cluster_names):
        sig = rts.activation[:, ci]

        # Rolling mean & std for baseline (causal: uses only past samples)
        rolling_mu, rolling_sd = _rolling_stats(sig, win)
        local_z = (sig - rolling_mu) / (rolling_sd + 1e-6)

        # Contiguous runs where local_z > threshold
        hot = local_z > z_threshold
        for start_idx, end_idx in _runs(hot, min_length=min_samples):
            peak_idx = start_idx + int(np.argmax(local_z[start_idx:end_idx + 1]))
            spikes.append(Spike(
                cluster=cluster,
                t_start=float(t[start_idx]),
                t_end=float(t[end_idx]),
                t_peak=float(t[peak_idx]),
                peak_z=float(local_z[peak_idx]),
            ))

    # Sort by peak time
    spikes.sort(key=lambda s: (s.t_peak, -s.peak_z))
    return spikes


def _rolling_stats(x: np.ndarray, win: int) -> tuple[np.ndarray, np.ndarray]:
    """Causal rolling mean/std over the last `win` samples, with edge handling."""
    n = len(x)
    mu = np.zeros(n, dtype=np.float32)
    sd = np.zeros(n, dtype=np.float32)
    for i in range(n):
        lo = max(0, i - win + 1)
        window = x[lo:i + 1]
        mu[i] = window.mean()
        sd[i] = window.std()
    return mu, sd


def _runs(bool_arr: np.ndarray, min_length: int = 1) -> list[tuple[int, int]]:
    """Return (start, end) inclusive indices of runs of True in bool_arr."""
    runs = []
    i = 0
    n = len(bool_arr)
    while i < n:
        if bool_arr[i]:
            start = i
            while i < n and bool_arr[i]:
                i += 1
            end = i - 1
            if end - start + 1 >= min_length:
                runs.append((start, end))
        else:
            i += 1
    return runs


def extract_frame(video_path: str | Path, t_seconds: float, out_path: str | Path) -> Path:
    """Pull a JPEG frame from `video_path` at time `t_seconds` via ffmpeg.

    Uses the `-ss` seek before `-i` for fast (keyframe-snap) seeking, which
    is good enough for a thumbnail.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{max(0.0, t_seconds):.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-vf", "scale=320:-2",
        "-q:v", "3",
        str(out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None  # caller handles missing thumbnail
    return out if out.exists() else None


def attach_frames(spikes: list[Spike], video_path: str | Path, out_dir: str | Path) -> list[Spike]:
    """For each spike, extract a thumbnail at the peak moment. Mutates in-place."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, sp in enumerate(spikes):
        thumb = out_dir / f"spike_{i:03d}_{sp.cluster.replace(' ', '_').replace('&', 'and')}.jpg"
        result = extract_frame(video_path, sp.t_peak, thumb)
        if result:
            sp.frame_path = str(result)
    return spikes
