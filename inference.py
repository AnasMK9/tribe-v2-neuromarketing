"""
TRIBE v2 model wrapper.

Handles lazy loading, device selection (CUDA → MPS → CPU), and the
single inference call used by the Gradio app.

Public API:
    get_model() -> loaded model (cached)
    predict(video_path) -> (predictions, t_seconds)
        predictions: np.ndarray (n_timesteps, n_vertices ~20484)
        t_seconds:   np.ndarray (n_timesteps,)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

# The tribev2 Python package is loaded lazily inside get_model() so that the
# Gradio UI can start fast and so that missing model weights don't block
# non-inference pages (e.g. Methodology tab).

_MODEL = None
_DEVICE: Optional[str] = None


def _pick_device() -> str:
    """CUDA > MPS > CPU. Respects TRIBE_DEVICE env var for override."""
    override = os.environ.get("TRIBE_DEVICE")
    if override:
        return override

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_model():
    """Lazy-load TRIBE v2. Cached — subsequent calls are free."""
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return _MODEL

    from tribev2 import TribeModel

    _DEVICE = _pick_device()
    print(f"[tribe] loading facebook/tribev2 on device={_DEVICE}…")
    t0 = time.time()
    try:
        _MODEL = TribeModel.from_pretrained("facebook/tribev2", device=_DEVICE)
    except Exception as exc:
        # MPS support is not officially documented. Fall back to CPU with a
        # loud warning rather than crashing the app.
        if _DEVICE == "mps":
            print(f"[tribe] MPS load failed ({exc!r}) — falling back to CPU")
            _DEVICE = "cpu"
            _MODEL = TribeModel.from_pretrained("facebook/tribev2", device="cpu")
        else:
            raise
    print(f"[tribe] loaded in {time.time() - t0:.1f}s on {_DEVICE}")
    return _MODEL


def current_device() -> str:
    """Return the device the model is running on (after get_model())."""
    return _DEVICE or _pick_device()


def predict(video_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Run TRIBE v2 on a video and return (predictions, t_seconds).

    predictions: (n_timesteps, n_vertices) — per-timestep vertex activations
                 on the fsaverage5 cortical surface.
    t_seconds:   (n_timesteps,) — wall-clock timestamps of each sample, aligned
                 to the video's t=0.
    """
    model = get_model()

    events_df = model.get_events_dataframe(video_path=str(video_path))
    preds, segments = model.predict(events=events_df)

    # Convert tensor → numpy if needed
    if hasattr(preds, "detach"):
        preds = preds.detach().cpu().numpy()
    preds = np.asarray(preds, dtype=np.float32)

    # Derive t_seconds. TRIBE v2 samples at the fMRI TR (~1.49s) by default.
    # If `segments` carries explicit timings we prefer those; otherwise fall
    # back to uniform spacing.
    t_seconds = _extract_timestamps(segments, n=preds.shape[0])

    return preds, t_seconds


def _extract_timestamps(segments, n: int) -> np.ndarray:
    """Try hard to get per-sample timestamps. Fall back to uniform TR=1.49s."""
    try:
        if hasattr(segments, "columns"):
            for col in ("t_start", "start", "onset", "time", "t"):
                if col in segments.columns:
                    arr = segments[col].to_numpy()
                    if arr.shape[0] == n:
                        return arr.astype(np.float32)
        if hasattr(segments, "__len__") and len(segments) == n:
            arr = np.asarray(segments, dtype=np.float32)
            if arr.ndim == 1:
                return arr
    except Exception:
        pass

    # Default: TRIBE v2 targets TR ≈ 1.49s (HCP-style fMRI). Close enough for
    # the timeline display; real timestamps will override this if available.
    TR = 1.49
    return np.arange(n, dtype=np.float32) * TR
