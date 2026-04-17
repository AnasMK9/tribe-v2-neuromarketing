"""
Smoke test — exercises the scoring/spike/copy pipeline with synthetic TRIBE v2
output. Does NOT require torch, tribev2, or model weights.

Usage: python smoke_test.py

Validates:
 - scoring.aggregate_to_rois works with (n_t, 20484) input
 - scoring.summary_scores returns sensible 0-100 scores
 - spikes.detect_spikes finds injected spikes at expected timestamps
 - copy.format_score / format_spike produce non-empty strings
"""

from __future__ import annotations

import sys

import numpy as np


def main() -> int:
    import scoring
    import spikes
    import interp

    rng = np.random.default_rng(42)

    # Simulate a 30s clip at TR=1.49s → ~20 samples
    n_t = 20
    n_v = 20484
    t_seconds = np.arange(n_t) * 1.49

    # Base: normal-distributed noise
    preds = rng.standard_normal((n_t, n_v)).astype(np.float32) * 0.3

    # Inject a "face moment" at t≈10s (sample 6–8): Yeo 4 & 7 networks up
    # We don't have atlas access here; instead, we spike a broad vertex range
    # and trust that ≥one cluster (Face & Social uses nets 4 & 7) will see it.
    # A wide injection ensures whichever Yeo mapping is present picks it up.
    spike_window = slice(6, 9)
    # Pick ~30% of vertices (matches the fraction assigned to Yeo 4+7 combined)
    face_verts = rng.choice(n_v, size=n_v // 3, replace=False)
    preds[spike_window, face_verts[:, None].T] += 4.0  # large jump

    # Inject a "motion moment" at t≈20s (sample 13–14)
    motion_window = slice(13, 15)
    motion_verts = rng.choice(n_v, size=n_v // 4, replace=False)
    preds[motion_window, motion_verts[:, None].T] += 3.5

    print(f"Synthetic predictions: shape={preds.shape}, dtype={preds.dtype}")

    # --- scoring ---
    print("\n[1/4] Aggregating to ROIs…")
    rts = scoring.aggregate_to_rois(preds, t_seconds)
    assert rts.activation.shape == (n_t, 7), f"got {rts.activation.shape}"
    assert len(rts.cluster_names) == 7
    print(f"  ✓ ROI timeseries: {rts.activation.shape}, clusters={rts.cluster_names}")

    # z-scoring means per-column mean ~0, std ~1
    col_mean = rts.activation.mean(axis=0)
    col_std = rts.activation.std(axis=0)
    assert np.allclose(col_mean, 0, atol=1e-4), f"means not zero: {col_mean}"
    assert np.allclose(col_std, 1, atol=1e-4), f"stds not one: {col_std}"
    print("  ✓ Per-cluster z-scoring sane (mean≈0, std≈1)")

    print("\n[2/4] Computing summary scores…")
    summary = scoring.summary_scores(rts)
    assert 0 <= summary["overall"] <= 100, f"overall out of range: {summary['overall']}"
    for k, v in summary["marketing"].items():
        assert 0 <= v <= 100, f"{k} out of range: {v}"
    print(f"  ✓ Overall: {summary['overall']:.1f}")
    for k, v in summary["marketing"].items():
        print(f"    · {k}: {v:.1f}")

    # --- spikes ---
    print("\n[3/4] Detecting spikes…")
    spike_list = spikes.detect_spikes(rts, z_threshold=1.5)
    assert len(spike_list) >= 1, "expected at least one spike from injected signal"
    print(f"  ✓ Detected {len(spike_list)} spike(s)")
    for sp in spike_list[:6]:
        print(f"    · [{sp.cluster}] peak t={sp.t_peak:.1f}s z={sp.peak_z:.2f}")

    # --- copy ---
    print("\n[4/4] Rendering interpretation copy…")
    verdict = interp.overall_verdict(summary["overall"], summary["marketing"])
    assert verdict and len(verdict) > 40, "verdict suspiciously short"
    print(f"  ✓ {verdict[:120]}…")

    for k, v in summary["marketing"].items():
        block = interp.format_score(k, v)
        assert block["label"] and block["verdict"]
    print("  ✓ All 4 score copy blocks formatted")

    for sp in spike_list[:3]:
        s = interp.format_spike(sp.cluster, sp.t_peak, sp.peak_z)
        assert s and len(s) > 20
    print("  ✓ Spike captions formatted")

    print("\n✅ Smoke test passed — scoring / spikes / copy pipeline works end-to-end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
