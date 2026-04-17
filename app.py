"""
Gradio app — TRIBE v2 Neuromarketing Analyzer.

Three tabs:
  1. Analyze — upload one video, get scores + brain viz + spike timeline
  2. Compare — upload 2–5 videos, get leaderboard + radar overlay
  3. Methodology — what TRIBE v2 is, how scoring works, limitations

Launch:
  python app.py               # default: 0.0.0.0:7860
  GRADIO_SHARE=1 python app.py  # public ngrok-style URL (for remote testing)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

import inference
import scoring
import spikes
import viz
import interp

load_dotenv()

CACHE_DIR = Path(tempfile.gettempdir()) / "tribe_v2_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MAX_CLIP_SECONDS = 60

# ─────────────────────────────────────────────────────────────────────────────
# Core analysis pipeline
# ─────────────────────────────────────────────────────────────────────────────

def analyze_video(video_path: str, progress=gr.Progress()):
    """Run one video through the full pipeline."""
    if not video_path:
        raise gr.Error("Upload a video first.")

    progress(0.05, desc="Checking clip…")
    _guard_clip_length(video_path)

    progress(0.10, desc=f"Loading TRIBE v2 ({inference.current_device()})…")
    inference.get_model()

    progress(0.25, desc="Running TRIBE v2 inference (this is the slow part)…")
    predictions, t_seconds = inference.predict(video_path)

    progress(0.75, desc="Aggregating ROI timeseries…")
    rts = scoring.aggregate_to_rois(predictions, t_seconds)
    summary = scoring.summary_scores(rts)

    progress(0.85, desc="Detecting spikes and extracting frames…")
    spike_list = spikes.detect_spikes(rts, z_threshold=2.0)
    session_dir = CACHE_DIR / f"run_{Path(video_path).stem}"
    session_dir.mkdir(parents=True, exist_ok=True)
    spike_list = spikes.attach_frames(spike_list, video_path, session_dir / "frames")

    progress(0.93, desc="Rendering brain surface…")
    surf_png = viz.brain_surface_png(predictions, session_dir / "brain.png")

    progress(0.98, desc="Writing interpretation…")
    verdict = interp.overall_verdict(summary["overall"], summary["marketing"])
    score_blocks = [
        interp.format_score(k, v) for k, v in summary["marketing"].items()
    ]

    timeline_fig = viz.timeline_figure(rts, spike_list)
    radar_fig = viz.radar_figure(summary["marketing"])
    roi_fig = viz.roi_bar_figure(summary["roi_scores"])

    spike_gallery = _build_spike_gallery(spike_list)
    score_md = _render_score_md(summary, verdict, score_blocks)

    return (
        score_md,           # Markdown summary
        radar_fig,          # Radar
        roi_fig,            # ROI bar
        timeline_fig,       # Timeline
        str(surf_png),      # Brain image
        spike_gallery,      # Gallery of spike frames with captions
    )


def _guard_clip_length(video_path: str) -> None:
    """Reject clips longer than MAX_CLIP_SECONDS."""
    import subprocess
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, check=True,
        )
        duration = float(out.stdout.strip())
    except Exception:
        return  # if probe fails, let inference decide
    if duration > MAX_CLIP_SECONDS:
        raise gr.Error(
            f"Clip is {duration:.1f}s. Max is {MAX_CLIP_SECONDS}s — trim it first. "
            f"TRIBE v2 on a T4 takes ~2–5 min per 60s clip; longer clips exceed "
            f"the Space timeout."
        )


def _render_score_md(summary, verdict, score_blocks):
    lines = [f"## Overall: {summary['overall']:.0f}/100", "", verdict, ""]
    lines.append("| Dimension | Score | Tier | Verdict |")
    lines.append("|---|---|---|---|")
    for b in score_blocks:
        tier_emoji = {"low": "🔴 LOW", "med": "🟡 MED", "high": "🟢 HIGH"}[b["mood"]]
        verdict_short = b["verdict"][:120] + ("…" if len(b["verdict"]) > 120 else "")
        lines.append(f"| **{b['label']}** | {b['value']:.0f} | {tier_emoji} | {verdict_short} |")
    lines.append("")
    for b in score_blocks:
        if b["fix"]:
            lines.append(f"**Fix for {b['label']}:** {b['fix']}")
            lines.append("")
    return "\n".join(lines)


def _build_spike_gallery(spike_list):
    """Return a list of (image_path, caption) for gr.Gallery."""
    items = []
    for sp in spike_list[:24]:  # cap for display
        cap = interp.format_spike(sp.cluster, sp.t_peak, sp.peak_z)
        if sp.frame_path and Path(sp.frame_path).exists():
            items.append((sp.frame_path, cap))
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Compare tab
# ─────────────────────────────────────────────────────────────────────────────

def compare_videos(video_paths, progress=gr.Progress()):
    if not video_paths:
        raise gr.Error("Upload at least 2 videos.")
    if len(video_paths) > 5:
        raise gr.Error("Max 5 videos.")
    if len(video_paths) < 2:
        raise gr.Error("Need at least 2 to compare.")

    rows = []
    import plotly.graph_objects as go
    overlay = go.Figure()
    labels = ["Attention", "Engagement", "Comprehension", "Memorability"]
    palette = ["#4ECDC4", "#F7DC6F", "#EC7063", "#BB8FCE", "#58D68D"]

    for i, vp in enumerate(video_paths):
        vp_str = vp if isinstance(vp, str) else vp.name
        progress((i + 0.1) / len(video_paths), desc=f"Clip {i+1}/{len(video_paths)}: inference…")
        _guard_clip_length(vp_str)
        predictions, t_seconds = inference.predict(vp_str)
        rts = scoring.aggregate_to_rois(predictions, t_seconds)
        summ = scoring.summary_scores(rts)
        name = Path(vp_str).stem
        rows.append({"name": name, "overall": summ["overall"], **summ["marketing"], **summ["roi_scores"]})

        keys = ["attention_score", "engagement_score", "comprehension_score", "memorability_score"]
        values = [summ["marketing"][k] for k in keys]
        overlay.add_trace(go.Scatterpolar(
            r=values + values[:1],
            theta=labels + labels[:1],
            fill="toself",
            name=name,
            opacity=0.5,
            line=dict(color=palette[i % len(palette)], width=2),
        ))

    overlay.update_layout(
        polar=dict(
            bgcolor=viz.PANEL_BG,
            radialaxis=dict(range=[0, 100], color=viz.MUTED, gridcolor=viz.GRID),
            angularaxis=dict(color=viz.TEXT, gridcolor=viz.GRID),
        ),
        paper_bgcolor=viz.DARK_BG,
        font=dict(color=viz.TEXT, family="monospace"),
        showlegend=True,
        legend=dict(bgcolor=viz.PANEL_BG, bordercolor=viz.GRID),
        margin=dict(l=40, r=40, t=40, b=40),
        height=450,
    )

    leaderboard = viz.leaderboard_figure(rows)

    import pandas as pd
    df = pd.DataFrame(rows).set_index("name")
    # Sort columns for readability
    display_cols = (
        ["overall", "attention_score", "engagement_score", "comprehension_score", "memorability_score"]
        + list(scoring.ROI_CLUSTERS.keys())
    )
    df = df[[c for c in display_cols if c in df.columns]].round(1)
    df = df.sort_values("overall", ascending=False)

    # Which ROI most separates them?
    roi_cols = [c for c in scoring.ROI_CLUSTERS.keys() if c in df.columns]
    top_roi = df[roi_cols].var().sort_values(ascending=False).index[0] if roi_cols else ""
    explainer = (
        f"**Biggest differentiator:** {top_roi} — this is the ROI cluster that varies most "
        f"across your creatives. If one is winning overall, it's likely winning here."
    )

    return leaderboard, overlay, df.reset_index(), explainer


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

METHODOLOGY_MD = """
## What TRIBE v2 is

TRIBE v2 is a trimodal (video + audio + language) brain-encoder released by Meta AI Research. It takes a video and predicts the fMRI response of an average human brain on the **fsaverage5** cortical surface (~20,484 vertices).

We don't use the raw vertex output. We aggregate vertices into **7 neuromarketing ROI clusters**, then derive **4 composite scores**: Attention, Engagement, Comprehension, Memorability.

## What this does well

- Flags **peak moments**: when in the video each region fired. The spike timeline shows you exactly which frame lit up Face & Social or Emotion & Reward.
- Within-creative comparison: which creative activates *more* in the regions that matter for your campaign goal.
- Saves you from running real fMRI studies on every iteration.

## What this doesn't do

- **Encoding ≠ behavior.** High Engagement doesn't guarantee conversion. Pair with behavioral uplift studies.
- **Population-average brain.** TRIBE v2 was trained on ~720 general-population subjects. Your target demographic (age, culture, brand familiarity) may diverge.
- **No subcortex.** fsaverage5 is cortex-only. True amygdala / NAcc / hippocampus signal is not directly captured — Emotion & Reward and Memory Encoding clusters are cortical proxies.
- **Non-Latin scripts** have weaker zero-shot performance. Interpret comprehension scores for Arabic / CJK copy with caution.

## License

TRIBE v2 is **CC-BY-NC-4.0** — non-commercial research only. Do not build a paid product on top of this Space.

## References

- Meta AI Research, [TRIBE: A Trimodal Brain Encoder…](https://ai.meta.com/research/publications/tribe-a-trimodal-brain-encoder-for-any-fmri-subject-and-task/)
- Model card: https://huggingface.co/facebook/tribev2
- Yeo 7-network parcellation: Yeo et al., 2011, *J Neurophysiol*.
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="TRIBE v2 Neuromarketing") as demo:
        gr.Markdown(
            "# TRIBE v2 — Neuromarketing Creative Analyzer\n"
            "Upload a 15–60s video ad. Get predicted neural response + spike timeline + "
            "plain-English interpretation.\n\n"
            f"Running on: `{inference.current_device()}` · Max clip: **{MAX_CLIP_SECONDS}s**"
        )

        with gr.Tabs():
            with gr.Tab("Analyze"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_in = gr.Video(label="Upload video", sources=["upload"])
                        run_btn = gr.Button("Analyze", variant="primary")
                    with gr.Column(scale=2):
                        scores_md = gr.Markdown()

                with gr.Row():
                    radar_out = gr.Plot(label="Score radar")
                    roi_out = gr.Plot(label="ROI activation")

                timeline_out = gr.Plot(label="Spike timeline")
                brain_out = gr.Image(label="Cortical activation (fsaverage5)", type="filepath")
                spike_gallery_out = gr.Gallery(
                    label="Spike moments — hover for interpretation",
                    columns=4, height="auto", object_fit="cover",
                )

                run_btn.click(
                    analyze_video,
                    inputs=video_in,
                    outputs=[scores_md, radar_out, roi_out, timeline_out, brain_out, spike_gallery_out],
                )

            with gr.Tab("Compare"):
                gr.Markdown("Upload **2–5 videos** to compare side-by-side.")
                videos_in = gr.File(
                    label="Upload videos",
                    file_count="multiple",
                    file_types=["video"],
                )
                compare_btn = gr.Button("Compare", variant="primary")

                leader_out = gr.Plot(label="Leaderboard")
                radar_overlay = gr.Plot(label="Score radar overlay")
                table_out = gr.Dataframe(label="All scores", interactive=False, wrap=True)
                explainer_out = gr.Markdown()

                compare_btn.click(
                    compare_videos,
                    inputs=videos_in,
                    outputs=[leader_out, radar_overlay, table_out, explainer_out],
                )

            with gr.Tab("Methodology"):
                gr.Markdown(METHODOLOGY_MD)

    return demo


if __name__ == "__main__":
    share = os.environ.get("GRADIO_SHARE", "0") == "1"
    demo = build_ui()
    demo.queue(max_size=8).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=share,
        theme=gr.themes.Base(primary_hue="teal", neutral_hue="slate"),
    )
