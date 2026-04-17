"""
Plain-English interpretation strings for marketers.

Ported from the mock notebook's cell 13 and tuned for non-technical readers.
Every score and spike gets a "what it means" + "what to do" line written in
marketing language, not neuroscience jargon.
"""

from __future__ import annotations

SCORE_COPY = {
    "attention_score": {
        "label": "Attention",
        "what": (
            "How much your creative pulls the eye in the first second. Driven by contrast, "
            "motion, and faces — the brain regions that automatically orient toward salient stimuli."
        ),
        "high": (
            "Your creative is a scroll-stopper. Eyes lock on within 300ms. "
            "Good for feed ads, OOH, anything interruptive."
        ),
        "med": (
            "It gets noticed but doesn't dominate. Depending on placement, that may be fine."
        ),
        "low": (
            "It blends into the visual noise. In a feed, it will be skipped."
        ),
        "fix": (
            "Add motion in the first frame, boost contrast, or put a face at the focal point. "
            "Direct eye-contact especially pulls gaze."
        ),
    },
    "engagement_score": {
        "label": "Engagement",
        "what": (
            "How emotionally activated the viewer feels. Correlates with dwell time, shares, "
            "and stated purchase intent in published fMRI-behavior studies."
        ),
        "high": (
            "Viewers feel something. That's the engine behind sharing, saving, and intent-to-buy."
        ),
        "med": (
            "It's competent but emotionally flat. Viewers understand it without feeling it."
        ),
        "low": (
            "No emotional hook. The creative is being processed cognitively but not felt."
        ),
        "fix": (
            "Add a human story arc — even 3 frames of face-voice-stakes. Use music with clear "
            "emotional valence. Reframe the product around identity or belonging rather than features."
        ),
    },
    "comprehension_score": {
        "label": "Comprehension",
        "what": (
            "Whether the message actually lands. Measures how cleanly the brain decodes your copy "
            "and integrates it with the visual."
        ),
        "high": (
            "The value prop is received and understood without effort."
        ),
        "med": (
            "Viewers mostly get it, but some cognitive load remains. Simplification would help."
        ),
        "low": (
            "The message is scrambled. Copy and visual may be fighting each other."
        ),
        "fix": (
            "Cut copy to one clear claim. Make the visual literally show that claim. "
            "Reduce information density — less is more."
        ),
    },
    "memorability_score": {
        "label": "Memorability",
        "what": (
            "How likely this creative is to be remembered 24 hours later. Hippocampal encoding "
            "is boosted by surprise, narrative, and emotional salience."
        ),
        "high": (
            "Creates a memory trace — good for brand-building, not just performance."
        ),
        "med": (
            "Some stickiness. If brand recall matters, there's room to push further."
        ),
        "low": (
            "Will be forgotten shortly after exposure. Common in generic brand visuals."
        ),
        "fix": (
            "Add a narrative arc (even 3 frames). Use something unexpected — the brain encodes "
            "prediction errors. Pair with distinctive audio branding."
        ),
    },
}


SPIKE_COPY = {
    "Visual Attention": (
        "At {t:.1f}s, low-level visual salience spiked (z={z:.1f}). Something high-contrast, "
        "moving, or centrally placed pulled the eye here."
    ),
    "Object & Scene": (
        "At {t:.1f}s, the scene/object recognition network fired (z={z:.1f}). The brain is "
        "identifying what it's looking at — products, locations, context."
    ),
    "Face & Social": (
        "At {t:.1f}s, the Face & Social network spiked (z={z:.1f}). A face appeared, or a gaze / "
        "gesture triggered social-brain processing. These moments drive trust and identification."
    ),
    "Motion & Dynamics": (
        "At {t:.1f}s, motion processing surged (z={z:.1f}). A cut, movement, or energy change "
        "captured attention."
    ),
    "Language & Meaning": (
        "At {t:.1f}s, language comprehension lit up (z={z:.1f}). Copy was read/heard, or a "
        "verbal concept was integrated."
    ),
    "Emotion & Reward": (
        "At {t:.1f}s, the emotion/reward network activated (z={z:.1f}). Something felt good, "
        "important, or desirable. These are your 'feels' moments."
    ),
    "Memory Encoding": (
        "At {t:.1f}s, memory-encoding regions spiked (z={z:.1f}). This moment is likely to stick. "
        "If it's on-brand, that's gold; if off-brand, that's the thing they'll remember instead of you."
    ),
}


def format_score(score_key: str, value: float) -> dict:
    """Return a structured 'what this score means' block for a given value."""
    copy = SCORE_COPY[score_key]
    if value < 40:
        mood = "low"
    elif value < 65:
        mood = "med"
    else:
        mood = "high"
    return {
        "label": copy["label"],
        "value": value,
        "mood": mood,
        "what": copy["what"],
        "verdict": copy[mood],
        "fix": copy["fix"] if mood in ("low", "med") else None,
    }


def format_spike(cluster: str, t_peak: float, peak_z: float) -> str:
    tmpl = SPIKE_COPY.get(
        cluster,
        f"At {{t:.1f}}s, {cluster} activity spiked (z={{z:.1f}})."
    )
    return tmpl.format(t=t_peak, z=peak_z)


def overall_verdict(overall: float, marketing: dict) -> str:
    """One-paragraph executive summary."""
    sorted_scores = sorted(marketing.items(), key=lambda kv: kv[1])
    weakest_key, weakest_val = sorted_scores[0]
    strongest_key, strongest_val = sorted_scores[-1]
    weakest = SCORE_COPY[weakest_key]["label"]
    strongest = SCORE_COPY[strongest_key]["label"]

    if overall < 40:
        header = "**Rework.** Neural signal is too weak to compete in-market."
    elif overall < 65:
        header = "**Acceptable, with targeted fixes.**"
    else:
        header = "**Deploy.** Predicted to outperform category norms."

    return (
        f"{header} Overall score: {overall:.0f}/100. "
        f"Strongest dimension: {strongest} ({strongest_val:.0f}). "
        f"Weakest: {weakest} ({weakest_val:.0f}) — start your iteration here. "
        f"See the Fix field under that score for what to try."
    )
