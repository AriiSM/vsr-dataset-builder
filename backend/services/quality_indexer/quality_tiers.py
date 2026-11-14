"""
Quality Tiers

Derives a per-segment quality tier (A/B/C) from the metadata the pipeline
already collects. Tiers enable curriculum training and filtering WITHOUT
throwing data away (hard drops stay minimal and happen upstream).

  A — trustworthy: clean transcript, stable face, fresh lip landmarks,
      duration in the preferred range, no degraded scoring paths.
  B — usable: one or more soft issues (weak word, flagged landmarks,
      fallback scoring, forced/odd boundary, out-of-range duration).
  C — dubious: fails even the B thresholds. Kept, but excluded from
      first-pass training.

Pure logic (stdlib only) — used by the pipeline at export time AND by
backend/tools/rebuild_segments_index.py to recompute tiers offline after
threshold changes, without reprocessing any video.
"""

import math


# Scoring paths that mean "this metric is not the real model's opinion".
_DEGRADED_ASD_METHODS = {"fallback_motion"}
_DEGRADED_SYNC_METHODS = {"fallback_correlation", "error"}
_DEGRADED_MOUTH_METHODS = {"retinaface_fallback"}


def _passes(metrics: dict, thresholds: dict) -> bool:
    """Check one tier's numeric thresholds. Missing metrics fail closed."""
    checks = [
        ("whisper_conf", thresholds.get("whisper_conf"), False),
        ("whisper_conf_min", thresholds.get("whisper_conf_min"), False),
        ("face_visibility_ratio", thresholds.get("face_visibility"), False),
        ("mouth_landmark_fail_rate", thresholds.get("mouth_fail_rate"), True),
    ]
    for key, limit, lower_is_better in checks:
        if limit is None:
            continue
        value = metrics.get(key)
        if value is None:
            return False
        try:
            value = float(value)
        except (TypeError, ValueError):
            return False
        if math.isnan(value):
            # No signal at all (e.g. Whisper returned no word scores) —
            # NaN comparisons are always False, so fail explicitly.
            return False
        if lower_is_better:
            if value > float(limit):
                return False
        elif value < float(limit):
            return False
    return True


def compute_quality_tier(metrics: dict, tier_config: dict) -> str:
    """Assign a quality tier to one segment.

    Args:
        metrics: segment metadata — recognised keys:
            whisper_conf, whisper_conf_min, face_visibility_ratio,
            mouth_landmark_fail_rate, asd_method, syncnet_method,
            mouth_roi_method, boundary_start_type, boundary_end_type,
            duration.
        tier_config: the config.yaml `quality_tiers:` block —
            {"tier_a": {...}, "tier_b": {...},
             "preferred_duration": [min, max]}.

    Returns:
        "A" | "B" | "C"
    """
    tier_a = tier_config.get("tier_a", {})
    tier_b = tier_config.get("tier_b", {})

    # Conditions that cap a segment at B no matter how good the numbers are:
    # the numbers themselves come from a degraded path, or the clip boundary
    # was not a natural sentence/pause edge.
    capped_at_b = (
        metrics.get("asd_method") in _DEGRADED_ASD_METHODS
        or metrics.get("syncnet_method") in _DEGRADED_SYNC_METHODS
        or metrics.get("mouth_roi_method") in _DEGRADED_MOUTH_METHODS
        or metrics.get("boundary_start_type") == "forced"
        or metrics.get("boundary_end_type") == "forced"
    )

    preferred = tier_config.get("preferred_duration")
    if preferred and metrics.get("duration") is not None:
        duration = float(metrics["duration"])
        if not (float(preferred[0]) <= duration <= float(preferred[1])):
            capped_at_b = True

    if not capped_at_b and _passes(metrics, tier_a):
        return "A"
    if _passes(metrics, tier_b):
        return "B"
    return "C"
