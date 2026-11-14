"""
Crop trajectories — WHERE to cut, frame by frame.

A crop trajectory is the sequence of crop windows across a segment's frames.
It has to be COMPUTED because: (a) detections exist only every Nth frame →
interpolate between them; (b) raw positions jitter → smooth them (a shaking
crop is motion noise the model would waste capacity learning); (c) the
window size stays CONSTANT per segment (otherwise the mouth "pulses" with
camera zoom).

Three builders, all moved UNCHANGED from the Phase 2-3 implementation:
    build_face_trajectory      — Gaussian-smoothed centers, fixed size
                                 (largest face + margin) → stable head crop
    build_mouth_trajectory_v2  — dense MediaPipe lip landmarks + One-Euro
                                 smoothing, size from the MEDIAN mouth width
    build_mouth_trajectory_v1  — legacy 2-point RetinaFace fallback (used
                                 only when MediaPipe fails on most frames)
"""

import json
from typing import Dict, Tuple

import numpy as np

from services.mouth_exporter.mouth_landmarks import LipLandmarks, OneEuroFilter


def fill_nan_gaps(values: np.ndarray) -> np.ndarray:
    """Linear-interpolate NaN gaps in a 1-D array (in place, returned)."""
    nan_mask = np.isnan(values)
    if nan_mask.all():
        return values
    indices = np.arange(len(values))
    values[nan_mask] = np.interp(indices[nan_mask], indices[~nan_mask], values[~nan_mask])
    return values


def build_face_trajectory(
    track,
    start_frame: int,
    end_frame: int,
    smoothing_sigma: float,
    crop_margin: float,
    minimum_half_size_pixels: int,
):
    """
    Build a zoom-stable crop window for every frame in [start_frame, end_frame).

    Strategy (addresses camera-zoom instability):
    1. Collect raw face centers (cx, cy) from detections / interpolation.
    2. Gaussian-smooth the CENTER with a short sigma (≈0.1 s at 25 fps) so
       it follows natural head movement without over-smoothing.
    3. Compute a STABLE crop size = max detected face size in this segment
       enlarged by crop_margin.  Because the size is fixed for the whole
       segment, the crop window can never be too small for a zoomed-in face.
    4. Return a square BoundingBox centred at the smoothed center for each frame.
    """
    from scipy.ndimage import gaussian_filter1d
    from services.face_tracker.geometry import BoundingBox

    sigma = smoothing_sigma

    frames = list(range(start_frame, end_frame))
    n = len(frames)
    if n == 0:
        return {}

    cxs = np.full(n, np.nan)
    cys = np.full(n, np.nan)
    sizes = []  # collect detected face sizes to compute stable crop size

    for i, f in enumerate(frames):
        bbox = track.get_bbox_at_frame(f)
        if bbox is None:
            bbox = track.interpolate_bbox(f)
        if bbox is not None:
            cxs[i] = bbox.x + bbox.width  / 2.0
            cys[i] = bbox.y + bbox.height / 2.0
            sizes.append(max(bbox.width, bbox.height))

    if np.isnan(cxs).all():
        return {}

    # Fill gaps in center trajectory
    cxs = fill_nan_gaps(cxs)
    cys = fill_nan_gaps(cys)

    # Smooth centers
    cxs_s = gaussian_filter1d(cxs, sigma=sigma)
    cys_s = gaussian_filter1d(cys, sigma=sigma)

    # Stable crop size: largest face in segment + margin buffer
    # Using max (not median) so the window is always big enough even at peak zoom
    if sizes:
        stable_half = int(max(sizes) * (1.0 + crop_margin) / 2.0)
    else:
        stable_half = minimum_half_size_pixels

    result = {}
    for i, f in enumerate(frames):
        cx = int(round(cxs_s[i]))
        cy = int(round(cys_s[i]))
        result[f] = BoundingBox(
            x=cx - stable_half,
            y=cy - stable_half,
            width=stable_half * 2,
            height=stable_half * 2,
            confidence=1.0,
        )
    return result


def build_mouth_trajectory_v1(
    track,
    start_frame: int,
    end_frame: int,
    smoothing_sigma: float,
    width_multiplier: float,
    minimum_half_size_pixels: int,
) -> Dict[int, Tuple[int, int, int]]:
    """
    Compute a smoothed (mouth_cx, mouth_cy, half_size) for every frame.

    Uses 5-point RetinaFace landmarks [left_mouth=3, right_mouth=4].
    Falls back to lower-third of face bbox when landmarks are unavailable.

    Returns dict mapping frame_idx → (cx, cy, half_size).
    """
    from scipy.ndimage import gaussian_filter1d

    sigma = smoothing_sigma

    frames = list(range(start_frame, end_frame))
    n = len(frames)
    if n == 0:
        return {}

    mcxs = np.full(n, np.nan)
    mcys = np.full(n, np.nan)
    mwidths = []

    # Build a lookup: frame_idx → FaceDetection (for landmarks)
    det_by_frame = {d.frame_idx: d for d in track.detections}

    for i, f in enumerate(frames):
        det = det_by_frame.get(f)
        if det is None:
            # Try surrounding detections for landmark interpolation
            det = det_by_frame.get(
                min(det_by_frame.keys(), key=lambda k: abs(k - f), default=None)
            )

        if det is not None and det.landmarks is not None and len(det.landmarks) >= 5:
            lm = det.landmarks  # shape (5, 2): [le, re, nose, lm, rm]
            mcxs[i] = (lm[3][0] + lm[4][0]) / 2.0
            mcys[i] = (lm[3][1] + lm[4][1]) / 2.0
            mwidths.append(abs(lm[4][0] - lm[3][0]))
        else:
            # Fallback: approximate mouth from bbox
            bbox = track.get_bbox_at_frame(f) or track.interpolate_bbox(f)
            if bbox is not None:
                mcxs[i] = bbox.x + bbox.width / 2.0
                mcys[i] = bbox.y + bbox.height * 0.75
                mwidths.append(bbox.width * 0.5)

    if np.isnan(mcxs).all():
        return {}

    mcxs = fill_nan_gaps(mcxs)
    mcys = fill_nan_gaps(mcys)

    mcxs_s = gaussian_filter1d(mcxs, sigma=sigma)
    mcys_s = gaussian_filter1d(mcys, sigma=sigma)

    # Stable mouth crop size: widest mouth × multiplier for context,
    # floored at the configured minimum half-size.
    floor = minimum_half_size_pixels
    if mwidths:
        half = max(floor, int(max(mwidths) * width_multiplier / 2.0))
    else:
        half = floor

    result = {}
    for i, f in enumerate(frames):
        result[f] = (int(round(mcxs_s[i])), int(round(mcys_s[i])), half)
    return result


def build_mouth_trajectory_v2(
    landmarks_by_frame: Dict[int, LipLandmarks],
    start_frame: int,
    end_frame: int,
    fps: float,
    smoothing_type: str,
    one_euro_min_cutoff: float,
    one_euro_beta: float,
    gaussian_sigma: float,
    width_multiplier: float,
    minimum_half_size_pixels: int,
) -> Tuple[Dict[int, Tuple[int, int, int, float]], float, str]:
    """Build the per-frame mouth crop window from dense lip landmarks.

    Returns (trajectory, fail_rate, head_pose_avg_json) where trajectory
    maps frame → (center_x, center_y, half_size, roll_degrees).

    - Crop size is FIXED per segment: median mouth width × multiplier
      (median, not max — one open-mouth frame must not zoom the segment).
    - Center is smoothed with One-Euro (low lag) or Gaussian, per config.
    - Missing frames carry the previous landmarks (counted in fail_rate).
    """
    frames = list(range(start_frame, end_frame))
    total = len(frames)
    if total == 0 or not landmarks_by_frame:
        return {}, 1.0, ""

    detected = [landmarks_by_frame.get(f) for f in frames]
    fail_count = sum(1 for d in detected if d is None)
    fail_rate = fail_count / total

    widths = [d.mouth_width for d in detected if d is not None]
    half = max(
        minimum_half_size_pixels,
        int(float(np.median(widths)) * width_multiplier / 2.0),
    )

    # Head pose average over frames with real detections
    head_pose_avg = json.dumps([
        round(float(np.mean([d.yaw_proxy for d in detected if d])), 3),
        round(float(np.mean([d.pitch_proxy for d in detected if d])), 3),
        round(float(np.mean([d.roll_degrees for d in detected if d])), 1),
    ])

    # Carry-forward for gaps, then smooth the center trajectory
    centers_x = np.full(total, np.nan)
    centers_y = np.full(total, np.nan)
    rolls = np.zeros(total)
    for i, lips in enumerate(detected):
        if lips is not None:
            centers_x[i], centers_y[i] = lips.mouth_center
            rolls[i] = lips.roll_degrees
        elif i > 0:
            centers_x[i] = centers_x[i - 1]
            centers_y[i] = centers_y[i - 1]
            rolls[i] = rolls[i - 1]
    centers_x = fill_nan_gaps(centers_x)
    centers_y = fill_nan_gaps(centers_y)

    if smoothing_type == "one_euro":
        filter_x = OneEuroFilter(fps, one_euro_min_cutoff,
                                 one_euro_beta)
        filter_y = OneEuroFilter(fps, one_euro_min_cutoff,
                                 one_euro_beta)
        centers_x = np.array([filter_x.filter(float(v)) for v in centers_x])
        centers_y = np.array([filter_y.filter(float(v)) for v in centers_y])
    else:
        from scipy.ndimage import gaussian_filter1d
        centers_x = gaussian_filter1d(centers_x, gaussian_sigma)
        centers_y = gaussian_filter1d(centers_y, gaussian_sigma)

    trajectory = {
        f: (int(round(centers_x[i])), int(round(centers_y[i])), half,
            float(rolls[i]))
        for i, f in enumerate(frames)
    }
    return trajectory, fail_rate, head_pose_avg
