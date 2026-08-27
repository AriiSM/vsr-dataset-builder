/**
 * "Method & confidence" rows for a segment — the v3 provenance metadata:
 * which method produced each score (real model vs fallback), mouth-ROI
 * reliability, sentence-boundary type, extra confidence quantiles.
 *
 * Shared by the Explorer detail modal and the Review info panel. Only the
 * fields actually present on the segment are returned, so pre-v3 data
 * simply yields an empty list. Rows with `warn: true` flag values that
 * deserve attention (fallback methods, forced boundaries, high fail rates).
 */

/** True for empty-ish values as returned by the detail endpoint (strings). */
function isMissing(value) {
  return value == null || String(value).trim() === '' || String(value) === 'nan';
}

/** Formats a 0–1 ratio as a percentage string, or returns the raw text. */
function asPercent(value) {
  const n = parseFloat(value);
  return Number.isFinite(n) ? `${(n * 100).toFixed(1)}%` : String(value);
}

export function methodConfidenceRows(segment) {
  const rows = [];
  const add = (label, rawValue, { warn = false, format } = {}) => {
    if (isMissing(rawValue)) return;
    rows.push({
      label,
      value: format ? format(rawValue) : String(rawValue),
      warn,
    });
  };

  const isFallback = (v) => /fallback|none|missing/i.test(String(v));

  add('ASD method', segment.asd_method, { warn: isFallback(segment.asd_method) });
  add('SyncNet method', segment.syncnet_method, { warn: isFallback(segment.syncnet_method) });
  add('Mouth ROI', segment.mouth_roi_method, { warn: isFallback(segment.mouth_roi_method) });
  add('Mouth landmark fail rate', segment.mouth_landmark_fail_rate, {
    warn: parseFloat(segment.mouth_landmark_fail_rate) > 0.1,
    format: asPercent,
  });
  add('Face visibility', segment.face_visibility_ratio, { format: asPercent });
  add('Whisper conf (min)', segment.whisper_conf_min);
  add('Whisper conf (p25)', segment.whisper_conf_p25);
  add('Head pose (avg)', segment.head_pose_avg);
  add('Boundary type', segment.boundary_type, {
    warn: /forced/i.test(String(segment.boundary_type)),
  });
  add('Refiner WER (medium vs large-v3)', segment.wer_medium_vs_large);

  return rows;
}
