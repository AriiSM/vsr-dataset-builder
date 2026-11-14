/**
 * Formatting functions and quality thresholds, used throughout the app.
 * All are pure functions (no side effects) — easy to test.
 */

/** Formats seconds as "2h 5m 30s" / "5m 30s" / "30s". */
export function formatDuration(totalSeconds) {
  const secs = Math.max(0, Math.round(totalSeconds || 0));
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = secs % 60;
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

/** Short variant for the KPI cards: "3h 42m" / "42m 10s" / "10s". */
export function formatDurationShort(totalSeconds) {
  const secs = Math.max(0, Math.round(totalSeconds || 0));
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = secs % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

/** Formats a numeric value with `digits` decimal places, or "—" if missing. */
export function formatMetric(value, digits) {
  const n = parseFloat(value);
  return Number.isFinite(n) ? n.toFixed(digits) : '—';
}

// ── Quality thresholds (same as in the Explorer filters) ────────────────
// ASD (TalkNet) is a logit-style score, in practice roughly [0, 18].
// SyncNet and Whisper confidence are normalized to [0, 1].
export const QUALITY_THRESHOLDS = {
  HIGH_ASD: 5,
  MID_ASD: 2,
  HIGH_SYNC: 0.55,
  LOW_SYNC: 0.40,
  HIGH_WHISPER: 0.75,
  LOW_WHISPER: 0.55,
};

/** CSS color class for an ASD score (green / orange / red). */
export function asdColorClass(value) {
  const v = parseFloat(value);
  if (!Number.isFinite(v)) return 'dim';
  if (v >= QUALITY_THRESHOLDS.HIGH_ASD) return 'green';
  if (v >= QUALITY_THRESHOLDS.MID_ASD) return 'orange';
  return 'red';
}

/** CSS color class for SyncNet confidence. */
export function syncColorClass(value) {
  const v = parseFloat(value);
  if (!Number.isFinite(v)) return 'dim';
  if (v >= QUALITY_THRESHOLDS.HIGH_SYNC) return 'green';
  if (v >= QUALITY_THRESHOLDS.LOW_SYNC) return 'orange';
  return 'red';
}

/** CSS color class for Whisper confidence. */
export function whisperColorClass(value) {
  const v = parseFloat(value);
  if (!Number.isFinite(v)) return 'dim';
  if (v >= QUALITY_THRESHOLDS.HIGH_WHISPER) return 'green';
  if (v >= QUALITY_THRESHOLDS.LOW_WHISPER) return 'orange';
  return 'red';
}

/**
 * The list of pages shown in pagination: [1, '...', 4, 5, 6, '...', 20].
 * For ≤ 7 pages, all of them are shown.
 */
export function paginationRange(currentPage, totalPages) {
  if (totalPages <= 7) {
    return Array.from({ length: totalPages }, (_, i) => i + 1);
  }
  const pages = [1];
  if (currentPage > 3) pages.push('...');
  for (
    let p = Math.max(2, currentPage - 1);
    p <= Math.min(totalPages - 1, currentPage + 1);
    p++
  ) {
    pages.push(p);
  }
  if (currentPage < totalPages - 2) pages.push('...');
  pages.push(totalPages);
  return pages;
}
