/**
 * Capability detection for the v3 pipeline metadata.
 *
 * The v3 backend adds columns to segments_index.csv and speakers_registry.csv
 * only additively, and the API drops columns that don't exist yet. So the UI
 * inspects the ACTUAL rows it received and lights up each v3 feature only
 * when its data is present — the frontend is ready today and activates
 * itself, feature by feature, as the backend starts delivering.
 *
 * Column names follow the v3 contract in plan.md; if an implemented name
 * ever differs, this file is the single place to adjust.
 */

/** True when at least one row carries the key (API omits absent columns). */
function hasColumn(rows, key) {
  return rows.length > 0 && rows.some((row) => key in row);
}

/** What the segment rows (/api/segments) support. */
export function detectSegmentCapabilities(segments) {
  return {
    /** quality_tier A/B/C computed by quality_indexer */
    hasTiers: hasColumn(segments, 'quality_tier'),
    /** MediaPipe mouth ROI metadata (method + fail rate) */
    hasMouthRoi:
      hasColumn(segments, 'mouth_roi_method') ||
      hasColumn(segments, 'mouth_landmark_fail_rate'),
    /** sentence-segmentation boundary types (punctuation|silence|word_gap|forced) */
    hasBoundaries: hasColumn(segments, 'boundary_type'),
    /** transcript_refiner output (large-v3 pass) */
    hasRefiner:
      hasColumn(segments, 'needs_review') ||
      hasColumn(segments, 'wer_medium_vs_large'),
  };
}

/** What the speaker rows (/api/speakers) support. */
export function detectSpeakerCapabilities(speakers) {
  return {
    /** numeric age estimate + spread (age_estimate / age_std) */
    hasAgeEstimate: hasColumn(speakers, 'age_estimate'),
    /** majority-vote confidence for the gender prediction */
    hasGenderConfidence: hasColumn(speakers, 'gender_confidence'),
    /** cross-video identity matching (identity_match = auto) */
    hasIdentityMatch: hasColumn(speakers, 'identity_match'),
  };
}

/** Tier display constants shared by Stats / Explorer / Review. */
export const TIER_META = {
  A: { label: 'Tier A', cls: 'green',  color: 'var(--green)' },
  B: { label: 'Tier B', cls: 'cyan',   color: 'var(--cyan)' },
  C: { label: 'Tier C', cls: 'orange', color: 'var(--orange)' },
};

/** Normalises a raw tier value ('a', ' B ', null) to 'A'|'B'|'C'|null. */
export function normalizeTier(value) {
  const tier = String(value || '').trim().toUpperCase();
  return tier === 'A' || tier === 'B' || tier === 'C' ? tier : null;
}
