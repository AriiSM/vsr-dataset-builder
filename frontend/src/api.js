/**
 * API client — all HTTP calls to the Flask backend go through here.
 *
 * Benefit: components contain no URLs or direct `fetch` calls, so if an
 * endpoint changes, it only needs to be updated in one place.
 */

/** GET that returns JSON; throws an error if the response is not ok. */
async function getJson(url) {
  const response = await fetch(url);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || `Request failed: ${response.status}`);
  }
  return data;
}

/** POST with a JSON body; returns { ok, data } so the caller decides what to do. */
async function postJson(url, body) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  return { ok: response.ok, data };
}

export const api = {
  // ── Backend health (header indicator) ─────────────────────────────────
  /** Liveness + catalog info; throws when the backend is unreachable. */
  getHealth: () => getJson('/api/health'),

  // ── Pipeline (Process tab) ─────────────────────────────────────────────
  /** Starts the pipeline in a given mode (batch / single / resume). */
  startPipeline: (body) => postJson('/api/start', body),
  /** Starts a bulk import of YouTube URLs. */
  startBulkImport: (body) => postJson('/api/bulk_import', body),
  /** Stops the running pipeline. */
  stopPipeline: () => postJson('/api/stop', {}),
  /** Current state: running or not, recent log, start time. */
  getStatus: () => getJson('/api/status'),

  // ── Statistics (Stats tab) ────────────────────────────────────────────
  getStats: () => getJson('/api/stats'),
  getDistributions: () => getJson('/api/stats/distributions'),
  getVideos: () => getJson('/api/videos'),
  getVocabulary: () => getJson('/api/vocabulary'),
  getSpeakers: () => getJson('/api/speakers'),
  /** Saves a speaker's metadata (name, gender, age, accent). */
  updateSpeaker: (speakerId, payload) =>
    postJson(`/api/speaker/${encodeURIComponent(speakerId)}`, payload),

  // ── Segments (Explorer + Review) ──────────────────────────────────────
  getSegments: () => getJson('/api/segments'),
  getReviewStatus: () => getJson('/api/review_status'),
  getSegmentDetail: (segmentId) =>
    getJson(`/api/segment/${encodeURIComponent(segmentId)}`),
  /** Curation action: approve / reject / save / save_words / revert. */
  reviewSegment: (segmentId, payload) =>
    postJson(`/api/segment/${encodeURIComponent(segmentId)}/review`, payload),
  /** Trims the clip to the [start, end] interval (seconds). */
  trimSegment: (segmentId, start, end) =>
    postJson(`/api/segment/${encodeURIComponent(segmentId)}/trim`, { start, end }),
  /** Moves the segment to another speaker. */
  setSegmentSpeaker: (segmentId, speakerId) =>
    postJson(`/api/segment/${encodeURIComponent(segmentId)}/speaker`, { speaker_id: speakerId }),

  // ── Media URLs (not fetch calls, they only build the address) ─────────
  /** A segment's video; crop = 'face' or 'mouth'. */
  mediaUrl: (videoId, segmentId, crop = 'face') =>
    `/api/media/${encodeURIComponent(videoId)}/${encodeURIComponent(segmentId)}?type=${crop}`,
  /** A speaker's thumbnail; seg = sample index (0–3). */
  speakerThumbnailUrl: (speakerId, seg = 0, cacheBust = '') =>
    `/api/speaker/${encodeURIComponent(speakerId)}/thumbnail?seg=${seg}${cacheBust ? `&t=${cacheBust}` : ''}`,
};
