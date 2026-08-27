import { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { api } from '../../api.js';
import { formatDuration, formatDurationShort, QUALITY_THRESHOLDS } from '../../utils/format.js';
import { useDebouncedValue } from '../../hooks/useDebouncedValue.js';
import { useLocalStorage } from '../../hooks/useLocalStorage.js';
import { detectSegmentCapabilities, normalizeTier } from '../../utils/datasetCapabilities.js';
import { SegmentCard } from './SegmentCard.jsx';
import { SegmentModal } from './SegmentModal.jsx';

const SEGMENTS_PER_PAGE = 45; // 15 columns × 3 rows

/** Filter chips above the gallery. */
const FILTER_CHIPS = [
  { value: 'all',       label: 'All' },
  { value: 'approved',  label: 'Approved', dot: 'green' },
  { value: 'rejected',  label: 'Rejected', dot: 'red' },
  { value: 'pending',   label: 'Pending',  dot: 'muted' },
  { value: 'high-asd',  label: 'High ASD' },
  { value: 'high-sync', label: 'High Sync' },
  { value: 'high-wh',   label: 'High Wh' },
  { value: 'low-sync',  label: 'Low Sync' },
  { value: 'low-wh',    label: 'Low Wh' },
  { value: 'edited',    label: 'Edited ✎' },
];

/** Sort options in the dropdown. */
const SORT_OPTIONS = [
  { value: 'text-asc',  label: 'Text A→Z' },
  { value: 'text-desc', label: 'Text Z→A' },
  { value: 'dur-desc',  label: 'Duration ↓' },
  { value: 'dur-asc',   label: 'Duration ↑' },
  { value: 'asd-desc',  label: 'ASD ↓' },
  { value: 'asd-asc',   label: 'ASD ↑' },
  { value: 'sync-desc', label: 'Sync ↓' },
  { value: 'sync-asc',  label: 'Sync ↑' },
  { value: 'wh-desc',   label: 'Whisper ↓' },
  { value: 'wh-asc',    label: 'Whisper ↑' },
];

/** Review status of a segment (approved / rejected / pending). */
function reviewStatusOf(reviewMap, segmentId) {
  const status = reviewMap[segmentId]?.status;
  return status === 'approved' || status === 'rejected' ? status : 'pending';
}

/** Chips shown only when the v3 quality_tier column is present. */
const TIER_CHIPS = [
  { value: 'tier-a', label: 'Tier A', dot: 'green' },
  { value: 'tier-b', label: 'Tier B', dot: 'cyan' },
  { value: 'tier-c', label: 'Tier C', dot: 'orange' },
];

/** Applies the selected filter (quality band / status / tier / edited). */
function applyFilter(segments, filter, reviewMap) {
  const T = QUALITY_THRESHOLDS;
  switch (filter) {
    case 'tier-a':
    case 'tier-b':
    case 'tier-c': {
      const wanted = filter.slice(-1).toUpperCase();
      return segments.filter((s) => normalizeTier(s.quality_tier) === wanted);
    }
    case 'high-asd':
      return segments.filter((s) => parseFloat(s.asd_score) >= T.HIGH_ASD);
    case 'high-sync':
      // Only segments with a real SyncNet score (skip non-backfilled zeros).
      return segments.filter((s) => {
        const v = parseFloat(s.syncnet_conf);
        return Number.isFinite(v) && v >= T.HIGH_SYNC;
      });
    case 'low-sync':
      return segments.filter((s) => {
        const v = parseFloat(s.syncnet_conf);
        return Number.isFinite(v) && v > 0 && v < T.LOW_SYNC;
      });
    case 'high-wh':
      return segments.filter((s) => parseFloat(s.whisper_conf) >= T.HIGH_WHISPER);
    case 'low-wh':
      return segments.filter((s) => {
        const v = parseFloat(s.whisper_conf);
        return Number.isFinite(v) && v < T.LOW_WHISPER;
      });
    case 'edited':
      return segments.filter(
        (s) => s.original_text && s.text &&
          String(s.original_text).trim() !== String(s.text).trim()
      );
    case 'approved':
    case 'rejected':
    case 'pending':
      return segments.filter((s) => reviewStatusOf(reviewMap, s.segment_id) === filter);
    default:
      return segments;
  }
}

/** Sorts the segments by the "key-direction" option (e.g. "dur-desc"). */
function applySort(segments, sortOption) {
  const [key, dir] = sortOption.split('-');
  const asc = dir === 'asc';
  // Missing values always go to the end, regardless of direction.
  const missing = asc ? Infinity : -Infinity;
  const num = (v) => {
    const n = parseFloat(v);
    return Number.isFinite(n) ? n : missing;
  };

  const sorted = [...segments];
  sorted.sort((a, b) => {
    if (key === 'text') {
      const ta = (a.text || '').toLowerCase();
      const tb = (b.text || '').toLowerCase();
      return asc ? ta.localeCompare(tb) : tb.localeCompare(ta);
    }
    const FIELD = { dur: 'duration', asd: 'asd_score', sync: 'syncnet_conf', wh: 'whisper_conf' }[key];
    if (!FIELD) return 0;
    return asc ? num(a[FIELD]) - num(b[FIELD]) : num(b[FIELD]) - num(a[FIELD]);
  });
  return sorted;
}

/**
 * The Explorer tab: gallery of all exported segments, with search,
 * quality filters, sorting, pagination and a detail modal.
 */
export function ExplorerTab({ isActive }) {
  const [segments, setSegments] = useState([]);
  const [reviewMap, setReviewMap] = useState({});
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState(false);

  // Filter/sort/crop choices persist across page refreshes.
  const [filter, setFilter] = useLocalStorage('vsr-explorer-filter', 'all');
  // Region filter (MD-only dataset target); values come from the data.
  const [regionFilter, setRegionFilter] = useLocalStorage('vsr-explorer-region', 'all');
  const [crop, setCrop] = useLocalStorage('vsr-explorer-crop', 'face'); // 'face' | 'mouth'
  const [sortOption, setSortOption] = useLocalStorage('vsr-explorer-sort', 'text-asc');
  const [searchInput, setSearchInput] = useState('');
  const [page, setPage] = useState(1);
  const [openSegmentId, setOpenSegmentId] = useState(null);
  const search = useDebouncedValue(searchInput, 140);

  const gridRef = useRef(null);
  // A single IntersectionObserver shared by all cards — sets the video
  // src only once the card becomes visible.
  const observerRef = useRef(null);
  if (!observerRef.current && typeof IntersectionObserver !== 'undefined') {
    observerRef.current = new IntersectionObserver(
      (entries, obs) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          const video = entry.target;
          if (!video.src && video.dataset.src) video.src = video.dataset.src;
          obs.unobserve(video);
        }
      },
      { rootMargin: '120px', threshold: 0.01 }
    );
  }

  /** Loads the segments + review statuses in parallel. */
  const loadSegments = useCallback(async () => {
    setLoading(true);
    setLoadError(false);
    try {
      const [segRes, revRes] = await Promise.all([
        api.getSegments(),
        api.getReviewStatus().catch(() => ({})),
      ]);
      setSegments(segRes.segments || []);
      setReviewMap(revRes || {});
      setPage(1);
    } catch {
      setLoadError(true);
    } finally {
      setLoading(false);
    }
  }, []);

  // Reload every time the tab is activated.
  useEffect(() => {
    if (isActive) loadSegments();
  }, [isActive, loadSegments]);

  // Filtering + search + sorting — recomputed only when something relevant changes.
  // Which v3 features the loaded data actually supports.
  const capabilities = useMemo(() => detectSegmentCapabilities(segments), [segments]);

  // The regions present in the loaded data, for the filter dropdown.
  const regions = useMemo(
    () => [...new Set(segments.map((s) => s.region).filter(Boolean))].sort(),
    [segments]
  );

  const filtered = useMemo(() => {
    let result = applyFilter(segments, filter, reviewMap);
    if (regionFilter !== 'all') {
      result = result.filter((s) => (s.region || 'UNKNOWN') === regionFilter);
    }
    const q = search.trim().toLowerCase();
    if (q) {
      result = result.filter(
        (s) =>
          (s.text || '').toLowerCase().includes(q) ||
          (s.segment_id || '').toLowerCase().includes(q) ||
          (s.video_id || '').toLowerCase().includes(q)
      );
    }
    return applySort(result, sortOption);
  }, [segments, filter, regionFilter, reviewMap, search, sortOption]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / SEGMENTS_PER_PAGE));
  const safePage = Math.min(page, totalPages);
  const pageSegments = filtered.slice(
    (safePage - 1) * SEGMENTS_PER_PAGE,
    safePage * SEGMENTS_PER_PAGE
  );

  // Statistics for the strip above the gallery (on the filtered set).
  const overview = useMemo(() => {
    const source = filtered.length ? filtered : segments;
    const durations = source.map((s) => parseFloat(s.duration || 0)).filter((d) => d > 0);
    const totalSec = durations.reduce((a, b) => a + b, 0);
    return {
      count: source.length,
      totalSec,
      avgSec: durations.length ? totalSec / durations.length : 0,
      videoCount: new Set(source.map((s) => s.video_id)).size,
      approvedCount: source.filter((s) => reviewStatusOf(reviewMap, s.segment_id) === 'approved').length,
    };
  }, [filtered, segments, reviewMap]);

  function goToPage(p) {
    if (p < 1 || p > totalPages) return;
    setPage(p);
    gridRef.current?.scrollTo(0, 0);
  }

  return (
    <main className={`tab-content ${isActive ? 'active' : ''}`} id="tab-explorer">
      <div className="gallery-layout">
        {/* Top bar: title + Face/Mouth toggle + Refresh */}
        <div className="gallery-topbar">
          <span className="cmd-panel-label">Segment gallery</span>
          <span className="cmd-panel-count">
            {segments.length > 0 &&
              `${filtered.length.toLocaleString()} of ${segments.length.toLocaleString()} segments`}
          </span>
          <div className="gallery-topbar-actions">
            <div className="crop-toggle" role="group" aria-label="Crop type">
              <button
                className={`crop-btn ${crop === 'face' ? 'active' : ''}`}
                onClick={() => setCrop('face')}
              >
                Face
              </button>
              <button
                className={`crop-btn ${crop === 'mouth' ? 'active' : ''}`}
                onClick={() => setCrop('mouth')}
              >
                Mouth
              </button>
            </div>
            <button className="btn-micro" title="Reload" onClick={loadSegments}>Refresh</button>
          </div>
        </div>

        {/* Statistics strip (persistent) */}
        {segments.length > 0 && (
          <div className="gallery-overview">
            <div className="ov-card">
              <span className="ov-value cyan">{overview.count.toLocaleString()}</span>
              <span className="ov-label">Segments</span>
            </div>
            <div className="ov-card">
              <span className="ov-value green">{formatDurationShort(overview.totalSec)}</span>
              <span className="ov-label">Duration</span>
            </div>
            <div className="ov-card">
              <span className="ov-value">{formatDuration(overview.avgSec)}</span>
              <span className="ov-label">Avg clip</span>
            </div>
            <div className="ov-card">
              <span className="ov-value">{overview.videoCount}</span>
              <span className="ov-label">Videos</span>
            </div>
            <div className="ov-card">
              <span className="ov-value green">{overview.approvedCount.toLocaleString()}</span>
              <span className="ov-label">Approved</span>
            </div>
          </div>
        )}

        {/* Filter bar: search + chips + sorting */}
        <div className="gallery-filters">
          <div className="gallery-search">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <circle cx="11" cy="11" r="7" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
            <input
              type="text"
              placeholder="Search transcript, segment or video ID..."
              value={searchInput}
              onChange={(e) => { setSearchInput(e.target.value); setPage(1); }}
            />
          </div>

          <div className="gallery-chip-group">
            {[...FILTER_CHIPS, ...(capabilities.hasTiers ? TIER_CHIPS : [])].map((chip) => (
              <button
                key={chip.value}
                className={`chip ${filter === chip.value ? 'active' : ''}`}
                onClick={() => { setFilter(chip.value); setPage(1); }}
              >
                {chip.dot && <i className={`dot ${chip.dot}`}></i>}
                {chip.label}
              </button>
            ))}
          </div>

          {/* Region filter — shown only when region data is available. */}
          {regions.length > 0 && (
            <label className="gallery-sort">
              <span>Region</span>
              <select
                value={regionFilter}
                onChange={(e) => { setRegionFilter(e.target.value); setPage(1); }}
              >
                <option value="all">All</option>
                {regions.map((region) => (
                  <option key={region} value={region}>{region}</option>
                ))}
              </select>
            </label>
          )}

          <label className="gallery-sort">
            <span>Sort</span>
            <select
              value={sortOption}
              onChange={(e) => { setSortOption(e.target.value); setPage(1); }}
            >
              {SORT_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </label>
        </div>

        {/* Card grid */}
        <div className="gallery-grid" ref={gridRef}>
          {loading ? (
            <div className="gallery-empty">Loading...</div>
          ) : loadError ? (
            <div className="gallery-empty">Failed to load.</div>
          ) : pageSegments.length === 0 ? (
            <div className="gallery-empty">
              {segments.length === 0 ? 'No segments yet. Run the pipeline first.' : 'No matches.'}
            </div>
          ) : (
            pageSegments.map((segment) => (
              <SegmentCard
                key={`${segment.segment_id}-${crop}`}
                segment={segment}
                status={reviewStatusOf(reviewMap, segment.segment_id)}
                crop={crop}
                observer={observerRef.current}
                onOpen={setOpenSegmentId}
              />
            ))
          )}
        </div>

        {/* Simple pagination: forward/back only (the grid has 45 cards per page) */}
        {totalPages > 1 && (
          <div className="gallery-pagination">
            <span className="page-info">Page {safePage} of {totalPages}</span>
            <div className="page-btns">
              <button className="page-btn" onClick={() => goToPage(safePage - 1)} disabled={safePage === 1}>‹</button>
              <button className="page-btn" onClick={() => goToPage(safePage + 1)} disabled={safePage === totalPages}>›</button>
            </div>
          </div>
        )}
      </div>

      {/* Detail modal */}
      {openSegmentId && (
        <SegmentModal segmentId={openSegmentId} onClose={() => setOpenSegmentId(null)} />
      )}
    </main>
  );
}
