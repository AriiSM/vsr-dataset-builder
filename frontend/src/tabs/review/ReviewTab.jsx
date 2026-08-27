import { useState, useEffect, useRef, useCallback } from 'react';
import { api } from '../../api.js';
import { toast } from '../../components/toast.jsx';
import {
  formatDuration, asdColorClass, whisperColorClass,
} from '../../utils/format.js';
import { TrimBar } from './TrimBar.jsx';
import { SpeakerSelect } from './SpeakerSelect.jsx';
import { WordTimingsTable } from './WordTimingsTable.jsx';
import { useLocalStorage } from '../../hooks/useLocalStorage.js';
import { detectSegmentCapabilities, normalizeTier, TIER_META } from '../../utils/datasetCapabilities.js';
import { methodConfidenceRows } from '../../utils/methodInfo.jsx';
import { diffWords } from '../../utils/wordDiff.js';

/** Labels shown in the status ribbon. */
const STATUS_LABELS = {
  pending: 'Pending review',
  approved: 'Approved',
  rejected: 'Rejected',
  edited: 'Edited',
};

/** Loose truthiness for CSV-borne flags ('True', 'true', '1', 1, true). */
function isFlagSet(value) {
  return value === true || value === 1 || /^(true|1)$/i.test(String(value || ''));
}

/**
 * Thematic review queues. Beyond the classic pending/all, the v3 queues
 * target specific verification work; each appears only when its data
 * exists (capability-gated).
 */
const QUEUE_MODES = [
  { value: 'pending',         label: 'Pending only' },
  { value: 'all',             label: 'All segments' },
  { value: 'needs-review',    label: 'Needs review (refiner)',   capability: 'hasRefiner' },
  { value: 'auto-identity',   label: 'Auto-matched identities',  capability: 'hasIdentity' },
  { value: 'mouth-fallback',  label: 'Mouth-ROI fallback',       capability: 'hasMouthRoi' },
  { value: 'forced-boundary', label: 'Forced boundaries',        capability: 'hasBoundaries' },
];

/**
 * Sort orders for the review queue. The "worst first" options put the clips
 * most likely to need fixing at the front, so curation time goes where it
 * matters most.
 */
const QUEUE_ORDERS = [
  { value: 'default',       label: 'File order' },
  { value: 'worst-whisper', label: 'Worst Whisper first' },
  { value: 'worst-asd',     label: 'Worst ASD first' },
  { value: 'shortest',      label: 'Shortest first' },
  { value: 'longest',       label: 'Longest first' },
];

/** Returns the queue sorted by the chosen order (missing scores sort first). */
function orderQueue(queue, order) {
  if (order === 'default') return queue;
  const num = (value, missing) => {
    const n = parseFloat(value);
    return Number.isFinite(n) ? n : missing;
  };
  const sorted = [...queue];
  switch (order) {
    case 'worst-whisper':
      sorted.sort((a, b) => num(a.whisper_conf, 0) - num(b.whisper_conf, 0));
      break;
    case 'worst-asd':
      sorted.sort((a, b) => num(a.asd_score, 0) - num(b.asd_score, 0));
      break;
    case 'shortest':
      sorted.sort((a, b) => num(a.duration, Infinity) - num(b.duration, Infinity));
      break;
    case 'longest':
      sorted.sort((a, b) => num(b.duration, 0) - num(a.duration, 0));
      break;
    default:
      break;
  }
  return sorted;
}

/**
 * The Review tab: segment-by-segment curation.
 *
 * The flow: the queue is loaded (optionally only pending segments), the
 * current segment is shown with video + transcript + word timings, and the
 * curator approves (A), rejects (R), edits (E) or navigates (←/→, S=skip).
 * Reject deletes the files from disk, so it removes the segment from the queue.
 */
export function ReviewTab({ isActive }) {
  // ── The queue and the position in it ──
  const [queue, setQueue] = useState([]);
  const [index, setIndex] = useState(0);
  const [reviewMap, setReviewMap] = useState({});
  // The current segment, fully loaded from /api/segment/<id>.
  const [current, setCurrent] = useState(null);
  // Message shown when there is nothing to display (empty queue / error).
  const [emptyMessage, setEmptyMessage] = useState('Loading…');

  // ── Options (persisted across page refreshes) ──
  const [queueMode, setQueueMode] = useLocalStorage('vsr-review-queue', 'pending');
  const [autoAdvance, setAutoAdvance] = useLocalStorage('vsr-review-autoadvance', true);
  const [crop, setCrop] = useState('face'); // reset per segment, not persisted
  // Region filter (MD-only dataset target) + queue ordering.
  const [regionFilter, setRegionFilter] = useLocalStorage('vsr-review-region', 'all');
  const [queueOrder, setQueueOrder] = useLocalStorage('vsr-review-order', 'default');
  // The regions present in the data, for the filter dropdown.
  const [regions, setRegions] = useState([]);
  // v3 capabilities of the loaded data + identity info for the queue options.
  const [capabilities, setCapabilities] = useState({});

  // Playback speed — 0.25×/0.5× make lip-sync checking far more precise.
  // Persists across segments within the session.
  const [playbackRate, setPlaybackRate] = useState(1);
  // Index of the word currently being spoken (karaoke highlight), or null.
  const [activeWordIndex, setActiveWordIndex] = useState(null);
  // The "?" keyboard cheat-sheet overlay.
  const [showCheatsheet, setShowCheatsheet] = useState(false);

  // ── Transcript editing ──
  const [isEditing, setIsEditing] = useState(false);
  const [editorText, setEditorText] = useState('');
  // The status shown on the colored ribbon ('edited' is only visual, not in reviewMap).
  const [ribbonStatus, setRibbonStatus] = useState('pending');

  const videoRef = useRef(null);
  // Reject is destructive (it deletes the clip's files from disk), so it is
  // armed on the first press and only executes on a second press/click on the
  // SAME segment within a short window.
  const pendingRejectRef = useRef(null); // { segmentId, timer } | null

  const annotation = current?.annotation || {};
  const duration = parseFloat(current?.duration || 0);

  /** Loads a segment from the queue by index. */
  const loadSegment = useCallback(async (idx, queueOverride, reviewOverride) => {
    const q = queueOverride || queue;
    const rMap = reviewOverride || reviewMap;
    if (idx < 0 || idx >= q.length) return;
    setIndex(idx);
    setIsEditing(false);
    try {
      const data = await api.getSegmentDetail(q[idx].segment_id);
      setCurrent(data);
      setCrop('face');
      setRibbonStatus(rMap[data.segment_id]?.status || 'pending');
    } catch (err) {
      setCurrent(null);
      setEmptyMessage(`Failed to load segment: ${err.message || err}`);
    }
  }, [queue, reviewMap]);

  /** (Re)builds the queue from /api/segments + /api/review_status. */
  const loadQueue = useCallback(async () => {
    try {
      const [segRes, revRes, spkRes] = await Promise.all([
        api.getSegments(),
        api.getReviewStatus(),
        api.getSpeakers().catch(() => ({ speakers: [] })),
      ]);
      const rMap = revRes || {};
      const allSegments = segRes.segments || [];
      const speakers = spkRes.speakers || [];
      setRegions([...new Set(allSegments.map((s) => s.region).filter(Boolean))].sort());

      // Which v3 features exist in this dataset — gates the queue options.
      const segCaps = detectSegmentCapabilities(allSegments);
      const hasIdentity = speakers.some((sp) => 'identity_match' in sp);
      setCapabilities({ ...segCaps, hasIdentity });

      // Speaker ids matched automatically across videos — their segments
      // form the "Auto-matched identities" verification queue.
      const autoSpeakerIds = new Set(
        speakers
          .filter((sp) => String(sp.identity_match || '').trim() === 'auto')
          .map((sp) => sp.speaker_id)
      );

      /** Queue-mode predicate; thematic queues ignore the review status. */
      const inQueue = (s) => {
        const status = rMap[s.segment_id]?.status || 'pending';
        switch (queueMode) {
          case 'all':             return true;
          case 'needs-review':    return isFlagSet(s.needs_review);
          case 'auto-identity':   return autoSpeakerIds.has(s.speaker_id);
          case 'mouth-fallback':  return /fallback/i.test(String(s.mouth_roi_method || ''));
          case 'forced-boundary': return /forced/i.test(String(s.boundary_type || ''));
          case 'pending':
          default:
            return status === 'pending';
        }
      };

      const filtered = allSegments.filter((s) => {
        if (!inQueue(s)) return false;
        if (regionFilter !== 'all' && (s.region || 'UNKNOWN') !== regionFilter) return false;
        return true;
      });
      const newQueue = orderQueue(filtered, queueOrder);
      setReviewMap(rMap);
      setQueue(newQueue);
      if (newQueue.length === 0) {
        setCurrent(null);
        const queueLabel =
          QUEUE_MODES.find((m) => m.value === queueMode)?.label || queueMode;
        setEmptyMessage(
          allSegments.length === 0
            ? 'No segments in queue. Run the pipeline first.'
            : queueMode === 'pending' && regionFilter === 'all'
              ? 'All segments already reviewed. Switch the queue to "All segments" to see them.'
              : `No segments match "${queueLabel}"${regionFilter !== 'all' ? ` for region ${regionFilter}` : ''}.`
        );
      } else {
        await loadSegment(0, newQueue, rMap);
      }
    } catch (err) {
      setCurrent(null);
      setEmptyMessage(`Failed to load review queue: ${err.message || err}`);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queueMode, regionFilter, queueOrder]);

  // Reload the queue when the tab is activated or any queue option changes.
  useEffect(() => {
    if (isActive) loadQueue();
  }, [isActive, loadQueue]);

  /** Navigates the queue by delta steps (−1 / +1). */
  const navigate = useCallback((delta) => {
    const next = index + delta;
    if (next >= 0 && next < queue.length) loadSegment(next);
  }, [index, queue.length, loadSegment]);

  /** Approve / Reject / Revert on the current segment. */
  const runAction = useCallback(async (action) => {
    if (!current) return;
    const id = current.segment_id;

    // Two-step confirmation for the destructive reject.
    if (action === 'reject') {
      const pending = pendingRejectRef.current;
      if (!pending || pending.segmentId !== id) {
        if (pending) clearTimeout(pending.timer);
        pendingRejectRef.current = {
          segmentId: id,
          timer: setTimeout(() => { pendingRejectRef.current = null; }, 4000),
        };
        toast.info('Reject deletes this clip from disk — press R or click Reject again to confirm.');
        return;
      }
      clearTimeout(pending.timer);
      pendingRejectRef.current = null;
    }

    try {
      const { ok, data } = await api.reviewSegment(id, { action });
      if (!ok) {
        toast.error(data.error || 'Action failed');
        return;
      }

      const newStatus =
        action === 'approve' ? 'approved' : action === 'reject' ? 'rejected' : 'pending';
      const newReviewMap = { ...reviewMap, [id]: { status: newStatus } };
      setReviewMap(newReviewMap);
      setRibbonStatus(newStatus);

      // Reject deletes the segment from disk → also remove it from the queue.
      if (action === 'reject') {
        const newQueue = queue.filter((_, i) => i !== index);
        setQueue(newQueue);
        if (newQueue.length === 0) {
          setCurrent(null);
          setEmptyMessage('All segments reviewed!');
        } else {
          await loadSegment(Math.min(index, newQueue.length - 1), newQueue, newReviewMap);
        }
        return;
      }

      if (autoAdvance && index < queue.length - 1) {
        await loadSegment(index + 1, undefined, newReviewMap);
      }
    } catch (err) {
      toast.error(`Network error: ${err.message}`);
    }
  }, [current, reviewMap, queue, index, autoAdvance, loadSegment]);

  /** Saves the edited transcript. */
  async function saveTranscript() {
    if (!current) return;
    const text = editorText.trim();
    const { ok, data } = await api.reviewSegment(current.segment_id, { action: 'save', text });
    if (!ok) {
      toast.error(data.error || 'Save failed');
      return;
    }
    toast.success('Transcript saved');
    setCurrent({ ...current, annotation: { ...annotation, text } });
    setIsEditing(false);
    if (ribbonStatus === 'pending') setRibbonStatus('edited');
  }

  /** Applies the large-v3 refiner's suggested transcript. */
  async function applySuggestion(text) {
    if (!current) return;
    const { ok, data } = await api.reviewSegment(current.segment_id, { action: 'save', text });
    if (!ok) {
      toast.error(data.error || 'Save failed');
      return;
    }
    toast.success('large-v3 transcript applied');
    setCurrent({ ...current, annotation: { ...annotation, text } });
    if (ribbonStatus === 'pending') setRibbonStatus('edited');
  }

  /** Saves the edited word table; returns true on success. */
  async function saveWords(words) {
    if (!current) return false;
    const { ok, data } = await api.reviewSegment(current.segment_id, {
      action: 'save_words',
      words,
    });
    if (!ok) {
      toast.error(data.error || 'Save failed');
      return false;
    }
    toast.success('Word timings saved');
    setCurrent({ ...current, annotation: { ...annotation, words } });
    return true;
  }

  /** Applies the trim and reloads the segment (new duration/video). */
  async function applyTrim(start, end) {
    if (!current) return;
    const { ok, data } = await api.trimSegment(current.segment_id, start, end);
    if (!ok) {
      toast.error(data.error || 'Trim failed');
      return;
    }
    toast.success('Clip trimmed');
    // Keep the queue entry in sync so duration-based ordering stays correct.
    if (data.duration != null) {
      setQueue((prev) =>
        prev.map((item) =>
          item.segment_id === current.segment_id
            ? { ...item, duration: data.duration }
            : item
        )
      );
    }
    await loadSegment(index);
  }

  // ── Keyboard shortcuts (only when the tab is active) ──
  useEffect(() => {
    if (!isActive) return undefined;
    const handleKey = (e) => {
      // Don't intercept keys while the user is typing in a field.
      const tag = e.target.tagName;
      if (tag === 'TEXTAREA' || tag === 'INPUT' || e.target.isContentEditable) return;
      if (e.key === '?') { setShowCheatsheet((v) => !v); return; }
      if (e.key === 'Escape') { setShowCheatsheet(false); return; }
      if (e.key === ' ') {
        // Space toggles playback without needing the player focused.
        e.preventDefault();
        const video = videoRef.current;
        if (video) {
          if (video.paused) video.play().catch(() => {});
          else video.pause();
        }
        return;
      }
      switch (e.key.toUpperCase()) {
        case 'A': runAction('approve'); break;
        case 'R': runAction('reject'); break;
        case 'S': navigate(1); break;
        case 'U': runAction('revert'); break;
        case 'E': setIsEditing(true); setEditorText(annotation.text || ''); break;
        case 'ARROWLEFT': navigate(-1); break;
        case 'ARROWRIGHT': navigate(1); break;
        default: break;
      }
    };
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [isActive, runAction, navigate, annotation.text]);

  // Switch the video source when the segment or crop changes. Depends on the
  // segment id, NOT the whole `current` object — otherwise every transcript
  // or word-table save (which replaces `current`) would restart the video.
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !current?.has_video) return;
    video.load();
    video.playbackRate = playbackRate;
    video.play().catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current?.segment_id, current?.has_video, crop]);

  // Rate changes apply immediately to the playing video.
  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = playbackRate;
  }, [playbackRate]);

  // ── Karaoke sync: track which word is being spoken ──
  // Word times may be stored with a shifted origin (negative starts); the
  // same display shift as the word table brings them onto the video timeline.
  useEffect(() => {
    const video = videoRef.current;
    const words = annotation.words;
    if (!video || !words?.length) {
      setActiveWordIndex(null);
      return undefined;
    }
    const starts = words.map((w) => parseFloat(w.start)).filter(Number.isFinite);
    const offset = starts.length && Math.min(...starts) < 0 ? -Math.min(...starts) : 0;
    const handleTime = () => {
      const t = video.currentTime;
      let index = null;
      for (let i = 0; i < words.length; i++) {
        const start = (parseFloat(words[i].start) || 0) + offset;
        const end = (parseFloat(words[i].end) || 0) + offset;
        if (t >= start && t <= end) { index = i; break; }
      }
      setActiveWordIndex(index);
    };
    video.addEventListener('timeupdate', handleTime);
    return () => video.removeEventListener('timeupdate', handleTime);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current?.segment_id, annotation.words]);

  /** Seeks the video to a word (clicked in the transcript) and plays. */
  function seekToWord(wordIndex) {
    const video = videoRef.current;
    const words = annotation.words;
    if (!video || !words?.[wordIndex]) return;
    const starts = words.map((w) => parseFloat(w.start)).filter(Number.isFinite);
    const offset = starts.length && Math.min(...starts) < 0 ? -Math.min(...starts) : 0;
    const target = Math.max(0, (parseFloat(words[wordIndex].start) || 0) + offset - 0.05);
    try { video.currentTime = target; } catch { /* not seekable yet */ }
    video.play().catch(() => {});
  }

  // ── Statistics for the top bar ──
  const counts = { approved: 0, rejected: 0, pending: 0 };
  for (const s of queue) {
    const status = reviewMap[s.segment_id]?.status || 'pending';
    counts[status in counts ? status : 'pending']++;
  }
  const reviewedCount = counts.approved + counts.rejected;
  const progressPct = queue.length ? Math.round((reviewedCount / queue.length) * 100) : 0;

  // ── Metadata chips colored by quality ──
  const chips = [];
  if (current) {
    const asd = parseFloat(current.asd_score || 0);
    const whisper = parseFloat(current.whisper_confidence || current.whisper_conf || 0);
    if (duration > 0) chips.push({ k: 'Duration', v: formatDuration(duration), cls: '' });
    if (current.num_words) chips.push({ k: 'Words', v: String(current.num_words), cls: '' });
    if (current.asd_score) chips.push({ k: 'ASD', v: asd.toFixed(1), cls: asdColorClass(asd) });
    if (current.whisper_confidence || current.whisper_conf) {
      chips.push({ k: 'Whisper', v: whisper.toFixed(2), cls: whisperColorClass(whisper) });
    }
  }

  // ── Info grid ──
  const fmtNum = (v, digits = 2) => {
    const n = parseFloat(v);
    return v != null && v !== '' && !Number.isNaN(n) ? n.toFixed(digits) : '-';
  };
  const infoFields = current
    ? [
        ['Video',     current.video_id, 'mono'],
        ['Duration',  duration > 0 ? formatDuration(duration) : '-', ''],
        ['Words',     current.num_words || '-', ''],
        ['ASD Score', fmtNum(current.asd_score), ''],
        ['Whisper ↑', fmtNum(current.whisper_confidence || current.whisper_conf), ''],
        ['Conf',      annotation.conf != null ? `${annotation.conf}/3` : '-', ''],
      ]
    : [];

  const hasSegment = current != null;

  return (
    <main className={`tab-content ${isActive ? 'active' : ''}`} id="tab-review">
      {/* ── Top bar: progress + navigation ── */}
      <div className="rv-topbar">
        <div className="rv-topbar-row">
          <div className="rv-topbar-left">
            <span className="rv-topbar-title">Review queue</span>
            <span className="rv-topbar-hint">
              {hasSegment
                ? `${reviewedCount} / ${queue.length} reviewed (${progressPct}%)`
                : emptyMessage}
            </span>
          </div>

          <div className="rv-queue-cards">
            <div className="rv-q-card green">
              <span className="rv-q-count">{counts.approved}</span>
              <span className="rv-q-label">Approved</span>
            </div>
            <div className="rv-q-card red">
              <span className="rv-q-count">{counts.rejected}</span>
              <span className="rv-q-label">Rejected</span>
            </div>
            <div className="rv-q-card muted">
              <span className="rv-q-count">{counts.pending}</span>
              <span className="rv-q-label">Pending</span>
            </div>
          </div>

          <div className="rv-topbar-nav">
            <button
              className="rv-nav-btn"
              onClick={() => navigate(-1)}
              disabled={!hasSegment || index === 0}
            >
              ‹ Prev
            </button>
            <span className="rv-nav-count">
              {hasSegment ? `${index + 1} / ${queue.length}` : '— / —'}
            </span>
            <button
              className="rv-nav-btn"
              onClick={() => navigate(1)}
              disabled={!hasSegment || index === queue.length - 1}
            >
              Next ›
            </button>
          </div>
        </div>
        <div className="rv-progress-bar">
          <div className="rv-progress-fill" style={{ width: `${hasSegment ? progressPct : 0}%` }} />
        </div>
      </div>

      <div className="rv-layout">
        {/* ═══ Left: video + trim + options ═══ */}
        <div className="rv-left">
          <div className="rv-video-wrap">
            {hasSegment && current.has_video ? (
              <video
                ref={videoRef}
                controls
                loop
                style={{ width: '100%', background: '#000' }}
              >
                <source
                  src={api.mediaUrl(current.video_id, current.segment_id, crop)}
                  type="video/mp4"
                />
              </video>
            ) : (
              <div className="rv-video-placeholder">
                <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <polygon points="23 7 16 12 23 17 23 7"></polygon>
                  <rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect>
                </svg>
                <span>{hasSegment ? 'No video file found' : 'No segment selected'}</span>
              </div>
            )}
          </div>

          {/* Player controls: crop + playback speed (slow-motion is the
              precise way to check lip-sync). */}
          {hasSegment && current.has_video && (
            <div className="rv-player-controls">
              <div className="rv-crop-toggle">
                <button
                  className={`rv-crop-btn ${crop === 'face' ? 'active' : ''}`}
                  onClick={() => setCrop('face')}
                >
                  Face
                </button>
                <button
                  className={`rv-crop-btn ${crop === 'mouth' ? 'active' : ''}`}
                  onClick={() => setCrop('mouth')}
                >
                  Mouth
                </button>
              </div>
              <div className="rv-crop-toggle" role="group" aria-label="Playback speed">
                {[0.25, 0.5, 1].map((rate) => (
                  <button
                    key={rate}
                    className={`rv-crop-btn ${playbackRate === rate ? 'active' : ''}`}
                    onClick={() => setPlaybackRate(rate)}
                  >
                    {rate}×
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Trim bar */}
          {hasSegment && current.has_video && duration > 0 && (
            // Keyed by segment id so the handles reset on every navigation —
            // resetting only on a duration change would keep stale positions
            // when two consecutive clips happen to be equally long.
            <TrimBar key={current.segment_id} duration={duration} onApply={applyTrim} />
          )}

          {/* Metadata chips */}
          <div className="rv-meta">
            {chips.map((chip) => (
              <span className={`rv-meta-chip ${chip.cls}`} key={chip.k}>
                <span className="rv-meta-chip-k">{chip.k}</span>
                <span className="rv-meta-chip-v">{chip.v}</span>
              </span>
            ))}
          </div>

          {/* Options */}
          <div className="rv-options">
            <label className="rv-opt-label">
              <input
                type="checkbox"
                checked={autoAdvance}
                onChange={(e) => setAutoAdvance(e.target.checked)}
              />
              Auto-advance after approve/reject
            </label>
            {/* Thematic queue selector — v3 queues appear only when their
                data exists (capability-gated). */}
            <label className="rv-opt-label" style={{ gap: 8 }}>
              Queue
              <select
                className="select-sm"
                value={queueMode}
                onChange={(e) => setQueueMode(e.target.value)}
              >
                {QUEUE_MODES
                  .filter((m) => !m.capability || capabilities[m.capability])
                  .map((m) => (
                    <option key={m.value} value={m.value}>{m.label}</option>
                  ))}
              </select>
            </label>

            {/* Region filter — the dataset target is MD-only, so this lets
                the curator spend review time only on target clips. */}
            <label className="rv-opt-label" style={{ marginTop: 6, gap: 8 }}>
              Region
              <select
                className="select-sm"
                value={regionFilter}
                onChange={(e) => setRegionFilter(e.target.value)}
              >
                <option value="all">All</option>
                {regions.map((region) => (
                  <option key={region} value={region}>{region}</option>
                ))}
              </select>
            </label>

            {/* Queue order — "worst first" prioritizes problem clips. */}
            <label className="rv-opt-label" style={{ marginTop: 6, gap: 8 }}>
              Order
              <select
                className="select-sm"
                value={queueOrder}
                onChange={(e) => setQueueOrder(e.target.value)}
              >
                {QUEUE_ORDERS.map((o) => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
            </label>

            <button className="btn-micro" style={{ marginTop: 8 }} onClick={loadQueue}>
              ↺ Reload queue
            </button>
            <button
              className="btn-micro"
              style={{ marginTop: 6 }}
              onClick={() => setShowCheatsheet(true)}
            >
              ? Keyboard shortcuts
            </button>
          </div>
        </div>

        {/* ═══ Right: the review panel ═══ */}
        <div className="rv-right">
          {/* Status ribbon */}
          <div className={`rv-status-ribbon ${ribbonStatus}`}>
            <span className="rv-status-dot"></span>
            <span className="rv-status-text">{STATUS_LABELS[ribbonStatus] || ribbonStatus}</span>
            {hasSegment && normalizeTier(current.quality_tier) && (
              <span
                className={`rv-tier tier-${normalizeTier(current.quality_tier).toLowerCase()}`}
                title={`${TIER_META[normalizeTier(current.quality_tier)].label} — computed by quality_indexer`}
              >
                Tier {normalizeTier(current.quality_tier)}
              </span>
            )}
            <span className="rv-seg-id">{hasSegment ? current.segment_id : '—'}</span>
          </div>

          {/* Transcript (editable) */}
          <div className="rv-section">
            <div className="rv-section-hdr">
              <span className="rv-section-label">Transcript</span>
              {!isEditing && (
                <button
                  className="btn-micro"
                  disabled={!hasSegment}
                  onClick={() => { setIsEditing(true); setEditorText(annotation.text || ''); }}
                >
                  Edit
                </button>
              )}
            </div>
            {isEditing ? (
              <>
                <textarea
                  className="transcript-editor"
                  rows={3}
                  autoFocus
                  value={editorText}
                  onChange={(e) => setEditorText(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) saveTranscript();
                    if (e.key === 'Escape') setIsEditing(false);
                  }}
                />
                <div className="editor-actions">
                  <button className="review-btn save" onClick={saveTranscript}>Save text</button>
                  <button className="btn-micro" onClick={() => setIsEditing(false)}>Cancel</button>
                </div>
              </>
            ) : (
              <div className="rv-transcript">
                {!hasSegment ? (
                  'Select a segment to start reviewing.'
                ) : annotation.words?.length ? (
                  // Clickable karaoke transcript: click a word to jump the
                  // video there; the word being spoken lights up live.
                  annotation.words.map((w, i) => (
                    <span
                      key={i}
                      className={`rv-word${i === activeWordIndex ? ' active' : ''}`}
                      title="Click to play from this word"
                      onClick={() => seekToWord(i)}
                    >
                      {w.word}{' '}
                    </span>
                  ))
                ) : (
                  annotation.text || '(no transcript)'
                )}
              </div>
            )}

            {/* large-v3 refiner suggestion (v3 only): word-level diff vs the
                current transcript, with one-click apply. */}
            {hasSegment && !isEditing && (() => {
              const suggestion = String(current.text_largev3 || '').trim();
              const currentText = String(annotation.text || '').trim();
              if (!suggestion || suggestion === 'nan' || suggestion === currentText) return null;
              return (
                <div className="rv-suggestion">
                  <div className="rv-suggestion-hdr">
                    <span className="rv-section-label">large-v3 suggests</span>
                    <button className="btn-micro" onClick={() => applySuggestion(suggestion)}>
                      Apply
                    </button>
                  </div>
                  <div className="rv-suggestion-text">
                    {diffWords(currentText, suggestion).map((op, i) => (
                      <span key={i} className={op.type === 'changed' ? 'diff-changed' : ''}>
                        {op.word}{' '}
                      </span>
                    ))}
                  </div>
                </div>
              );
            })()}
          </div>

          {/* Action buttons — Approve/Reject dominate */}
          <div className="rv-actions">
            <button
              className="rv-action-btn rv-approve primary"
              disabled={!hasSegment}
              onClick={() => runAction('approve')}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <polyline points="20 6 9 17 4 12" />
              </svg>
              Approve
              <kbd className="rv-action-kbd">A</kbd>
            </button>
            <button
              className="rv-action-btn rv-reject primary"
              disabled={!hasSegment}
              onClick={() => runAction('reject')}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
              Reject
              <kbd className="rv-action-kbd">R</kbd>
            </button>
            <button
              className="rv-action-btn rv-skip"
              disabled={!hasSegment}
              title="Skip (S / →)"
              onClick={() => navigate(1)}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="9 18 15 12 9 6" />
              </svg>
              Skip
            </button>
            <button
              className="rv-action-btn rv-revert"
              disabled={!hasSegment}
              title="Set back to pending (U)"
              onClick={() => runAction('revert')}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="1 4 1 10 7 10" />
                <path d="M3.51 15a9 9 0 1 0 .49-3.85" />
              </svg>
              Revert
            </button>
          </div>

          {/* Info grid + speaker */}
          <div className="rv-section rv-section-info">
            <div className="rv-section-hdr">
              <span className="rv-section-label">Segment info</span>
            </div>
            <div className="detail-info-grid" style={{ marginTop: 0 }}>
              {infoFields.map(([label, value, cls]) => (
                <div className="detail-info-item" key={label}>
                  <span className="detail-info-label">{label}</span>
                  <span className={`detail-info-value${cls ? ` ${cls}` : ''}`}>{String(value)}</span>
                </div>
              ))}
              {hasSegment && (
                <div className="detail-info-item rv-speaker-cell">
                  <span className="detail-info-label">Speaker</span>
                  <span className="detail-info-value">
                    <SpeakerSelect
                      segmentId={current.segment_id}
                      videoId={current.video_id}
                      currentSpeakerId={String(current.speaker_id || '').trim()}
                      onChanged={(newId) => setCurrent({ ...current, speaker_id: newId })}
                    />
                  </span>
                </div>
              )}
              {/* v3 provenance: which method produced each score, ROI
                  reliability, boundary type — warnings highlighted. */}
              {hasSegment &&
                methodConfidenceRows(current).map(({ label, value, warn }) => (
                  <div className="detail-info-item" key={label}>
                    <span className="detail-info-label">{label}</span>
                    <span className={`detail-info-value${warn ? ' value-warn' : ''}`}>
                      {value}
                    </span>
                  </div>
                ))}
            </div>
          </div>

          {/* Word table (editable) */}
          {hasSegment && (
            <WordTimingsTable
              words={annotation.words || []}
              videoRef={videoRef}
              onSave={saveWords}
            />
          )}
        </div>
      </div>

      {/* Keyboard cheat-sheet (toggled with "?") */}
      {showCheatsheet && (
        <div
          className="rv-cheatsheet-backdrop"
          onClick={(e) => { if (e.target === e.currentTarget) setShowCheatsheet(false); }}
        >
          <div className="rv-cheatsheet" role="dialog" aria-modal="true" aria-label="Keyboard shortcuts">
            <div className="rv-cheatsheet-hdr">
              <span className="rv-section-label">Keyboard shortcuts</span>
              <button className="btn-micro" onClick={() => setShowCheatsheet(false)}>✕</button>
            </div>
            {[
              ['A', 'Approve segment'],
              ['R  R', 'Reject segment (press twice — deletes files)'],
              ['S  /  →', 'Skip to the next segment'],
              ['←', 'Previous segment'],
              ['U', 'Revert to pending'],
              ['E', 'Edit transcript'],
              ['Space', 'Play / pause the video'],
              ['Ctrl+Enter', 'Save transcript (while editing)'],
              ['Esc', 'Cancel editing / close this panel'],
              ['?', 'Toggle this panel'],
            ].map(([keys, action]) => (
              <div className="rv-cheatsheet-row" key={keys}>
                <kbd>{keys}</kbd>
                <span>{action}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </main>
  );
}
