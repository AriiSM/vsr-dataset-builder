import { STAGES, STAGE_META, parseBulkImportState, bulkImportFeedRows } from '../../utils/logParser.js';
import { ActivityFeed } from './ActivityFeed.jsx';

/** A small counter in the stage banner (value + label). */
function Counter({ value, label, colorClass = '' }) {
  return (
    <div className="factory-counter">
      <span className={`factory-counter-value ${colorClass}`}>{value}</span>
      <span className="factory-counter-label">{label}</span>
    </div>
  );
}

/** Formats elapsed seconds as "MM:SS" (or "H:MM:SS" past an hour). */
function formatElapsed(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

/**
 * Rough time-remaining estimate. Progress = finished videos plus the clip
 * fraction of the current one; the ETA extrapolates the elapsed time over
 * the remaining fraction. Returns null until there is enough signal
 * (>2% done, >30s elapsed) — better no estimate than a wild one.
 */
function estimateRemaining(parsed, elapsedSeconds) {
  const clipFraction = parsed.clipProgress.total > 0
    ? parsed.clipProgress.processed / parsed.clipProgress.total
    : 0;
  let fractionDone;
  if (parsed.videoTotal > 0) {
    fractionDone = (Math.max(0, parsed.videoPos - 1) + clipFraction) / parsed.videoTotal;
  } else {
    fractionDone = clipFraction; // single-video mode
  }
  if (fractionDone < 0.02 || elapsedSeconds < 30) return null;
  const remainingSeconds = (elapsedSeconds * (1 - fractionDone)) / fractionDone;
  const h = Math.floor(remainingSeconds / 3600);
  const m = Math.round((remainingSeconds % 3600) / 60);
  if (h > 0) return `~${h}h ${m}m left`;
  if (m > 0) return `~${m}m left`;
  return '~1m left';
}

/** The header title's text + color, depending on the overall state. */
function headerFor(controlState, runningTitle) {
  switch (controlState) {
    case 'running': return { text: runningTitle || 'Processing', color: 'var(--cyan)' };
    case 'done':    return { text: runningTitle || 'Complete', color: 'var(--green)' };
    case 'error':   return { text: runningTitle || 'Finished with errors', color: 'var(--red)' };
    default:        return { text: 'Standby', color: '' };
  }
}

/**
 * The right-hand panel (the "control room"): overall state, the current
 * stage's banner with counters, the activity feed, and the raw log.
 *
 * For bulk-import there are no stages — the banner shows per-URL progress.
 */
export function ControlRoom({ controlState, elapsedSeconds, parsed, logLines, mode, isRunning, finishedTitle }) {
  const isBulkImport = mode === 'bulk-import';

  // ── The stage banner's data ─────────────────────────────────────────────
  let banner;
  let feedEntries;

  if (isBulkImport) {
    const bulk = parseBulkImportState(logLines);
    const description = !bulk.total
      ? 'Starting import…'
      : bulk.decided === bulk.current
        ? `Processing URL ${bulk.current} of ${bulk.total}`
        : `URL ${bulk.current} of ${bulk.total} (${bulk.decided} decided)`;
    banner = {
      stageNum: bulk.total ? `${bulk.decided}/${bulk.total}` : '—',
      name: 'Bulk import',
      description,
      counters: [
        bulk.total   ? { value: bulk.decided, label: `of ${bulk.total} decided`, colorClass: 'accent' } : null,
        bulk.ok      ? { value: bulk.ok, label: 'ok', colorClass: 'green' } : null,
        bulk.failed  ? { value: bulk.failed, label: 'failed', colorClass: 'red' } : null,
        bulk.skipped ? { value: bulk.skipped, label: 'skipped', colorClass: '' } : null,
      ].filter(Boolean),
    };
    feedEntries = bulkImportFeedRows(logLines);
  } else {
    // The displayed stage: the active one, else the last finished one, else the first.
    const detailStage =
      parsed.currentStage ||
      STAGES.filter((s) => parsed.doneStages.has(s) || parsed.errorStages.has(s)).pop() ||
      STAGES[0];
    const stageIndex = STAGES.indexOf(detailStage);
    const stageName = STAGE_META[detailStage]?.name || detailStage.toUpperCase();
    const descriptionParts = [];
    if (parsed.videoTotal > 0) {
      descriptionParts.push(`Video ${parsed.videoPos} of ${parsed.videoTotal}`);
    }
    if (STAGE_META[detailStage]?.desc) {
      descriptionParts.push(STAGE_META[detailStage].desc);
    }
    const counter = parsed.clipProgress || {};
    banner = {
      stageNum: `${stageIndex + 1}/${STAGES.length}`,
      name: parsed.currentVideo ? `${stageName} · ${parsed.currentVideo}` : stageName,
      description: descriptionParts.join(' · ') || 'Select a mode on the left and start the pipeline',
      counters: [
        counter.total > 0
          ? { value: counter.processed, label: `of ${counter.total}`, colorClass: 'accent' }
          : counter.processed > 0
            ? { value: counter.processed, label: 'clips', colorClass: 'accent' }
            : null,
        parsed.exportedSegments > 0 ? { value: parsed.exportedSegments, label: 'exported', colorClass: 'green' } : null,
        parsed.finishedVideos > 0 ? { value: parsed.finishedVideos, label: 'done', colorClass: 'green' } : null,
        parsed.failedVideos > 0 ? { value: parsed.failedVideos, label: 'failed', colorClass: 'red' } : null,
      ].filter(Boolean),
    };
    feedEntries = parsed.allEntries;
  }

  // ── The header title ────────────────────────────────────────────────────
  let runningTitle = finishedTitle;
  if (isRunning) {
    if (isBulkImport) {
      const bulk = parseBulkImportState(logLines);
      runningTitle = bulk.total ? `Importing ${bulk.current}/${bulk.total}` : 'Importing…';
    } else if (parsed.currentStage) {
      const stageName = STAGE_META[parsed.currentStage]?.name || parsed.currentStage;
      const position = parsed.videoTotal > 0 ? ` · video ${parsed.videoPos}/${parsed.videoTotal}` : '';
      runningTitle = `${stageName}${position} · running`;
    }
  }
  const header = headerFor(controlState, runningTitle);

  return (
    <section className="factory-right">
      {/* Overall state + the timer */}
      <div className="control-header">
        <div className="control-header-left">
          <span className={`control-led ${controlState !== 'standby' ? controlState : ''}`}></span>
          <span className="control-title" style={{ color: header.color }}>{header.text}</span>
        </div>
        <span className="control-elapsed">
          {formatElapsed(elapsedSeconds)}
          {isRunning && !isBulkImport && (() => {
            const eta = estimateRemaining(parsed, elapsedSeconds);
            return eta ? <span className="control-eta"> · {eta}</span> : null;
          })()}
        </span>
      </div>

      {/* The current stage's banner */}
      <div className="control-stage">
        <div className="control-stage-left">
          <span className="control-stage-num">{banner.stageNum}</span>
          <div>
            <div className="control-stage-name">{banner.name}</div>
            <div className="control-stage-desc">{banner.description}</div>
          </div>
        </div>
        <div className="control-counters">
          {banner.counters.map((c, i) => (
            <Counter key={i} value={c.value} label={c.label} colorClass={c.colorClass} />
          ))}
        </div>
      </div>

      {/* The activity feed + the raw log */}
      <div className="control-body">
        <div className="control-activity-wrap">
          <div className="control-subhdr">
            <span className="control-subhdr-label">Activity feed</span>
            <span className="control-subhdr-hint">All events, color-tagged by stage · full log saved per session in logs/sessions/</span>
          </div>
          <ActivityFeed entries={feedEntries} />
        </div>
      </div>
    </section>
  );
}
