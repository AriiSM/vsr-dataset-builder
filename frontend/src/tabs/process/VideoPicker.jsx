import { useState } from 'react';
import { formatDurationShort } from '../../utils/format.js';

/**
 * Checkbox picker for the videos a run will process.
 *
 * The list shows the videos eligible for the selected mode (pending for
 * Batch; interrupted/failed for Resume), in registry order. Quick actions:
 * Select all, Clear, and "Next X" — selects the first X eligible videos,
 * i.e. the next X after the last fully processed one, since processed
 * videos are no longer in this list.
 *
 * An empty selection means "run ALL eligible videos" (the backend default),
 * and the summary line says so explicitly.
 */
export function VideoPicker({ videos, selectedIds, onSelectionChange }) {
  const [nextCount, setNextCount] = useState('10');

  if (!videos.length) {
    return (
      <div className="video-picker-empty">
        No eligible videos in the registry for this mode.
      </div>
    );
  }

  const toggle = (videoId) => {
    const next = new Set(selectedIds);
    if (next.has(videoId)) next.delete(videoId);
    else next.add(videoId);
    onSelectionChange(next);
  };

  const selectAll = () => onSelectionChange(new Set(videos.map((v) => v.video_id)));
  const clear = () => onSelectionChange(new Set());
  const selectNext = () => {
    const count = parseInt(nextCount) || 0;
    if (count <= 0) return;
    onSelectionChange(new Set(videos.slice(0, count).map((v) => v.video_id)));
  };

  // Summary: what will actually run when Start is pressed.
  const selected = videos.filter((v) => selectedIds.has(v.video_id));
  const sumSourceS = (list) =>
    list.reduce((sum, v) => {
      const s = parseFloat(v.duration_seconds);
      return sum + (Number.isFinite(s) ? s : 0);
    }, 0);
  const summary = selected.length
    ? `${selected.length} of ${videos.length} selected · ${formatDurationShort(sumSourceS(selected))} of source video`
    : `no selection — all ${videos.length} videos will run (${formatDurationShort(sumSourceS(videos))})`;

  return (
    <div className="video-picker">
      {/* Quick selection actions */}
      <div className="video-picker-actions">
        <button className="btn-micro" onClick={selectAll}>Select all</button>
        <button className="btn-micro" onClick={clear}>Clear</button>
        <span className="video-picker-next">
          <button className="btn-micro" onClick={selectNext}>Select next</button>
          <input
            className="text-input-sm"
            type="number"
            min={1}
            value={nextCount}
            onChange={(e) => setNextCount(e.target.value)}
            aria-label="How many videos to select"
          />
        </span>
      </div>

      {/* The eligible videos, in registry order */}
      <div className="video-picker-list">
        {videos.map((video) => (
          <label className="video-picker-row" key={video.video_id}>
            <input
              type="checkbox"
              checked={selectedIds.has(video.video_id)}
              onChange={() => toggle(video.video_id)}
            />
            <span className="video-picker-id mono">{video.video_id}</span>
            <span className="video-picker-title" title={video.title || ''}>
              {video.title || '—'}
            </span>
            <span className="video-picker-dur mono">
              {Number.isFinite(parseFloat(video.duration_seconds))
                ? formatDurationShort(parseFloat(video.duration_seconds))
                : '—'}
            </span>
            <span className={`badge ${video.status || 'pending'}`}>{video.status}</span>
          </label>
        ))}
      </div>

      <div className="video-picker-summary">{summary}</div>
    </div>
  );
}
