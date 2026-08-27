import { useState, useEffect, useRef } from 'react';

/**
 * Editable cell: plain text; on double-click it becomes an input that
 * saves on Enter/blur and cancels with Escape.
 * (Defined at module level, not inside the table — otherwise React would
 * remount the input on every render and the typed text would be lost.)
 */
function EditableCell({ isEditing, display, onStartEdit, onCommit, onCancel }) {
  if (isEditing) {
    return (
      <td>
        <input
          autoFocus
          defaultValue={display}
          style={{
            width: '100%', background: 'var(--bg)', border: '1px solid var(--cyan)',
            borderRadius: 3, color: 'var(--text)', font: 'inherit', padding: '1px 4px',
          }}
          onBlur={(e) => onCommit(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') e.currentTarget.blur();
            if (e.key === 'Escape') onCancel();
          }}
        />
      </td>
    );
  }
  return <td onDoubleClick={onStartEdit}>{display}</td>;
}

/**
 * The editable word-timings table (Word / Start / End / ASD).
 *
 *  - the ▶ button plays exactly the word's interval in the main player;
 *  - double-clicking a cell makes it editable (input); Enter/blur saves
 *    locally; the "Save" button sends the whole table to the backend;
 *  - ✕ deletes a word from the list.
 *
 * The timings may be stored with a non-zero origin (negative values
 * inherited from the pipeline) — we shift them for display so the first
 * word starts at 0, and undo the shift on save.
 */
export function WordTimingsTable({ words, videoRef, onSave }) {
  // The display shift: if the first start is negative, move everything to 0.
  const starts = words.map((w) => parseFloat(w.start)).filter((v) => !Number.isNaN(v));
  const offset = starts.length && Math.min(...starts) < 0 ? -Math.min(...starts) : 0;

  // The editable copy of the words, with timings already shifted for display.
  const [rows, setRows] = useState([]);
  const [dirty, setDirty] = useState(false);
  // The cell being edited: { rowIndex, field } or null.
  const [editingCell, setEditingCell] = useState(null);
  // The index of the word currently playing (to highlight the ▶ button).
  const [playingIndex, setPlayingIndex] = useState(null);
  // Row currently being spoken during normal playback (karaoke highlight).
  const [spokenIndex, setSpokenIndex] = useState(null);

  const stopTimerRef = useRef(null);
  const prevLoopRef = useRef(null);

  // Re-initialize the local copy when the segment changes.
  useEffect(() => {
    setRows(
      words.map((w) => ({
        word: w.word,
        start: (parseFloat(w.start) || 0) + offset,
        end: (parseFloat(w.end) || 0) + offset,
        asd: w.asd_score ?? w.score ?? w.asd ?? null,
      }))
    );
    setDirty(false);
    setEditingCell(null);
    stopWordPlayback();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [words]);

  // Follow normal playback and light up the row whose word is being said.
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !rows.length) return undefined;
    const handleTime = () => {
      const t = video.currentTime;
      let index = null;
      for (let i = 0; i < rows.length; i++) {
        if (t >= rows[i].start && t <= rows[i].end) { index = i; break; }
      }
      setSpokenIndex(index);
    };
    video.addEventListener('timeupdate', handleTime);
    return () => video.removeEventListener('timeupdate', handleTime);
  }, [rows, videoRef]);

  /** Stops word playback and restores the player state. */
  function stopWordPlayback() {
    if (stopTimerRef.current) {
      clearTimeout(stopTimerRef.current);
      stopTimerRef.current = null;
    }
    setPlayingIndex(null);
    const video = videoRef.current;
    if (video) {
      try { video.pause(); } catch { /* ignored */ }
      if (prevLoopRef.current !== null) {
        video.loop = prevLoopRef.current;
        prevLoopRef.current = null;
      }
    }
  }

  /**
   * Plays a single word: pause → seek → verify the seek settled →
   * play → stop after the word's duration. The rAF check exists because
   * in Chrome, setting currentTime on a playing video sometimes doesn't
   * take effect immediately.
   */
  function playWord(rowIndex) {
    const video = videoRef.current;
    const row = rows[rowIndex];
    if (!video || !row) return;
    const start = row.start;
    const end = row.end;
    if (Number.isNaN(start) || Number.isNaN(end) || end <= start) return;

    stopWordPlayback();

    const pad = 0.08;
    const videoDur = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 0;
    const seekTo = Math.max(0, Math.min(start - pad, (videoDur || 1e9) - 0.05));
    const stopAt = videoDur > 0 ? Math.min(videoDur - 0.02, end + pad) : end + pad;
    const playMs = Math.max(150, (stopAt - seekTo) * 1000);

    setPlayingIndex(rowIndex);
    prevLoopRef.current = video.loop;
    video.loop = false; // no loop, otherwise the video wraps around mid-word

    try { video.pause(); } catch { /* ignored */ }
    try { video.currentTime = seekTo; } catch { /* ignored */ }

    let attempts = 0;
    const tryPlay = () => {
      attempts++;
      if (Math.abs(video.currentTime - seekTo) > 0.2 && attempts < 20) {
        try { video.currentTime = seekTo; } catch { /* ignored */ }
        requestAnimationFrame(tryPlay);
        return;
      }
      video.play()?.catch(() => stopWordPlayback());
      stopTimerRef.current = setTimeout(stopWordPlayback, playMs);
    };
    requestAnimationFrame(tryPlay);
  }

  /** Saves a cell's edited value into the local copy. */
  function commitEdit(rowIndex, field, rawValue) {
    setRows((prev) =>
      prev.map((row, i) => {
        if (i !== rowIndex) return row;
        if (field === 'word') return { ...row, word: rawValue.trim() };
        if (field === 'asd') {
          const trimmed = rawValue.trim();
          return { ...row, asd: trimmed === '-' || trimmed === '' ? null : parseFloat(trimmed) || 0 };
        }
        return { ...row, [field]: parseFloat(rawValue) || 0 };
      })
    );
    setEditingCell(null);
    setDirty(true);
  }

  function deleteRow(rowIndex) {
    setRows((prev) => prev.filter((_, i) => i !== rowIndex));
    setDirty(true);
  }

  /** Sends the table to the backend, with the display shift undone. */
  async function handleSave() {
    const payload = rows.map((row) => ({
      word: row.word,
      start: row.start - offset,
      end: row.end - offset,
      asd_score: row.asd == null ? null : parseFloat(row.asd) || 0,
    }));
    const saved = await onSave(payload);
    if (saved) setDirty(false);
  }

  if (!rows.length) return null;

  /** Props for an editable cell at (rowIndex, field). */
  const cellProps = (rowIndex, field, display) => ({
    display,
    isEditing:
      editingCell != null &&
      editingCell.rowIndex === rowIndex &&
      editingCell.field === field,
    onStartEdit: () => setEditingCell({ rowIndex, field }),
    onCommit: (value) => commitEdit(rowIndex, field, value),
    onCancel: () => setEditingCell(null),
  });

  return (
    <div className="rv-section rv-section-words">
      <div className="rv-section-hdr">
        <span className="rv-section-label">Word timings</span>
        <span className="rv-section-hint">Double-click a cell to edit · ✕ to remove</span>
        {dirty && (
          <button className="btn-micro" onClick={handleSave}>Save</button>
        )}
      </div>
      <div className="rv-words-scroll">
        <table className="rv-words-table">
          <thead>
            <tr>
              <th style={{ width: 34 }}></th>
              <th>Word</th>
              <th>Start</th>
              <th>End</th>
              <th>ASD</th>
              <th style={{ width: 32 }}></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} className={i === spokenIndex ? 'word-active' : ''}>
                <td className="word-play-cell">
                  <button
                    className={`word-play-btn ${playingIndex === i ? 'playing' : ''}`}
                    title="Play this word"
                    onClick={() => playWord(i)}
                  >
                    <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor">
                      <polygon points="5 3 19 12 5 21 5 3" />
                    </svg>
                  </button>
                </td>
                <EditableCell {...cellProps(i, 'word', row.word)} />
                <EditableCell {...cellProps(i, 'start', row.start.toFixed(3))} />
                <EditableCell {...cellProps(i, 'end', row.end.toFixed(3))} />
                <EditableCell
                  {...cellProps(
                    i,
                    'asd',
                    row.asd != null && row.asd !== '' ? parseFloat(row.asd).toFixed(1) : '-'
                  )}
                />
                <td>
                  <button className="btn-del-word" title="Remove word" onClick={() => deleteRow(i)}>
                    ×
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
