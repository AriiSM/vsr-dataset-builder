import { useState, useEffect } from 'react';
import { toast } from '../../components/toast.jsx';

/**
 * The trim bar with two handles: pick the [start, end] range of the clip
 * and send it to the backend, which re-cuts the video files with ffmpeg.
 *
 * The sliders work in percentages (0–100) of the clip duration; conversion
 * to seconds happens only for display and on Apply.
 */
export function TrimBar({ duration, onApply }) {
  const [inPct, setInPct] = useState(0);
  const [outPct, setOutPct] = useState(100);
  const [applying, setApplying] = useState(false);

  // Reset the handles when the segment changes (the duration is another clip).
  useEffect(() => {
    setInPct(0);
    setOutPct(100);
  }, [duration]);

  const inSec = (inPct / 100) * duration;
  const outSec = (outPct / 100) * duration;

  /** The start handle cannot pass the end handle. */
  function handleInChange(value) {
    const v = parseFloat(value);
    setInPct(v >= outPct ? Math.max(0, outPct - 0.5) : v);
  }

  function handleOutChange(value) {
    const v = parseFloat(value);
    setOutPct(v <= inPct ? Math.min(100, inPct + 0.5) : v);
  }

  async function handleApply() {
    if (outSec - inSec < 0.1) {
      toast.error('Trim range too short.');
      return;
    }
    setApplying(true);
    try {
      await onApply(inSec, outSec);
    } finally {
      setApplying(false);
    }
  }

  return (
    <div className="rv-trim-wrap">
      <div className="rv-trim-hdr">
        <span className="rv-section-label">Trim</span>
        <span className="rv-trim-range">{inSec.toFixed(2)}s – {outSec.toFixed(2)}s</span>
        <button className="btn-micro rv-trim-apply" onClick={handleApply} disabled={applying}>
          {applying ? '...' : '✂ Apply'}
        </button>
        <button className="btn-micro" onClick={() => { setInPct(0); setOutPct(100); }}>
          ↺ Reset
        </button>
      </div>
      {/* Double slider: two range inputs overlaid on the same track */}
      <div className="rv-trim-slider-wrap">
        <input
          type="range" className="rv-trim-slider" min="0" max="100" step="0.1"
          value={inPct} onChange={(e) => handleInChange(e.target.value)}
        />
        <input
          type="range" className="rv-trim-slider" min="0" max="100" step="0.1"
          value={outPct} onChange={(e) => handleOutChange(e.target.value)}
        />
        <div className="rv-trim-track">
          <div
            className="rv-trim-selection"
            style={{ left: `${inPct}%`, width: `${outPct - inPct}%` }}
          />
        </div>
      </div>
      <div className="rv-trim-labels">
        <span>{inSec.toFixed(2)}s</span>
        <span>{outSec.toFixed(2)}s</span>
      </div>
    </div>
  );
}
