import { STAGES } from '../../utils/logParser.js';

/** The SVG icons of the five stages (same as in the original UI). */
const STAGE_ICONS = {
  download: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" y1="15" x2="12" y2="3" />
    </svg>
  ),
  vad: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z" />
      <path d="M19 10v2a7 7 0 01-14 0v-2" />
      <line x1="12" y1="19" x2="12" y2="23" />
      <line x1="8" y1="23" x2="16" y2="23" />
    </svg>
  ),
  face: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  ),
  asr: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="4 7 4 4 20 4 20 7" />
      <line x1="9" y1="20" x2="15" y2="20" />
      <line x1="12" y1="4" x2="12" y2="20" />
    </svg>
  ),
  asd: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
      <path d="M15.54 8.46a5 5 0 010 7.07" />
      <path d="M19.07 4.93a10 10 0 010 14.14" />
    </svg>
  ),
  export: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  ),
  quality: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
      <polyline points="9 12 11 14 15 10" />
    </svg>
  ),
};

/** The short name + the description under each "machine". */
// Short labels shown under each "machine". They mirror the backend services
// (backend/services/*); segmentation covers both the legacy VAD strategy and
// the v3 sentence strategy (full-video Whisper guiding the cut points).
const STAGE_LABELS = {
  download: { name: 'Download',          desc: 'downloader · yt-dlp · CC' },
  vad:      { name: 'Segmentation',      desc: 'segmenter · VAD + sentences' },
  face:     { name: 'Face tracking',     desc: 'face_tracker · RetinaFace' },
  asd:      { name: 'Speaker detection', desc: 'speaker_detector · ASD + SyncNet' },
  asr:      { name: 'Transcription',     desc: 'segmenter · WhisperX RO' },
  export:   { name: 'LRS2 Export',       desc: 'mouth_exporter · crops' },
  quality:  { name: 'Speaker ID & Quality', desc: 'quality_indexer · clustering' },
};

/** The four stages that cycle per clip and share the CLIP N/M progress. */
const PER_CLIP_STAGES = new Set(['face', 'asd', 'asr', 'export']);

/**
 * Computes the (CSS class, displayed text) for a stage, based on the state
 * derived from the log — same priority logic as in the original UI:
 * error > done > active > waiting.
 */
function stageDisplay(stage, parsed, isRunning) {
  if (parsed.errorStages.has(stage)) {
    return { cls: 'error', readout: '✗ Failed' };
  }
  if (parsed.doneStages.has(stage)) {
    if (stage === 'export' && parsed.exportedSegments > 0) {
      return { cls: 'done', readout: `✓ ${parsed.exportedSegments} segs` };
    }
    if (stage === 'quality' && parsed.speakersIdentified > 0) {
      return { cls: 'done', readout: `✓ ${parsed.speakersIdentified} spk` };
    }
    if (PER_CLIP_STAGES.has(stage) && parsed.clipProgress.processed > 0) {
      return { cls: 'done', readout: `✓ ${parsed.clipProgress.processed} ok` };
    }
    return { cls: 'done', readout: '✓ Done' };
  }
  // The four per-clip stages share ONE clip counter, so the count/percent
  // readout stays visible on ALL of them while clips are being processed —
  // the highlight alone marks which one is active right now.
  if (PER_CLIP_STAGES.has(stage) && isRunning && parsed.clipProgress.total > 0) {
    const pct = (parsed.clipProgress.processed / parsed.clipProgress.total) * 100;
    return {
      cls: stage === parsed.currentStage ? 'active' : '',
      readout: `${parsed.clipProgress.processed}/${parsed.clipProgress.total} · ${pct.toFixed(0)}%`,
    };
  }
  if (stage === parsed.currentStage && isRunning) {
    return { cls: 'active', readout: 'Running…' };
  }
  // Not started yet: "Queued" while a run is in progress, "Idle" otherwise.
  return { cls: '', readout: isRunning ? 'Queued' : 'Idle' };
}

/**
 * The column with the pipeline's 5 "machines" (display only — all
 * stages always run, in order).
 */
export function PipelineStages({ parsed, isRunning }) {
  return (
    <div className="machine-section">
      <div className="machine-section-hdr">
        <span className="machine-label">Pipeline stages</span>
      </div>
      <div className="machines">
        {STAGES.map((stage, index) => {
          const { cls, readout } = stageDisplay(stage, parsed, isRunning);
          return (
            <div key={stage}>
              {index > 0 && <div className="machine-pipe"></div>}
              <div className={`machine machine-display ${cls}`}>
                <div className="machine-body">
                  <div className="machine-head">
                    <div className="machine-led"></div>
                    <div className="machine-icon">{STAGE_ICONS[stage]}</div>
                    <div className="machine-info">
                      <span className="machine-name">{STAGE_LABELS[stage].name}</span>
                      <span className="machine-desc">{STAGE_LABELS[stage].desc}</span>
                    </div>
                  </div>
                  <div className="machine-readout">{readout}</div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
