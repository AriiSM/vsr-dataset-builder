/**
 * The pipeline log parser — pure functions, no DOM.
 *
 * The backend sends the raw log (lines of text) via /api/status. From this
 * text we reconstruct the visual state of the "factory": which stage is
 * active, what has finished, clip counters, the current video in the batch, etc.
 *
 * Line formats:
 *   - loguru (pipeline):  "HH:MM:SS | INFO | message"
 *   - wrapper (app.py):   "[HH:MM:SS] message"
 *
 * Stage detection rules (based on the messages in src/pipeline.py):
 *   "Step 1:"                        → download active
 *   "Step 2:" / "Splitting by VAD"   → vad active
 *   "Step 3:" / "Processing clips"   → face+asd active
 *   "CLIP N/M dropped|exported"      → per-clip progress
 *   "Exported N segments"            → video finished
 *   "Pipeline failed for <id>:"      → video failed
 */

/**
 * The pipeline stages, in order — one per backend service that runs inside a
 * pipeline session (transcript_refiner is an offline tool, so it is not a
 * stage here). face/asd/asr/export cycle per clip; quality runs per video.
 */
export const STAGES = ['download', 'vad', 'face', 'asd', 'asr', 'export', 'quality'];

/** Displayed names and descriptions for each stage. */
// Names/descriptions mirror the backend microservices (backend/services/*).
// Segmentation covers both strategies: legacy (VAD silence cuts, Whisper per
// clip) and v3 sentence mode (full-video Whisper guides the cut points, so
// the per-clip Transcription stage is skipped — its words come pre-computed).
export const STAGE_META = {
  download: { name: 'Download',         desc: 'downloader — yt-dlp fetch (Creative Commons filter)' },
  vad:      { name: 'Segmentation',     desc: 'segmenter — Silero VAD + sentence windows (v3: full-video WhisperX guides the cuts)' },
  face:     { name: 'Face tracking',    desc: 'face_tracker — RetinaFace detection + Kalman tracking' },
  asd:      { name: 'Speaker detection', desc: 'speaker_detector — TalkNet active-speaker scoring + SyncNet A/V sync check' },
  asr:      { name: 'Transcription',    desc: 'segmenter — WhisperX Romanian per clip (legacy mode; v3 already has the words from segmentation)' },
  export:   { name: 'LRS2 Export',      desc: 'mouth_exporter — face 256×256 + mouth crop @ 25fps + annotation' },
  quality:  { name: 'Speaker ID & Quality', desc: 'quality_indexer — ArcFace speaker clustering, demographics, quality tiers' },
};

/** Extracts (timestamp, message) from a log line, regardless of format. */
function splitLogLine(rawLine) {
  const loguruMatch = rawLine.match(/^(\d{2}:\d{2}:\d{2})\s*\|\s*\w+\s*\|\s*(.*)/);
  if (loguruMatch) return { ts: loguruMatch[1], msg: loguruMatch[2] };
  const wrapperMatch = rawLine.match(/^\[(\d{2}:\d{2}:\d{2})\]\s*(.*)/);
  if (wrapperMatch) return { ts: wrapperMatch[1], msg: wrapperMatch[2] };
  return { ts: '', msg: rawLine };
}

/**
 * Colored tag shown next to each activity-feed line, identifying the stage
 * the line belongs to. Lines outside any stage (batch boundaries, wrapper
 * messages) get no tag.
 */
export const STAGE_TAGS = {
  download: { label: 'DL',   color: '#38bdf8' },
  vad:      { label: 'SEG',  color: '#fbbf24' },
  face:     { label: 'FACE', color: '#c084fc' },
  asd:      { label: 'SPK',  color: '#f472b6' },
  asr:      { label: 'ASR',  color: '#34d399' },
  export:   { label: 'EXP',  color: '#2dd4bf' },
  quality:  { label: 'QC',   color: '#a3e635' },
};

/** Classifies a log message for coloring in the activity feed. */
export function classifyLogLine(msg) {
  if (/error|fail|exception|traceback/i.test(msg)) return 'error';
  if (/skip|skipping|dropped/i.test(msg)) return 'warn';
  if (/done|completed|success|saved|exported|finished|wrote/i.test(msg)) return 'success';
  return 'info';
}

/**
 * Reconstructs the pipeline state from the entire log.
 *
 * Called on every polling tick over the WHOLE log (not incrementally),
 * which is why we use Sets keyed by video_id: the same video cannot be
 * counted twice even if its line appears multiple times.
 */
export function parseLogState(logLines) {
  const state = {
    currentStage: null,
    doneStages: new Set(),
    errorStages: new Set(),
    stageLogs: {},      // stage → [{ts, msg, type}]
    allEntries: [],     // ALL lines in order: [{ts, msg, type, stage|null}]
    counters: {},       // stage → {processed, total, errors}
    videos: [],         // all video_ids seen
    currentVideo: null, // the video being processed now
    videoPos: 0,        // N from "Processing video N/M"
    videoTotal: 0,      // M from "Processing video N/M"
    finishedVideoIds: new Set(),
    failedVideoIds: new Set(),
    exportedSegments: 0,
    clipProgress: { processed: 0, total: 0 },  // shared by face/asd/asr/export
    speakersIdentified: 0,  // total speakers found by quality_indexer
    finishedVideos: 0,  // numeric versions, populated at the end
    failedVideos: 0,
  };

  /** Records a line into the chronological feed, tagged with its stage. */
  function record(ts, msg, stage, type) {
    state.allEntries.push({ ts, msg, type: type || classifyLogLine(msg), stage });
  }

  /** Makes sure the structures for a stage exist. */
  function ensure(stage) {
    if (!state.stageLogs[stage]) state.stageLogs[stage] = [];
    if (!state.counters[stage]) state.counters[stage] = { processed: 0, total: 0, errors: 0 };
  }

  /**
   * Resets the per-video state when a new video starts, so stages don't
   * stay marked "done" from the previous one. Cross-video totals remain.
   */
  function resetPerVideo(newVideoId) {
    state.currentStage = null;
    state.doneStages = new Set();
    state.errorStages = new Set();
    state.stageLogs = {};
    state.counters = {};
    state.clipProgress = { processed: 0, total: 0 };
    state.currentVideo = newVideoId || null;
  }

  for (const rawLine of logLines) {
    const { ts, msg } = splitLogLine(rawLine);
    const m = msg.trim();

    // ── Batch boundary: "Processing video N/M: video_id" ──
    const batchMatch = m.match(/Processing video\s+(\d+)\/(\d+):\s*(\S+)/i);
    if (batchMatch) {
      state.videoPos = parseInt(batchMatch[1]);
      state.videoTotal = parseInt(batchMatch[2]);
      const videoId = batchMatch[3];
      if (!state.videos.includes(videoId)) state.videos.push(videoId);
      record(ts, m, null, 'info');
      resetPerVideo(videoId);
      continue;
    }

    // ── Video-level stage transitions ──
    if (/Step 1:/i.test(m)) {
      // A new video can also start without the "Processing video N/M" line
      // (single mode) — we reset so the old stages fade out.
      const videoMatch = m.match(/video:\s*(\S+)/i);
      const newVideoId = videoMatch ? videoMatch[1] : null;
      if (newVideoId && newVideoId !== state.currentVideo) {
        resetPerVideo(newVideoId);
        if (!state.videos.includes(newVideoId)) state.videos.push(newVideoId);
      }
      state.currentStage = 'download';
      ensure('download');
      record(ts, m, 'download');
      continue;
    }

    if (/Step 2:|Splitting video by VAD|existing clips found/i.test(m)) {
      if (state.currentStage === 'download') state.doneStages.add('download');
      state.currentStage = 'vad';
      ensure('vad');
      record(ts, m, 'vad');
      continue;
    }

    if (/Step 3:|Processing clips\.\.\./i.test(m)) {
      if (state.currentStage === 'vad') state.doneStages.add('vad');
      state.currentStage = 'face';
      ensure('face');
      ensure('asd');
      ensure('asr');
      ensure('export');
      record(ts, m, 'face');
      continue;
    }

    // ── Clip counters ──
    const clipsTotalMatch = m.match(/(\d+)\s+clips?\s+to\s+process/i);
    if (clipsTotalMatch) {
      state.clipProgress.total = parseInt(clipsTotalMatch[1]);
    }

    // Per-clip messages move the active sub-stage (face / asr / export).
    // Prefix matches only — the backend appends details after these markers,
    // e.g. "[clip_003] ASD scoring (3/5 tracks)...", so no trailing "...".
    if (/\]\s*face detection/i.test(m)) {
      state.currentStage = 'face';
      ensure('face');
    } else if (/\]\s*ASD scoring/i.test(m) || /\]\s*SyncNet verification/i.test(m)) {
      // TalkNet ASD and SyncNet both live in the speaker_detector service.
      state.currentStage = 'asd';
      ensure('asd');
    } else if (/\]\s*transcribing/i.test(m)) {
      state.currentStage = 'asr';
      ensure('asr');
    } else if (/\]\s*exporting LRS2/i.test(m)) {
      state.currentStage = 'export';
      ensure('export');
    } else if (/^Speaker identity \[|^Auto-set metadata for/i.test(m)) {
      // quality_indexer output (speaker clustering + demographics).
      state.currentStage = 'quality';
      ensure('quality');
      const spkMatch = m.match(/→\s*(\d+)\s+speakers?/i);
      if (spkMatch) state.speakersIdentified += parseInt(spkMatch[1]);
    }

    // "CLIP 42/259 dropped (...)" or "CLIP 42/259 exported: ..."
    const clipProgressMatch = m.match(/CLIP\s+(\d+)\/(\d+)\s+(dropped|exported)/i);
    if (clipProgressMatch) {
      state.clipProgress.processed = parseInt(clipProgressMatch[1]);
      state.clipProgress.total = parseInt(clipProgressMatch[2]);
      if (/exported/i.test(clipProgressMatch[3])) {
        state.exportedSegments += 1;
      }
    }

    // ── Export summary — don't double-count what the CLIP lines already counted ──
    const exportedMatch = m.match(/Exported\s+(\d+)\s+segments?/i);
    if (exportedMatch) {
      const total = parseInt(exportedMatch[1]);
      if (state.exportedSegments === 0) state.exportedSegments = total;
      state.doneStages.add('face');
      state.doneStages.add('asd');
      state.doneStages.add('asr');
      state.doneStages.add('export');
      if (state.currentVideo) state.finishedVideoIds.add(state.currentVideo);
    }

    // ── Errors — ONLY the strict "Pipeline failed for <id>:" pattern. Any
    //    looser regex would catch per-clip noise and inflate the count. ──
    const failMatch = m.match(/Pipeline failed for\s+(\S+?)\s*:/i);
    if (failMatch) {
      state.failedVideoIds.add(failMatch[1]);
      if (state.currentStage) {
        state.errorStages.add(state.currentStage);
      } else {
        state.errorStages.add('download');
        ensure('download');
        state.stageLogs.download.push({ ts, msg: m, type: 'error' });
      }
    }

    // ── Successful finish (must not match "finished with errors") ──
    if (/Pipeline finished(?!\s+with\s+errors)|Process exited with code 0/i.test(m)) {
      STAGES.forEach((s) => {
        if (!state.errorStages.has(s)) state.doneStages.add(s);
      });
    }

    if (/Process exited with code [^0]|Pipeline finished with errors/i.test(m)) {
      if (state.currentStage) {
        state.errorStages.add(state.currentStage);
      } else if (state.errorStages.size === 0) {
        state.errorStages.add('download');
      }
    }

    // ── Collect the line into the current stage's log ──
    const stage =
      state.currentStage ||
      (state.errorStages.size > 0 ? [...state.errorStages][0] : null);
    const type = classifyLogLine(m);
    // Every line lands in the chronological feed, stage-tagged when known.
    record(ts, m, stage, type);
    if (!stage) continue;
    ensure(stage);
    state.stageLogs[stage].push({ ts, msg: m, type });
  }

  state.finishedVideos = state.finishedVideoIds.size;
  state.failedVideos = state.failedVideoIds.size;
  return state;
}

/**
 * The progress of a bulk import, deduced from the log. There are no "stages"
 * here, just the verdict for each URL: ok / failed / skipped (already in CSV).
 */
export function parseBulkImportState(logLines) {
  let current = 0;
  let total = 0;
  // Sets keyed by video_id (or by the whole line when we have no id),
  // so the same event is not counted twice.
  const okIds = new Set();
  const failedIds = new Set();
  const skippedIds = new Set();

  for (const raw of logLines) {
    const importMatch = raw.match(/\[(\d+)\/(\d+)\]\s*Importing:\s*(\S+)/i);
    if (importMatch) {
      current = parseInt(importMatch[1]);
      total = parseInt(importMatch[2]);
      continue;
    }
    const okMatch = raw.match(/\b([a-z]+_\d+)\s+downloaded OK/i);
    if (okMatch) { okIds.add(okMatch[1]); continue; }

    const failMatch = raw.match(/\b([a-z]+_\d+)\s+download error/i);
    if (failMatch) { failedIds.add(failMatch[1]); continue; }

    if (/download returned (no path|None)/i.test(raw)) {
      failedIds.add(raw);
      continue;
    }
    const skipMatch = raw.match(/Already in CSV\s*\(same YouTube ID\s+(\S+?)\)\s*:\s*skipped/i);
    if (skipMatch) { skippedIds.add(skipMatch[1]); continue; }
    if (/Already in CSV.*?:\s*skipped/i.test(raw)) {
      skippedIds.add(raw);
    }
  }

  return {
    current,
    total,
    ok: okIds.size,
    failed: failedIds.size,
    skipped: skippedIds.size,
    decided: okIds.size + failedIds.size + skippedIds.size,
  };
}

/** Turns the bulk-import lines into entries for the activity feed. */
export function bulkImportFeedRows(logLines) {
  return logLines.map((raw) => {
    const { ts, msg } = splitLogLine(raw);
    let type = 'info';
    if (/error|failed|traceback|exception/i.test(msg)) type = 'error';
    else if (/skipped|already in csv/i.test(msg)) type = 'warn';
    else if (/downloaded ok|marked pending|CSV updated/i.test(msg)) type = 'success';
    return { ts, msg: msg.trim(), type };
  });
}
