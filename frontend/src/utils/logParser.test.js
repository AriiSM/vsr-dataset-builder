import { describe, it, expect } from 'vitest';
import {
  parseLogState,
  parseBulkImportState,
  classifyLogLine,
} from './logParser.js';

/**
 * These tests pin the parser to the EXACT log messages emitted by
 * backend/orchestrator/pipeline.py and cli.py. If the backend wording
 * changes (e.g. the v3 sentence-segmentation rewrite), these tests fail
 * loudly instead of the UI silently showing wrong stage states.
 */

/** Helper: builds a loguru-formatted line like the backend emits. */
const log = (msg) => `12:34:56 | INFO | ${msg}`;

describe('parseLogState — stage transitions', () => {
  it('marks download active on Step 1', () => {
    const state = parseLogState([log('Step 1: Downloading video: md_001')]);
    expect(state.currentStage).toBe('download');
    expect(state.currentVideo).toBe('md_001');
  });

  it('moves to vad on Step 2 and marks download done', () => {
    const state = parseLogState([
      log('Step 1: Downloading video: md_001'),
      log('Step 2: Splitting video by VAD...'),
    ]);
    expect(state.currentStage).toBe('vad');
    expect(state.doneStages.has('download')).toBe(true);
  });

  it('moves to face on Step 3', () => {
    const state = parseLogState([
      log('Step 1: Downloading video: md_001'),
      log('Step 2: Splitting video by VAD...'),
      log('Step 3: Processing clips...'),
    ]);
    expect(state.currentStage).toBe('face');
    expect(state.doneStages.has('vad')).toBe(true);
  });

  it('detects the speaker-detection sub-stage even with the track count suffix', () => {
    // Real message: "    [clip_003] ASD scoring (3/5 tracks)..."
    const state = parseLogState([
      log('Step 3: Processing clips...'),
      log('    [clip_003] ASD scoring (3/5 tracks)...'),
    ]);
    expect(state.currentStage).toBe('asd');
  });

  it('routes per-clip markers to their service stages', () => {
    const base = [log('Step 3: Processing clips...')];
    expect(parseLogState([...base, log('    [clip_1] face detection...')]).currentStage).toBe('face');
    expect(parseLogState([...base, log('    [clip_1] transcribing...')]).currentStage).toBe('asr');
    // ASD and SyncNet both belong to the speaker_detector service.
    expect(parseLogState([...base, log('    [clip_1] SyncNet verification...')]).currentStage).toBe('asd');
    expect(parseLogState([...base, log('    [clip_1] exporting LRS2 format...')]).currentStage).toBe('export');
  });

  it('routes quality_indexer output to the quality stage', () => {
    const state = parseLogState([
      log('Exported 29 segments (163.4s total)'),
      log('Speaker identity [md_101]: 29 segments → 2 speakers (1 matched cross-video)'),
    ]);
    expect(state.currentStage).toBe('quality');
    expect(state.speakersIdentified).toBe(2);
  });
});

describe('parseLogState — counters and totals', () => {
  it('tracks clip progress from CLIP N/M lines', () => {
    const state = parseLogState([
      log('Step 3: Processing clips...'),
      log('  42 clips to process'),
      log('  CLIP 7/42 exported: md_001_0007'),
    ]);
    expect(state.clipProgress.processed).toBe(7);
    expect(state.clipProgress.total).toBe(42);
    expect(state.exportedSegments).toBe(1);
  });

  it('does not double-count the export summary after CLIP lines', () => {
    const state = parseLogState([
      log('Step 3: Processing clips...'),
      log('  CLIP 1/2 exported: a'),
      log('  CLIP 2/2 exported: b'),
      log('Exported 2 segments (5 dropped)'),
    ]);
    expect(state.exportedSegments).toBe(2);
    expect(state.doneStages.has('export')).toBe(true);
  });

  it('uses the summary count when no CLIP lines were seen', () => {
    const state = parseLogState([log('Exported 9 segments (0 dropped)')]);
    expect(state.exportedSegments).toBe(9);
  });
});

describe('parseLogState — batch and error handling', () => {
  it('resets per-video state at each "Processing video N/M" boundary', () => {
    const state = parseLogState([
      log('Processing video 1/3: md_001'),
      log('Step 1: Downloading video: md_001'),
      log('Exported 4 segments (0 dropped)'),
      log('Processing video 2/3: md_002'),
      log('Step 1: Downloading video: md_002'),
    ]);
    expect(state.videoPos).toBe(2);
    expect(state.videoTotal).toBe(3);
    expect(state.currentVideo).toBe('md_002');
    // The new video's stages start fresh...
    expect(state.doneStages.has('export')).toBe(false);
    // ...but cross-video totals survive.
    expect(state.finishedVideos).toBe(1);
  });

  it('counts a failed video exactly once, via the strict pattern', () => {
    const failLine = log('Pipeline failed for md_007: CUDA out of memory');
    // The whole log is re-parsed on every poll, so the same line repeats.
    const state = parseLogState([failLine, failLine]);
    expect(state.failedVideos).toBe(1);
  });

  it('does NOT treat per-clip noise as a video failure', () => {
    const state = parseLogState([
      log('Step 3: Processing clips...'),
      log('Whisper failed on clip_004: timeout'),
    ]);
    expect(state.failedVideos).toBe(0);
  });

  it('distinguishes clean finish from finish-with-errors', () => {
    const clean = parseLogState(['[12:00:00] Pipeline finished']);
    expect(clean.doneStages.size).toBe(7);
    const withErrors = parseLogState(['[12:00:00] Pipeline finished with errors']);
    expect(withErrors.doneStages.size).toBe(0);
    expect(withErrors.errorStages.size).toBeGreaterThan(0);
  });
});

describe('parseLogState — chronological feed (allEntries)', () => {
  it('keeps every line in order with stage tags, across video boundaries', () => {
    const state = parseLogState([
      log('Processing video 1/2: md_001'),
      log('Step 1: Downloading video: md_001'),
      log('Step 2: Splitting video by VAD...'),
      log('Processing video 2/2: md_002'),
      log('Step 1: Downloading video: md_002'),
    ]);
    // The per-video reset must NOT wipe the chronological feed.
    expect(state.allEntries).toHaveLength(5);
    expect(state.allEntries.map((e) => e.stage)).toEqual([
      null, 'download', 'vad', null, 'download',
    ]);
  });

  it('tags per-clip lines with their sub-stage', () => {
    const state = parseLogState([
      log('Step 3: Processing clips...'),
      log('    [clip_1] face detection...'),
      log('    [clip_1] transcribing...'),
      log('    [clip_1] exporting LRS2 format...'),
    ]);
    expect(state.allEntries.map((e) => e.stage)).toEqual([
      'face', 'face', 'asr', 'export',
    ]);
  });

  it('tags speaker-detector lines with the asd stage', () => {
    const state = parseLogState([
      log('Step 3: Processing clips...'),
      log('    [clip_1] ASD scoring (1/2 tracks)...'),
      log('    [clip_1] SyncNet verification...'),
    ]);
    expect(state.allEntries.map((e) => e.stage)).toEqual(['face', 'asd', 'asd']);
  });
});

describe('parseBulkImportState', () => {
  it('tracks per-URL verdicts without double counting', () => {
    const lines = [
      log('[1/3] Importing: https://youtu.be/aaa'),
      log('  md_075 downloaded OK → marked pending'),
      log('[2/3] Importing: https://youtu.be/bbb'),
      log('  Already in CSV (same YouTube ID bbb123): skipped'),
      log('[3/3] Importing: https://youtu.be/ccc'),
      log('  md_076 download error: HTTP 403'),
      // Polling re-parses everything — duplicates must not inflate counts.
      log('  md_075 downloaded OK → marked pending'),
    ];
    const bulk = parseBulkImportState(lines);
    expect(bulk).toMatchObject({ current: 3, total: 3, ok: 1, skipped: 1, failed: 1, decided: 3 });
  });
});

describe('classifyLogLine', () => {
  it('classifies errors, warnings and successes', () => {
    expect(classifyLogLine('Traceback (most recent call last)')).toBe('error');
    expect(classifyLogLine('CLIP 3/10 dropped (no face)')).toBe('warn');
    expect(classifyLogLine('Exported 5 segments')).toBe('success');
    expect(classifyLogLine('42 clips to process')).toBe('info');
  });
});
