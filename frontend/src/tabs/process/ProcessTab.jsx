import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { api } from '../../api.js';
import { toast } from '../../components/toast.jsx';
import { STAGES, parseLogState } from '../../utils/logParser.js';
import { ModeSelector } from './ModeSelector.jsx';
import { PipelineStages } from './PipelineStages.jsx';
import { ControlRoom } from './ControlRoom.jsx';
import { DEMO_LOG } from '../../demoLog.js';

/** How often we poll /api/status while the pipeline is running (ms). */
const STATUS_POLL_INTERVAL_MS = 1200;

/** Initial form values (all modes in one place). */
const INITIAL_FORM = {
  // batch / resume
  limit: '',
  // resume: retry failed videos from scratch instead of resuming interrupted
  retryFailed: false,
  // bulk-import
  bulkPrefix: 'md',
  bulkRegion: 'MD',  // the dataset target is MD-only, so MD is the default
  bulkSource: 'YouTube_CC',
  bulkUrls: '',
  bulkNoCcCheck: false,
  // pre-downloaded corpus: lines are "md_001 <url>"; raw files already in
  // data/raw — the backend maps them (metadata fetch only, no download).
  bulkPreDownloaded: false,
  // cookies (shared)
  cookies: '',
  cookiesBrowser: '',
};

/**
 * Builds the request body and endpoint for /api/start or
 * /api/bulk_import, based on the mode and the form. Returns
 * { error } if validation fails.
 */
function buildStartRequest(mode, form, selectedIds) {
  const cookies = form.cookies.trim();
  const browser = form.cookiesBrowser;

  if (mode === 'bulk-import') {
    const urls = form.bulkUrls
      .split(/[\r\n]+/)
      .map((s) => s.trim())
      .filter((s) => s && !s.startsWith('#'));
    if (!urls.length) {
      return {
        error: form.bulkPreDownloaded
          ? 'Paste at least one line: md_001 https://...'
          : 'Paste at least one YouTube URL.',
      };
    }
    if (form.bulkPreDownloaded) {
      // Client-side strict check so a bad line is caught before the job.
      const bad = urls.find(
        (u) => !(u.split(/\s+/).length === 2 && u.split(/\s+/)[1].includes('://')),
      );
      if (bad) return { error: `Expected "md_001 https://..." — got: ${bad}` };
    }
    return {
      isBulk: true,
      body: {
        urls,
        prefix: form.bulkPrefix.trim() || 'vid',
        region: form.bulkRegion,
        source: form.bulkSource,
        no_cc_check: form.bulkNoCcCheck,
        pre_downloaded: form.bulkPreDownloaded,
        cookies: cookies || undefined,
        cookies_from_browser: browser || undefined,
      },
    };
  }

  // "Resume + retry failed" maps to the backend's batch --status failed;
  // plain Resume maps to resume-batch (continue interrupted videos).
  const effectiveMode =
    mode === 'resume' && form.retryFailed ? 'batch-failed' : mode;

  const body = { mode: effectiveMode, cookies_from_browser: browser || undefined };
  if (cookies) body.cookies = cookies;
  const limit = form.limit.trim();
  if (limit) body.limit = parseInt(limit);

  // The picker's selection; empty selection = all eligible videos.
  if (selectedIds.size > 0) body.video_ids = [...selectedIds];

  return { isBulk: false, body };
}

/**
 * The Process tab: starts/stops the pipeline and tracks progress live.
 *
 * While the pipeline is running, we poll /api/status every 1.2s; the
 * visual state (stages, counters, activity feed) is derived from the log
 * with `parseLogState` — a pure function, so easy to follow and test.
 */
export function ProcessTab({ isActive, onPipelineRunning }) {
  // ── The form on the left ──
  const [mode, setMode] = useState('batch-pending');
  const [form, setForm] = useState(INITIAL_FORM);

  // ── The video registry, for the "Videos to process" picker ──
  const [registryVideos, setRegistryVideos] = useState([]);
  const [selectedIds, setSelectedIds] = useState(new Set());

  const fetchRegistry = useCallback(async () => {
    try {
      const res = await api.getVideos();
      setRegistryVideos(res.videos || []);
    } catch {
      setRegistryVideos([]); // backend not reachable — the picker shows empty
    }
  }, []);

  // Load the registry when the tab is shown; refresh it after every run
  // (statuses change as videos finish).
  useEffect(() => {
    if (isActive) fetchRegistry();
  }, [isActive, fetchRegistry]);

  // The videos eligible for the current mode, in registry order:
  //   Batch  → not yet processed (pending)
  //   Resume → interrupted (processing) + failed; retry-from-scratch → failed only
  const pickerVideos = useMemo(() => {
    if (mode === 'batch-pending') {
      return registryVideos.filter((v) => v.status === 'pending');
    }
    if (mode === 'resume') {
      return form.retryFailed
        ? registryVideos.filter((v) => v.status === 'failed')
        : registryVideos.filter((v) => v.status === 'processing' || v.status === 'failed');
    }
    return [];
  }, [registryVideos, mode, form.retryFailed]);

  // A selection made for one mode makes no sense in another — reset it.
  useEffect(() => {
    setSelectedIds(new Set());
  }, [mode, form.retryFailed]);

  // ── Pipeline state ──
  const [isRunning, setIsRunning] = useState(false);
  const [logLines, setLogLines] = useState([]);
  const [serverMode, setServerMode] = useState(null);  // mode reported by the backend
  const [controlState, setControlState] = useState('standby'); // standby|running|done|error
  const [finishedTitle, setFinishedTitle] = useState(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  // Start time (ms) — used only by the timer, so it lives in a ref.
  const startTimeRef = useRef(null);
  // Becomes true only once we've seen the server report running=true; avoids
  // the race where the first poll arrives before the thread has started.
  const sawRunningRef = useRef(false);
  // true while the polling loop should keep going.
  const [isPolling, setIsPolling] = useState(false);

  // State derived from the log — recomputed only when the log changes.
  const parsed = useMemo(() => parseLogState(logLines), [logLines]);

  /** Marks the end of the run: sets the title + LED color. */
  const finishRun = useCallback((finalLog) => {
    // An empty log on "finish" means the backend lost its state (it was
    // restarted mid-run) — report that instead of a bogus "Complete".
    if (finalLog.length === 0) {
      setControlState('standby');
      setFinishedTitle(null);
      setIsPolling(false);
      setIsRunning(false);
      toast.info('Pipeline state was reset — the backend restarted.');
      return;
    }
    const finalParsed = parseLogState(finalLog);
    const hasErrors = finalParsed.errorStages.size > 0;
    const doneCount = STAGES.filter((s) => finalParsed.doneStages.has(s)).length;
    setControlState(hasErrors ? 'error' : 'done');
    setFinishedTitle(
      hasErrors
        ? `Done with errors — ${doneCount}/${STAGES.length} OK`
        : `Complete · ${finalParsed.exportedSegments} segments exported`
    );
    setIsPolling(false);
    setIsRunning(false);
    fetchRegistry(); // statuses changed — refresh the picker
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── The polling loop (active only while isPolling is true) ──
  useEffect(() => {
    if (!isPolling) return undefined;
    const timer = setInterval(async () => {
      try {
        const data = await api.getStatus();
        setLogLines(data.log || []);
        setServerMode(data.mode);
        setIsRunning(data.running);
        if (data.running) {
          sawRunningRef.current = true;
        } else if (sawRunningRef.current) {
          finishRun(data.log || []);
        }
      } catch (err) {
        console.error('Status poll failed', err);
      }
    }, STATUS_POLL_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [isPolling, finishRun]);

  // ── The timer (1s) — started together with the pipeline ──
  useEffect(() => {
    if (!isRunning) return undefined;
    const timer = setInterval(() => {
      if (startTimeRef.current) {
        setElapsedSeconds(Math.floor((Date.now() - startTimeRef.current) / 1000));
      }
    }, 1000);
    return () => clearInterval(timer);
  }, [isRunning]);

  // ── On mount: if the pipeline is already running (page refresh),
  //    resume polling and bring the user to the Process tab. ──
  useEffect(() => {
    if (new URLSearchParams(window.location.search).has('demo')) return undefined;
    let cancelled = false;
    (async () => {
      try {
        const data = await api.getStatus();
        if (cancelled || !data.running) return;
        startTimeRef.current = data.started_at
          ? new Date(data.started_at).getTime()
          : Date.now();
        sawRunningRef.current = true;
        setServerMode(data.mode);
        setLogLines(data.log || []);
        setIsRunning(true);
        setControlState('running');
        setIsPolling(true);
        onPipelineRunning?.();
      } catch {
        // backend unavailable on mount — not a problem, we stay in standby
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Demo mode (?demo=1): replays a synthetic log so the whole Process
  //    tab (stages, counters, color-tagged feed) can be previewed without
  //    running the real pipeline. Purely visual — remove the query param
  //    from the URL to return to normal operation. ──
  useEffect(() => {
    if (!new URLSearchParams(window.location.search).has('demo')) return undefined;
    startTimeRef.current = Date.now();
    setControlState('running');
    setIsRunning(true);
    setServerMode('batch-pending');
    toast.info('Demo mode: replaying a simulated pipeline log.');
    let lineCount = 0;
    const timer = setInterval(() => {
      lineCount += 1;
      setLogLines(DEMO_LOG.slice(0, lineCount));
      if (lineCount >= DEMO_LOG.length) {
        clearInterval(timer);
        finishRun(DEMO_LOG);
      }
    }, 350);
    return () => clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /** Starts the pipeline (or the bulk import) with the current settings. */
  async function handleStart() {
    // Guard: only send ids that are still eligible (a selected video may
    // have finished in a previous run and left the list).
    const eligibleSelection = new Set(
      [...selectedIds].filter((id) => pickerVideos.some((v) => v.video_id === id))
    );
    const request = buildStartRequest(mode, form, eligibleSelection);
    if (request.error) {
      toast.error(request.error);
      return;
    }

    // Reset the visual state before starting.
    startTimeRef.current = Date.now();
    setElapsedSeconds(0);
    setLogLines([]);
    setFinishedTitle(null);
    setControlState('running');
    setIsRunning(true);
    sawRunningRef.current = false;

    try {
      const { ok, data } = request.isBulk
        ? await api.startBulkImport(request.body)
        : await api.startPipeline(request.body);
      if (!ok) {
        toast.error(data.error || 'Failed to start');
        setControlState('standby');
        setIsRunning(false);
        return;
      }
      setServerMode(mode);
      setIsPolling(true);
    } catch (err) {
      toast.error(`Network error: ${err.message}`);
      setControlState('standby');
      setIsRunning(false);
    }
  }

  /** Stops the running pipeline (with confirmation). */
  function handleStop() {
    toast.confirm('Stop the running pipeline?', async () => {
      try {
        await api.stopPipeline();
      } catch (err) {
        toast.error(`Failed to stop: ${err.message}`);
      }
    });
  }

  const startLabel = mode === 'bulk-import' ? 'Start import' : 'Start pipeline';

  return (
    <main className={`tab-content ${isActive ? 'active' : ''}`} id="tab-process">
      <div className="factory">
        {/* ═══ Left: settings + pipeline stages ═══ */}
        <aside className="factory-left">
          <ModeSelector
            mode={mode}
            onModeChange={setMode}
            form={form}
            onFormChange={setForm}
            pickerVideos={pickerVideos}
            selectedIds={selectedIds}
            onSelectionChange={setSelectedIds}
          />
          <PipelineStages parsed={parsed} isRunning={isRunning} />

          {/* Sticky dock: the Start/Stop button stays visible even when the
              taller 7-stage column scrolls — critical for reaching Stop
              without scrolling during a run. */}
          <div className="factory-run-dock">
            {isRunning ? (
              <button
                className="btn-run-factory"
                style={{ background: 'linear-gradient(135deg,var(--red),#b91c1c)' }}
                onClick={handleStop}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                  <rect x="4" y="4" width="16" height="16" rx="2" />
                </svg>
                <span>Stop</span>
              </button>
            ) : (
              <button className="btn-run-factory" onClick={handleStart}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                  <polygon points="5 3 19 12 5 21 5 3" />
                </svg>
                <span>{startLabel}</span>
              </button>
            )}
          </div>
        </aside>

        {/* ═══ Right: the control room ═══ */}
        <ControlRoom
          controlState={controlState}
          elapsedSeconds={elapsedSeconds}
          parsed={parsed}
          logLines={logLines}
          mode={serverMode || mode}
          isRunning={isRunning}
          finishedTitle={finishedTitle}
        />
      </div>
    </main>
  );
}
