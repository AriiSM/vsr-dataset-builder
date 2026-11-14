import { useRef, useLayoutEffect } from 'react';
import { STAGE_TAGS, STAGE_META } from '../../utils/logParser.js';

/**
 * Rewrites the backend's terse log lines into unambiguous wording for the
 * activity feed. The terminology it enforces:
 *
 *   - "source video"  = the big raw YouTube video being processed
 *   - "clip"          = a candidate piece cut out of the source video
 *   - "segment"       = a clip that passed all filters and was exported
 *                       into the dataset
 *
 * The backend's own "CLIP x/y" reads ambiguously (it sounds like the big
 * video), so the feed spells it out. Display-only: the untouched machine
 * output is still saved to logs/sessions/ on disk, and the parser keeps
 * matching the original messages.
 */
const REWRITE_RULES = [
  {
    pattern: /^Processing video\s+(\d+)\/(\d+):\s*(\S+)/i,
    replace: 'Source video $1 of $2 — $3',
  },
  {
    pattern: /^Step 1: Downloading video:\s*(\S+)/i,
    replace: 'Downloading source video $1',
  },
  {
    pattern: /^Step 2: Sentence segmentation.*/i,
    replace: 'Cutting the source video into clips (VAD + full-video Whisper, sentence windows)',
  },
  {
    pattern: /^Step 2: Splitting video by VAD.*/i,
    replace: 'Cutting the source video into clips (VAD silence detection)',
  },
  {
    pattern: /^Step 3: Processing clips\.\.\./i,
    replace: 'Processing each clip (face → speaker → transcript → export)',
  },
  {
    pattern: /^(\d+) clips to process$/i,
    replace: '$1 clips cut from this source video — filtering each one now',
  },
  {
    pattern: /^CLIP\s+(\d+)\/(\d+)\s+exported:?\s*(.*)/i,
    replace: 'clip $1/$2 ✓ exported as dataset segment $3',
  },
  {
    pattern: /^CLIP\s+(\d+)\/(\d+)\s+dropped\s*(.*)/i,
    replace: 'clip $1/$2 ✗ dropped $3— not a dataset segment',
  },
  {
    pattern: /^Exported\s+(\d+)\s+segments?\s*(.*)/i,
    replace: 'Source video done — $1 dataset segments exported $2',
  },
];

/** Applies the first matching rewrite rule; unknown lines pass through. */
function clarifyMessage(msg) {
  for (const rule of REWRITE_RULES) {
    if (rule.pattern.test(msg)) {
      return msg.replace(rule.pattern, rule.replace);
    }
  }
  return msg;
}

/** True for lines that mark the start of a new source video in a batch. */
function isSourceVideoBoundary(msg) {
  return /^Processing video\s+\d+\/\d+:/i.test(msg);
}

/** Highlights video ids (md_075) and progress (42/259) in the message. */
function highlightMessage(msg) {
  // Split the text on the two patterns and render the matches as <strong>.
  // The split regex needs /g, but the membership test must NOT use it:
  // a global regex keeps `lastIndex` between .test() calls, which would
  // silently skip every other match.
  const splitPattern = /(\b[a-z]+_\d{3,}\b|\d+\s*\/\s*\d+)/g;
  const testPattern = /^(\b[a-z]+_\d{3,}\b|\d+\s*\/\s*\d+)$/;
  return msg.split(splitPattern).map((part, i) =>
    testPattern.test(part) ? <strong key={i}>{part}</strong> : part
  );
}

/**
 * The activity feed: the latest events from the log, colored by type.
 *
 * "Polite" auto-scroll: it stays pinned to the bottom ONLY if the user
 * was already there — if they scrolled up to read, we don't drag them
 * back down on every update.
 */
export function ActivityFeed({ entries }) {
  const containerRef = useRef(null);
  const wasAtBottomRef = useRef(true);

  // On every scroll, remember whether the user is near the bottom (48px).
  const handleScroll = () => {
    const el = containerRef.current;
    if (!el) return;
    wasAtBottomRef.current =
      el.scrollHeight - el.scrollTop - el.clientHeight <= 48;
  };

  // After every content update, scroll only if we were at the bottom.
  useLayoutEffect(() => {
    const el = containerRef.current;
    if (el && wasAtBottomRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [entries]);

  const visible = entries.slice(-250);

  return (
    <div className="control-activity" ref={containerRef} onScroll={handleScroll}>
      {visible.length === 0 ? (
        <div className="control-activity-empty">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" opacity="0.3">
            <circle cx="12" cy="12" r="10" />
            <polyline points="12 6 12 12 16 14" />
          </svg>
          <span>Waiting for output…</span>
        </div>
      ) : (
        visible.map((entry, i) => {
          const lineClass =
            entry.type === 'error' ? ' error-line' :
            entry.type === 'success' ? ' success-line' : '';
          const tag = entry.stage ? STAGE_TAGS[entry.stage] : null;
          // New-source-video lines get separator styling so each raw video's
          // section is visually distinct in the continuous feed.
          const boundaryClass = isSourceVideoBoundary(entry.msg) ? ' video-boundary' : '';
          return (
            <div className={`factory-activity-item${lineClass}${boundaryClass}`} key={i}>
              <span className="factory-activity-ts">{entry.ts || ''}</span>
              {/* Colored stage tag — the feed is one continuous stream, so
                  the tag tells at a glance which stage each line came from. */}
              <span
                className="factory-activity-stage"
                style={tag ? { color: tag.color, borderColor: tag.color } : { opacity: 0.35 }}
                title={
                  entry.stage
                    ? `${STAGE_META[entry.stage].name} — ${STAGE_META[entry.stage].desc}`
                    : 'System message (batch boundary / wrapper)'
                }
              >
                {tag ? tag.label : '·'}
              </span>
              <span className={`factory-activity-dot ${entry.type || 'info'}`}></span>
              <span className="factory-activity-msg">{highlightMessage(clarifyMessage(entry.msg))}</span>
            </div>
          );
        })
      )}
    </div>
  );
}
