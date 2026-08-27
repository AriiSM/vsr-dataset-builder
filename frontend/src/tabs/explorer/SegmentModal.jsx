import { useState, useEffect } from 'react';
import { api } from '../../api.js';
import { formatDuration, formatMetric } from '../../utils/format.js';
import { methodConfidenceRows } from '../../utils/methodInfo.jsx';

/**
 * The detail modal for a segment: transcript, metadata, word timeline,
 * face-cropped video and the raw annotation file.
 * Closes on Escape or on a click on the backdrop.
 */
export function SegmentModal({ segmentId, onClose }) {
  const [segment, setSegment] = useState(null);
  const [error, setError] = useState(null);

  // Load the segment details on open.
  useEffect(() => {
    let cancelled = false;
    setSegment(null);
    setError(null);
    api.getSegmentDetail(segmentId)
      .then((data) => { if (!cancelled) setSegment(data); })
      .catch((err) => { if (!cancelled) setError(err.message || 'Failed to load'); });
    return () => { cancelled = true; };
  }, [segmentId]);

  // Escape closes the modal.
  useEffect(() => {
    const handleKey = (e) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [onClose]);

  const annotation = segment?.annotation || {};

  const infoFields = segment
    ? [
        ['Segment',   segment.segment_id, 'mono'],
        ['Video',     segment.video_id, 'mono'],
        ['Duration',  segment.duration ? formatDuration(parseFloat(segment.duration)) : '—', ''],
        ['Words',     segment.num_words || '—', ''],
        ['ASD Score', formatMetric(segment.asd_score, 2), ''],
        ['SyncNet',   formatMetric(segment.syncnet_conf ?? segment.syncnet_confidence, 3), ''],
        ['Whisper ↑', formatMetric(segment.whisper_conf ?? segment.whisper_confidence, 3), ''],
        ['Conf',      annotation.conf != null ? `${annotation.conf}/3` : '—', ''],
      ]
    : [];

  return (
    <div
      className="gallery-modal"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="gallery-modal-card" role="dialog" aria-modal="true" aria-label={`Segment ${segmentId}`}>
        <div className="gallery-modal-hdr">
          <span className="cmd-panel-label">{segmentId}</span>
          <button className="btn-micro" onClick={onClose}>Close</button>
        </div>
        <div className="gallery-modal-body">
          {!segment && !error && (
            <div className="explorer-placeholder"><p className="placeholder-title">Loading...</p></div>
          )}
          {error && (
            <div className="explorer-placeholder"><p className="placeholder-title">{error}</p></div>
          )}
          {segment && (
            <>
              {/* Transcript */}
              <div className="detail-section">
                {annotation.text ? (
                  <div className="detail-transcript">{annotation.text}</div>
                ) : (
                  <div className="detail-transcript dim">No annotation text</div>
                )}
              </div>

              {/* Metadata grid */}
              <div className="detail-section">
                <div className="detail-info-grid">
                  {infoFields.map(([label, value, cls]) => (
                    <div className="detail-info-item" key={label}>
                      <span className="detail-info-label">{label}</span>
                      <span className={`detail-info-value${cls ? ` ${cls}` : ''}`}>{String(value)}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* v3 method & confidence metadata (only fields present) */}
              {methodConfidenceRows(segment).length > 0 && (
                <div className="detail-section">
                  <h4>Method &amp; Confidence</h4>
                  <div className="detail-info-grid">
                    {methodConfidenceRows(segment).map(({ label, value, warn }) => (
                      <div className="detail-info-item" key={label}>
                        <span className="detail-info-label">{label}</span>
                        <span className={`detail-info-value${warn ? ' value-warn' : ''}`}>
                          {value}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Word timeline */}
              {annotation.words?.length > 0 && (
                <div className="detail-section">
                  <h4>Word Timings</h4>
                  <div className="word-timeline">
                    {annotation.words.map((w, i) => (
                      <div className="word-chip" key={i}>
                        {w.word}
                        <em>{w.start.toFixed(2)}–{w.end.toFixed(2)}s</em>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Face-cropped video */}
              {segment.has_video && (
                <div className="detail-section">
                  <h4>Face-Cropped Video (256×256)</h4>
                  <div className="media-card">
                    <div className="media-card-label">{segment.segment_id}</div>
                    <video controls preload="metadata" loop>
                      <source src={api.mediaUrl(segment.video_id, segment.segment_id)} type="video/mp4" />
                    </video>
                  </div>
                </div>
              )}

              {/* Raw annotation file */}
              {segment.annotation_raw && (
                <div className="detail-section">
                  <h4>Annotation File</h4>
                  <pre
                    style={{
                      fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-dim)',
                      background: 'var(--surface)', border: '1px solid var(--border)',
                      borderRadius: 'var(--radius)', padding: 12, overflowX: 'auto', lineHeight: 1.7,
                    }}
                  >
                    {segment.annotation_raw}
                  </pre>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
