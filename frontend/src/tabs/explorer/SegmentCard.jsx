import { useRef, useEffect } from 'react';
import { api } from '../../api.js';
import {
  formatDuration, formatMetric,
  asdColorClass, syncColorClass, whisperColorClass,
} from '../../utils/format.js';
import { normalizeTier } from '../../utils/datasetCapabilities.js';

/**
 * A card in the Explorer gallery: video (lazily loaded), review status,
 * transcript and the three quality scores.
 *
 * Efficiency: the video's `src` is set only when the card enters the
 * viewport (IntersectionObserver) — otherwise every page change would
 * fire ~45 video metadata requests at once.
 */
export function SegmentCard({ segment, status, crop, observer, onOpen }) {
  const videoRef = useRef(null);
  const mediaUrl = api.mediaUrl(segment.video_id || '', segment.segment_id || '', crop);

  // Register the video element with the observer; it sets src on intersection.
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !observer) return undefined;
    video.dataset.src = mediaUrl;
    // If the crop changed after src was already set, update it directly.
    if (video.src && !video.src.endsWith(mediaUrl)) {
      video.src = mediaUrl;
    }
    observer.observe(video);
    return () => observer.unobserve(video);
  }, [mediaUrl, observer]);

  const text = segment.text || '(no transcript)';
  const duration = segment.duration != null ? formatDuration(parseFloat(segment.duration)) : '—';
  const isEdited =
    segment.original_text && segment.text &&
    String(segment.original_text).trim() !== String(segment.text).trim();

  return (
    <div className="seg-card" onClick={() => onOpen(segment.segment_id)}>
      <div className="seg-card-media">
        <video
          ref={videoRef}
          muted
          loop
          playsInline
          preload="none"
          onMouseEnter={(e) => {
            const v = e.currentTarget;
            if (!v.src && v.dataset.src) v.src = v.dataset.src;
            v.play().catch(() => {});
          }}
          onMouseLeave={(e) => e.currentTarget.pause()}
        />
        <span className={`seg-status ${status}`} title={status}>{status}</span>
        {/* v3 quality tier badge (bottom-left, mirroring the duration badge) */}
        {normalizeTier(segment.quality_tier) && (
          <span
            className={`seg-tier tier-${normalizeTier(segment.quality_tier).toLowerCase()}`}
            title={`Quality tier ${normalizeTier(segment.quality_tier)}`}
          >
            {normalizeTier(segment.quality_tier)}
          </span>
        )}
        {isEdited && <span className="seg-edited" title="Transcript edited">✎</span>}
        <span className="seg-dur">{duration}</span>
      </div>
      <div className="seg-card-body">
        <div className="seg-text" title={text}>{text}</div>
        <div className="seg-meta">
          <span className="seg-meta-item">
            <span className="seg-meta-k">ASD</span>{' '}
            <span className={`seg-meta-v ${asdColorClass(segment.asd_score)}`}>
              {formatMetric(segment.asd_score, 1)}
            </span>
          </span>
          <span className="seg-meta-item">
            <span className="seg-meta-k">Sync</span>
            <span className={`seg-meta-v ${syncColorClass(segment.syncnet_conf)}`}>
              {formatMetric(segment.syncnet_conf, 2)}
            </span>
          </span>
          <span className="seg-meta-item">
            <span className="seg-meta-k">Wh</span>{' '}
            <span className={`seg-meta-v ${whisperColorClass(segment.whisper_conf)}`}>
              {formatMetric(segment.whisper_conf, 2)}
            </span>
          </span>
        </div>
      </div>
    </div>
  );
}
