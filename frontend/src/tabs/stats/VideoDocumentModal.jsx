import { useEffect, useState } from 'react';
import { api } from '../../api.js';
import { DocumentTree } from '../../components/DocumentTree.jsx';

/**
 * A video's MongoDB document: source{}, metadata{}, pipeline{}, stats{} and
 * the embedded dropped_clips[] (every rejected clip with its reason).
 * Closes on Escape or on a click on the backdrop.
 */
export function VideoDocumentModal({ videoId, onClose }) {
  const [doc, setDoc] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setDoc(null);
    setError(null);
    api.getVideoDocument(videoId)
      .then((data) => { if (!cancelled) setDoc(data); })
      .catch((err) => { if (!cancelled) setError(err.message || 'Failed to load'); });
    return () => { cancelled = true; };
  }, [videoId]);

  useEffect(() => {
    const handleKey = (e) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [onClose]);

  const dropped = doc?.dropped_clips || [];
  const reasons = dropped.reduce((acc, d) => {
    const code = d.reason?.code || 'unknown';
    acc[code] = (acc[code] || 0) + 1;
    return acc;
  }, {});

  return (
    <div className="gallery-modal" onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="gallery-modal-card" role="dialog" aria-modal="true" aria-label={`Video ${videoId}`}>
        <div className="gallery-modal-hdr">
          <span className="cmd-panel-label">{videoId}</span>
          <button className="btn-micro" onClick={onClose}>Close</button>
        </div>
        <div className="gallery-modal-body">
          {!doc && !error && <div className="explorer-placeholder"><p className="placeholder-title">Loading...</p></div>}
          {error && <div className="explorer-placeholder"><p className="placeholder-title">{error}</p></div>}
          {doc && (
            <>
              {doc.title && <div className="detail-section"><div className="detail-transcript">{doc.title}</div></div>}
              {dropped.length > 0 && (
                <div className="detail-section">
                  <h4>Rejected clips ({dropped.length})</h4>
                  <div className="detail-info-grid">
                    {Object.entries(reasons).sort((a, b) => b[1] - a[1]).map(([code, n]) => (
                      <div className="detail-info-item" key={code}>
                        <span className="detail-info-label">{code}</span>
                        <span className="detail-info-value">{n}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              <div className="detail-section">
                <h4>MongoDB document</h4>
                <DocumentTree document={doc} />
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
