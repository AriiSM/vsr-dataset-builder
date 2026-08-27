import { useState, useEffect, useCallback } from 'react';
import { api } from '../../api.js';
import { toast } from '../../components/toast.jsx';

// Module-level cache: video_id → the list of speakers for that video.
// Avoids re-downloading the whole registry for every segment of the same video.
const speakersCache = new Map();

async function fetchSpeakersForVideo(videoId) {
  if (speakersCache.has(videoId)) return speakersCache.get(videoId);
  try {
    const all = await api.getSpeakers();
    const list = (all.speakers || []).filter(
      (s) => s.speaker_id && s.speaker_id.startsWith(`${videoId}_`)
    );
    speakersCache.set(videoId, list);
    return list;
  } catch {
    return [];
  }
}

/**
 * The speaker dropdown in the Review panel: pick an existing speaker of the
 * video or create a new one ("+ New speaker...", with a suggested index).
 */
export function SpeakerSelect({ segmentId, videoId, currentSpeakerId, onChanged }) {
  const [speakers, setSpeakers] = useState(null); // null = still loading
  const [selected, setSelected] = useState(currentSpeakerId || '');

  // (Re)load the list when the video or the current speaker changes.
  const reload = useCallback(async () => {
    setSpeakers(await fetchSpeakersForVideo(videoId));
  }, [videoId]);

  useEffect(() => {
    setSelected(currentSpeakerId || '');
    reload();
  }, [segmentId, currentSpeakerId, reload]);

  async function handleChange(event) {
    const previous = selected;
    let newId = event.target.value;

    if (newId === '__NEW__') {
      // Create the speaker directly at the next free index for this video
      // (video_spkN) — no blocking prompt; merging into an existing speaker
      // is done by simply picking it from the list instead.
      const existing = await fetchSpeakersForVideo(videoId);
      const usedNumbers = existing
        .map((s) => s.speaker_id.match(new RegExp(`^${videoId}_spk(\\d+)$`)))
        .filter(Boolean)
        .map((m) => parseInt(m[1], 10));
      const nextIndex = usedNumbers.length ? Math.max(...usedNumbers) + 1 : 0;
      newId = `${videoId}_spk${nextIndex}`;
    }

    if (newId === previous) return;

    try {
      const { ok, data } = await api.setSegmentSpeaker(segmentId, newId);
      if (!ok) {
        toast.error(`Speaker change failed: ${data.error || 'unknown error'}`);
        return;
      }
      setSelected(newId);
      toast.success(`Speaker set to ${newId}`);
      speakersCache.delete(videoId); // the aggregates have changed
      await reload();
      onChanged?.(newId);
    } catch (err) {
      toast.error(`Speaker change error: ${err}`);
    }
  }

  if (speakers === null) {
    return <span className="dim">loading…</span>;
  }

  const knownIds = new Set(speakers.map((s) => s.speaker_id));

  return (
    <select className="rv-speaker-select" value={selected || ''} onChange={handleChange}>
      {/* The current speaker may be missing from the registry if it was just
          created and the aggregates haven't refreshed — show it explicitly. */}
      {selected && !knownIds.has(selected) && (
        <option value={selected}>{selected}</option>
      )}
      {!selected && <option value="" disabled>(none)</option>}
      {speakers.map((s) => (
        <option key={s.speaker_id} value={s.speaker_id}>
          {s.speaker_name ? `${s.speaker_id} — ${s.speaker_name}` : s.speaker_id}
        </option>
      ))}
      <option value="__NEW__">+ New speaker...</option>
    </select>
  );
}
