import { useState, useEffect, useCallback } from 'react';
import { api } from '../../api.js';

/**
 * Shared data loader for the Dashboard and Data tabs.
 *
 * Loads every stats source in parallel when the owning tab becomes active
 * (or when the refresh key changes); partial failures never block the rest.
 * Each tab owns its own instance — they fetch only while active, so there
 * is no duplicate traffic from the hidden tab.
 */
export function useStatsData(isActive, refreshKey) {
  const [stats, setStats] = useState({});
  const [videos, setVideos] = useState([]);
  const [distributions, setDistributions] = useState({});
  const [speakers, setSpeakers] = useState([]);
  const [vocabulary, setVocabulary] = useState({ words: [], total_unique: 0 });

  const loadAll = useCallback(async () => {
    try {
      const [statsRes, videosRes, distRes, speakersRes, vocabRes] = await Promise.all([
        api.getStats(),
        api.getVideos().catch(() => ({ videos: [] })),
        api.getDistributions().catch(() => ({})),
        api.getSpeakers().catch(() => ({ speakers: [] })),
        api.getVocabulary().catch(() => ({ words: [], total_unique: 0 })),
      ]);
      setStats(statsRes);
      setVideos(videosRes.videos || []);
      setDistributions(distRes);
      setSpeakers(speakersRes.speakers || []);
      setVocabulary(vocabRes);
    } catch (err) {
      console.error('Failed to load stats', err);
    }
  }, []);

  /** Reloads only the speakers (after an edit in the modal). */
  const reloadSpeakers = useCallback(async () => {
    try {
      const res = await api.getSpeakers();
      setSpeakers(res.speakers || []);
    } catch (err) {
      console.error('Failed to reload speakers', err);
    }
  }, []);

  useEffect(() => {
    if (isActive) loadAll();
  }, [isActive, refreshKey, loadAll]);

  return { stats, videos, distributions, speakers, vocabulary, reloadSpeakers };
}
