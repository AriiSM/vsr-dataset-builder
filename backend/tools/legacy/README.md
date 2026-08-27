# legacy/ — unelte din era CSV (storage v1), ÎNLOCUITE

Păstrate temporar doar pentru că repo-ul nu e încă sub git (ștergerea ar fi
ireversibilă). După pilotul pe mașina de procesare + git init, folderul se
șterge. NU le rula pe datele storage v2 — operează pe layout-ul vechi
(data/metadata/*.csv, data/annotations/).

| Unealtă | Înlocuită de |
|---|---|
| cluster_speakers.py | identitatea vorbitorilor la prima trecere (quality_indexer, DBSCAN per video + re-ID) |
| predict_speaker_metadata.py | sex/vârstă per CLUSTER în pipeline (aggregate_demographics) |
| backfill_syncnet.py | reprocesarea totală v3 (SyncNet rulează mereu) |
| rebuild_segments_index.py | dataset.db e sursa de adevăr + export_catalog.py + sync_from_disk |
| cleanup_orphan_segments.py | verify_dataset.py (detectează orfanii disc↔DB) |
| drop_failed_and_renumber.py | statusuri în tabela videos; renumerotarea nu mai e necesară |
| plot_duration_histogram.py | /api/stats — duration_bands + duration_buckets |
| corpus_metadata_stats.py | /api/stats + view-ul dataset_overview |
