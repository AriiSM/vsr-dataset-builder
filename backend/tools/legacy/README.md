# backend/tools/legacy/

Unelte din prima versiune a proiectului, când metadatele stăteau în CSV-uri
(`data/metadata/*.csv`, `data/annotations/`). Au fost înlocuite de catalogul
MongoDB și de funcțiile pipeline-ului actual. Sunt păstrate doar ca referință.

**Nu le rula pe datele actuale:** citesc și scriu structura veche de fișiere.

| Unealtă | Înlocuită de |
|---|---|
| `cluster_speakers.py` | Identificarea vorbitorilor din pipeline (`quality_indexer`: DBSCAN per video + re-identificare între videoclipuri) |
| `predict_speaker_metadata.py` | Estimarea genului și vârstei per persoană, în pipeline |
| `backfill_syncnet.py` | SyncNet rulează acum pentru fiecare clip |
| `rebuild_segments_index.py` | Catalogul MongoDB + `export_catalog.py` |
| `cleanup_orphan_segments.py` | `verify_dataset.py` |
| `drop_failed_and_renumber.py` | Stările din colecția `videos` |
| `plot_duration_histogram.py` | Cardul *Clip durations* din Dashboard (`/api/stats`) |
| `corpus_metadata_stats.py` | `/api/stats` și view-ul `dataset_overview` |
