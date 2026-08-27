# data/ — layout v2 (Storage v2, Faza 6.5)

```
data/
├── raw/                      # video-urile sursă descărcate (yt-dlp)
│   └── {video_id}.mp4
├── clips/                    # clipuri intermediare (temporare; cleanup_clips le șterge)
│   └── {video_id}/
│       ├── clip_XXX.mp4      # DOAR video
│       ├── clip_XXX.wav      # DOAR audio
│       ├── clips.json        # manifestul v2 (cuvinte + timpi + speakeri)
│       └── .checkpoint.json  # resume per clip
├── processed/                # folder AUTONOM per video — datasetul final
│   └── {video_id}/
│       ├── face_crop/{segment_id}.mp4    # capul 256×256 (review + identitate)
│       ├── mouth_crop/{segment_id}.mp4   # gura 96×96 grayscale (antrenare)
│       ├── audio/{segment_id}.wav        # audio-ul segmentului
│       └── text/{segment_id}.txt         # transcript + timpi per cuvânt
├── catalog/                  # TOATE metadatele
│   ├── dataset.db            # SQLite (WAL) — sursa de adevăr: videos, segments,
│   │                         #   words, speakers, embeddings, dropped_clips, jobs
│   ├── exports/              # CSV-uri la cerere (backend/tools/export_catalog.py)
│   └── backups/              # backup-uri dataset.db (automate, cu rotație)
└── logs/                     # log-urile joburilor ({job_id}.log)
```

Deschide `catalog/dataset.db` cu DB Browser for SQLite (read-only când pipeline-ul rulează).
