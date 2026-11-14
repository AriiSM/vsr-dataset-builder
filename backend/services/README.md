# backend/services/

Etapele pipeline-ului. Fiecare folder este un serviciu cu propriul
`requirements.txt` și poate deveni o imagine Docker separată. Deocamdată,
orchestratorul (`backend/orchestrator/`) le importă în același proces.

## Fluxul de date

```
YouTube URL
  │ downloader
  ▼
data/raw/{video_id}.mp4
  │ segmenter: Silero VAD + WhisperX pe tot audio-ul, tăiere pe propoziții
  ▼
data/clips/{video_id}/clip_XXX.{mp4,wav} + clips.json
  │ pentru fiecare clip:
  │   face_tracker      fețe + urmărire
  │   speaker_detector  cine vorbește (TalkNet) + sincronizare (SyncNet)
  │   mouth_exporter    decupajul gurii și al capului, audio, text
  │   quality_indexer   nivel de calitate A/B/C, evidență ArcFace
  ▼
data/processed/{video_id}/…  +  segmentele în catalogul MongoDB
  │ la final de video:
  │   quality_indexer   gruparea fețelor pe persoane (DBSCAN) + re-identificare între videoclipuri
  ▼
speaker_id pentru fiecare segment
```

## Servicii

| Serviciu | Fișierul principal | Ce face |
|---|---|---|
| `downloader/` | `youtube_downloader.py` | Descarcă videoclipul cu yt-dlp și verifică licența (Creative Commons) |
| `segmenter/` | `vad_splitter.py` | Mai multe etape, detaliate sub tabel |
| `face_tracker/` | `video_processor.py` | Detectează fețele (RetinaFace ResNet50), le asociază în piste (IoU) și netezește cadrele (Kalman) |
| `speaker_detector/` | `speaker_selector.py` | Alege vorbitorul activ cu TalkNet-ASD și verifică sincronizarea audio-video cu SyncNet. Dacă un model lipsește, comută vizibil (cu avertisment) pe o metodă de rezervă |
| `mouth_exporter/` | `segment_exporter.py` | Mai multe etape, detaliate sub tabel |
| `quality_indexer/` | `speaker_identifier.py`, `quality_tiers.py` | Mai multe etape, detaliate sub tabel |
| `transcript_refiner/` | `main.py` | Rulat separat, după pipeline: transcrie din nou segmentele cu Whisper large-v3 și marchează cu `needs_review` segmentele cu dezacord mare |

**`segmenter/`:**
- detectează vorbirea (Silero VAD);
- transcrie tot audio-ul o singură dată (WhisperX);
- găsește granițele de propoziție (`sentence_segmenter.py`);
- face diarizarea (pyannote);
- taie clipurile (ffmpeg).

**`mouth_exporter/`:**
- detectează reperele buzelor (MediaPipe FaceMesh);
- calculează traiectoria decupajului, netezită cu One-Euro;
- encodează `mouth_crop` 96×96 și `face_crop` 256×256;
- scrie audio-ul și transcrierea.

**`quality_indexer/`:**
- calculează nivelul de calitate A/B/C al fiecărui segment;
- extrage vectorii ArcFace;
- grupează fețele pe persoane cu DBSCAN și le recunoaște între videoclipuri
  (prin centroizi);
- estimează genul și vârsta;
- verifică dacă vocea și fața aparțin aceleiași persoane.

## Reguli comune

- Serviciile nu scriu direct în baza de date. Rezultatele trec prin
  `orchestrator/catalog_writer.py`.
- Modelele se încarcă la prima folosire, prin `orchestrator/service_container.py`.
- Toți parametrii vin din `config/config.yaml`.
