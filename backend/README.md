# backend/

Tot codul Python al proiectului. Este organizat ca un **monolit modular**:
- fiecare etapă a pipeline-ului este un serviciu separat, în `services/`;
- orchestratorul le importă în același proces;
- serviciile comunică prin fișiere pe disc (`data/`) și prin catalogul
  MongoDB, nu prin HTTP.

| Folder / fișier | Rol |
|---|---|
| [`api/`](#api) | API-ul FastAPI pentru interfața web (fără GPU, fără PyTorch) |
| [`worker/`](#worker) | Consumatorul cozii de joburi: singurul proces cu GPU |
| [`orchestrator/`](#orchestrator) | Pipeline-ul propriu-zis: compune serviciile, video cu video și clip cu clip |
| [`services/`](services/README.md) | Etapele pipeline-ului (descărcare, segmentare, fețe, vorbitor activ, export, calitate) |
| [`shared/vsr_shared/`](#shared) | Cod comun: accesul la catalog, modelul documentelor, normalizare text, WER |
| [`tools/`](tools/README.md) | Unelte de linie de comandă: verificare, migrări, transcrieri suplimentare, exporturi |
| [`tests/`](#tests) | Testele unitare și de integrare |
| `run_api.py` / `run_worker.py` | Pornesc API-ul și worker-ul din orice director |

Importurile se rezolvă cu `backend/` și `backend/shared/` în `sys.path`.
Lansatoarele (`run_*.py`, `orchestrator/cli.py`, uneltele) își configurează
singure calea.

## api

Aplicația FastAPI din `api/main.py`. Servește interfața compilată și
endpoint-urile `/api/*`. Documentația OpenAPI se generează automat la `/docs`.

| Router | Endpoint-uri |
|---|---|
| `routers/jobs.py` | Coada de joburi: creare, stare, oprire, log; plus straturile de compatibilitate `/api/start`, `/api/status`, `/api/stop` |
| `routers/videos.py` | Registrul de videoclipuri, cu statistici calculate din segmente |
| `routers/segments.py` | Lista din Explorer, detaliile unui segment, fișierele media, reasignarea vorbitorului |
| `routers/review.py` | Aprobare / respingere / editare / tăiere a unui segment |
| `routers/speakers.py` | Vorbitorii, cu agregate și editări manuale |
| `routers/stats.py` | Agregatele din Dashboard, distribuțiile, vocabularul |

- `stats_service.py` calculează agregatele.
- `cache.py` păstrează în memorie răspunsurile mari (lista de segmente,
  statisticile) și le reîmprospătează în fundal.

## worker

`worker/main.py` preia atomic joburile din colecția `jobs` și le rulează cu
`VSRPipeline`, prin `job_executor.py`. Progresul ajunge înapoi în catalog prin
`progress_reporter.py`, iar logul fiecărui job e scris în
`data/logs/job_{id}.log`.

- Tipuri de joburi: `single`, `batch`, `resume`, `resume_batch`, `sync`.
- Configurația se recitește la fiecare job.
- Modelele rămân încărcate între joburi cât timp `config.yaml` nu se schimbă.

## orchestrator

| Fișier | Rol |
|---|---|
| `pipeline.py` | `VSRPipeline`: procesarea unui video, de la descărcare la identitatea vorbitorilor |
| `clip_processor.py` | Cascada pentru un clip: fețe → filtru de vizibilitate → transcriere → vorbitor activ → export → calitate |
| `service_container.py` | Construiește serviciile la prima folosire (modelele se încarcă doar când e nevoie) |
| `catalog_writer.py` | Scrie în catalog tot ce produce pipeline-ul, într-un singur lot per clip |
| `checkpoint_store.py` | Starea de reluare (`data/clips/{video_id}/.checkpoint.json`) |
| `pipeline_config.py` | Citește `config/config.yaml` |
| `cli.py` | Linia de comandă: `init`, `single`, `batch`, `stats`, `resume` |

## shared

| Modul | Rol |
|---|---|
| `catalog_db.py` | **Singurul** modul care știe de baza de date: colecții, interogări, backup-uri |
| `catalog_model.py` | Conversia exactă rânduri plate ⇄ documente MongoDB (vezi mai jos) |
| `speakers_registry.py` | Vorbitorii cu metadate editate manual și agregate calculate |
| `text_normalization.py` | Curățarea transcrierilor Whisper (diacritice ș/ț, punctuație) |
| `wer_utils.py` | Calculul WER |
| `model_env.py` | Redirecționează cache-urile tuturor modelelor în `models/` |
| `excel_schema.py` | Coloanele exporturilor CSV |

### Catalogul MongoDB (baza `vsr-dataset-md`)

| Colecție | Conținut |
|---|---|
| `videos` | Un document per video sursă: `source`, `metadata`, `pipeline`, `stats` și clipurile eliminate (`dropped_clips[]`) |
| `segments` | Un document per segment exportat, adică setul de date propriu-zis (structura e detaliată sub tabel) |
| `speakers` | Identitatea: gen, vârstă, centroidul ArcFace (512 valori) |
| `segment_embeddings` | Vectorul ArcFace al fiecărui segment |
| `jobs` | Coada dintre API și worker |
| view-uri | `speaker_stats`, `video_stats`, `dataset_overview` (agregate calculate live) |

Un document din `segments` conține:
- `time`: timpii segmentului;
- `quality.{tier, asd, syncnet, whisper}`: nivelul de calitate și scorurile;
- `face.{bbox, head_pose}`: poziția și orientarea feței;
- `mouth`: datele decupajului gurii;
- `transcripts.*`: cele trei variante Whisper și consensul;
- `wer`: erorile dintre variante;
- `review`: starea revizuirii;
- `words[]`: cuvintele, cu timpii lor.

Restul codului lucrează cu rânduri plate, cu numele istorice ale coloanelor.
`catalog_model.py` face conversia în ambele sensuri.

## tests

```bash
python -m pytest backend/tests
```

Testele de catalog folosesc o bază de date MongoDB temporară, care se șterge
la final, deci vsr-mongo trebuie să ruleze. Testele care au nevoie de modele
(GPU) se sar automat dacă modelele lipsesc.
