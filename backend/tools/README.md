# backend/tools/

Unelte de linie de comandă pentru întreținerea și analiza setului de date. Nu
sunt servicii și nu au imagini proprii. Rulează în containerul worker:

```bash
docker compose -f docker/compose.yaml run --rm worker python backend/tools/<unealta>.py [opțiuni]
# sau, cu worker-ul deja pornit:
docker exec -it vsr-pipeline-worker python /app/backend/tools/<unealta>.py [opțiuni]
```

Fiecare script are documentația completă în docstring și un `--help`.

## Pregătirea mașinii

| Unealtă | Rol |
|---|---|
| `doctor.py` | Raport verde/roșu: GPU, pachete, modele, conexiunea MongoDB, configurație |
| `fetch_models.py` | Descarcă și verifică greutățile din `config/models.yaml`; `--pin` fixează hash-urile |

## Catalog: verificare și exporturi

| Unealtă | Rol |
|---|---|
| `verify_dataset.py` | Compară catalogul cu fișierele din `data/processed/` (segmente orfane, fișiere lipsă) |
| `export_catalog.py` | Exportă catalogul în CSV, în `data/catalog/exports/` |
| `rebuild_segments_index.py` | Recalculează nivelul de calitate A/B/C pentru toate segmentele, după o schimbare a pragurilor din `config.yaml`, fără reprocesare |
| `import_videos.py` | Importă în catalog videoclipurile dintr-un CSV de curator (poate fi rulat din nou; nu suprascrie datele pipeline-ului) |

## Transcrieri dialectale

| Unealtă | Rol |
|---|---|
| `whisper_dialectat_mediu.py` | Transcrie din nou segmentele cu Whisper medium și promptul dialectal din `config/dialect_prompt.txt` |
| `whisper_dialectat_large.py` | Același lucru, cu Whisper large-v3 |
| `transcript_consensus.py` | Calculează WER între cele 3 variante și încadrează fiecare segment într-o categorie de consens: `all_agree`, `dialectal_agree`, `std_large_agree`, `std_med_agree`, `all_disagree` |
| `review_transcripts_gui.py` | Interfață desktop (Tkinter) pentru alegerea manuală a variantei corecte dintre cele 3 |

## Migrări (rulate o singură dată)

| Unealtă | Rol |
|---|---|
| `migrate_to_document_model.py` | Convertește catalogul plat (`lip_reading`) în modelul de documente (`vsr-dataset-md`), cu verificare dus-întors |
| `import_review_status.py` | Importă vechiul `review_status.json` în câmpurile `review` ale segmentelor |

## Metadate video (fluxul vechi, pe CSV)

Aceste unelte lucrează pe `videos_master.csv`, nu direct pe catalog.

| Unealtă | Rol |
|---|---|
| `tag_conditions.py` | Etichetare interactivă a mediului și a zgomotului, video cu video |
| `auto_tag_conditions.py` | Propuneri automate de etichete (`proposals.csv`) |
| `set_env_proposals.py` | Corectează propuneri pentru anumite videoclipuri |
| `apply_tag_proposals.py` | Aplică propunerile în `videos_master.csv` |
| `review_gui.py` | Vechea interfață desktop de revizuire. Revizuirea curentă se face din tabul **Review** al interfeței web |

## legacy/

Unelte din etapa în care metadatele stăteau în CSV-uri. Au fost înlocuite și
nu trebuie rulate pe datele actuale; vezi [`legacy/README.md`](legacy/README.md).
