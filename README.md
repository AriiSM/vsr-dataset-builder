# Romanian VSR Dataset Builder

Pipeline pentru construirea unui set de date de **Visual Speech Recognition**
(citire pe buze) în limba română cu accent moldovenesc, pornind de la
videoclipuri publice de pe YouTube.

Pentru fiecare video, pipeline-ul:

1. îl descarcă (yt-dlp, cu filtru de licență);
2. îl taie în **propoziții**, la pauze și punctuație, niciodată în mijlocul
   unui cuvânt (Silero VAD + WhisperX pe tot audio-ul);
3. urmărește fețele (RetinaFace + Kalman) și alege vorbitorul activ
   (TalkNet-ASD, verificat cu SyncNet);
4. decupează gura (96×96, grayscale, aliniată) și capul (256×256);
5. salvează transcrierea cu timpi pentru fiecare cuvânt;
6. grupează fețele pe persoane (ArcFace + DBSCAN) și calculează un nivel de
   calitate A/B/C pentru fiecare segment.

Fișierele media rămân pe disc, în `data/`. Toate metadatele stau într-un
catalog **MongoDB**. O interfață web (React) pornește pipeline-ul, arată
statisticile și servește la revizuirea manuală a segmentelor.

## Arhitectura

```
┌──────────────┐  /api    ┌──────────────┐   citește/scrie  ┌─────────────────────┐
│ vsr-frontend │ ───────▶ │   vsr-api    │ ◀──────────────▶ │     vsr-mongo       │
│ React+nginx  │          │ FastAPI, CPU │                  │ DB vsr-dataset-md   │
└──────────────┘          └──────────────┘                  └─────────────────────┘
       │ /media                  │ coada de joburi                    ▲
       ▼                         ▼                                    │
  data/processed/        ┌─────────────────────┐   rulează pipeline  │
  (read-only)            │ vsr-pipeline-worker │ ────────────────────┘
                         │ GPU (PyTorch, CUDA) │ ──▶ data/processed/{video_id}/
                         └─────────────────────┘
```

- **vsr-frontend:** servește interfața, fișierele video (`/media`, direct de
  pe disc) și trimite cererile `/api` mai departe.
- **vsr-api:** catalogul, revizuirea și coada de joburi. Nu folosește GPU
  și nu are nevoie de PyTorch.
- **vsr-pipeline-worker:** singurul proces care folosește GPU. Preia joburile
  din coadă și rulează pipeline-ul.
- **vsr-mongo:** serviciu separat (`../vsr-mongo`), comun tuturor containerelor
  prin rețeaua Docker `vsr-net`.

## Structura repository-ului

| Folder | Conținut |
|---|---|
| [`backend/`](backend/README.md) | Tot codul Python: API, worker, orchestrator, servicii, unelte, teste |
| [`frontend/`](frontend/README.md) | Interfața web React (Vite) |
| [`docker/`](docker/README.md) | Dockerfile-urile și `compose.yaml` |
| [`config/`](config/README.md) | Parametrii pipeline-ului și manifestul modelelor |
| [`data/`](data/README.md) | Video-urile sursă și setul de date rezultat (nu intră în git) |
| [`notebooks/`](notebooks/README.md) | Analiză statistică interactivă |
| `models/` | Greutățile modelelor (nu intră în git; vezi pasul 5) |
| `TalkNet-ASD/` | Repo extern, clonat separat (vezi pasul 2) |

---

## 1. Cerințe

| Ce | Notă |
|---|---|
| git | |
| Docker | Windows: Docker Desktop cu WSL2. Linux: docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| GPU NVIDIA (≥ 4 GB VRAM) + driver actual | Doar pentru worker; `nvidia-smi` trebuie să meargă pe gazdă |
| Cont Hugging Face + token | Pentru diarizare (pyannote), vezi pasul 3 |

Pe Windows, limitează memoria WSL2 în `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=8GB
```

## 2. Clonare

```bash
git clone https://github.com/AriiSM/vsr-dataset-builder.git
cd vsr-dataset-builder

# TalkNet-ASD (detecția vorbitorului activ): nu se instalează cu pip,
# codul îl folosește direct de pe disc, iar imaginea worker îl copiază la build.
git clone https://github.com/TaoRuijie/TalkNet-ASD.git
```

## 3. Configurare

```bash
cp .env.example .env          # Windows cmd: copy .env.example .env
```

Variabile din `.env`:

| Variabilă | Rol |
|---|---|
| `HF_TOKEN` | Token Hugging Face (*read*). Trebuie acceptați termenii modelului `pyannote/speaker-diarization-3.1` |
| `MONGO_URI` | Conexiunea la vsr-mongo, de ex. `mongodb://<user>:<parolă>@127.0.0.1:27018/?authSource=admin` |
| `MONGO_DB` | Baza de date a catalogului: `vsr-dataset-md` |
| `VSR_UI_PORT` | Portul interfeței (implicit `8080`) |
| `IMAGE_PREFIX`, `TAG`, `GIT_SHA` | Numele și versiunea imaginilor Docker |

Parametrii pipeline-ului (modele Whisper, praguri, filtre) sunt în
[`config/config.yaml`](config/README.md). Worker-ul recitește fișierul la
fiecare job, fără restart.

## 4. Pornirea MongoDB

```bash
cd ../vsr-mongo
docker compose up -d          # creează și rețeaua vsr-net
cd ../vsr-dataset-builder
```

## 5. Build și modele

```bash
docker compose -f docker/compose.yaml build
docker compose -f docker/compose.yaml run --rm worker python backend/tools/fetch_models.py
```

- `fetch_models.py` descarcă tot ce are URL public (Whisper, Silero, insightface)
  în `models/`.
- `talknet_asd.pth` și `syncnet_v2.pth` se copiază manual în `models/`.
- Comanda de mai jos fixează hash-urile, ca integritatea fișierelor să poată
  fi verificată:

  ```bash
  docker compose -f docker/compose.yaml run --rm worker python backend/tools/fetch_models.py --pin
  ```

## 6. Verificarea mașinii

```bash
docker compose -f docker/compose.yaml run --rm worker python backend/tools/doctor.py
```

Raportul verde/roșu verifică:
- sistemul, GPU-ul și pachetele;
- modelele (prezență + hash);
- conexiunea la MongoDB și configurația.

Nu porni pipeline-ul până când toate verificările blocante nu sunt verzi
(`exit 1` înseamnă că mai e ceva de rezolvat).

## 7. Inițializare și pornire

```bash
# structura data/ + indexurile catalogului (o singură dată)
docker compose -f docker/compose.yaml run --rm worker python backend/orchestrator/cli.py init ./data

# interfața + API-ul + worker-ul
docker compose -f docker/compose.yaml --profile ui --profile worker up -d
```

Deschide **http://localhost:8080**. Interfața are cinci taburi:

| Tab | Rol |
|---|---|
| **Dashboard** | Progresul colectării, calitatea, câte ore rămân după filtrele de antrenare, consensul transcrierilor, echilibrul vorbitorilor |
| **Data** | Tabelele cu videoclipuri, vorbitori și vocabular |
| **Explorer** | Galerie cu toate segmentele, căutare și filtre |
| **Review** | Revizuirea manuală, segment cu segment (taste: A aprobă, R respinge, S sare, E editează) |
| **Process** | Adaugă videoclipuri și pornește joburile (batch, single, resume), cu log live |

## 8. Operare zilnică

```bash
docker compose -f docker/compose.yaml ps                     # ce rulează
docker compose -f docker/compose.yaml logs -f worker         # logul worker-ului
docker compose -f docker/compose.yaml down                   # oprire (datele rămân)

# după modificări de cod (codul e copiat în imagini, deci build-ul e obligatoriu)
git pull
docker compose -f docker/compose.yaml build
docker compose -f docker/compose.yaml --profile ui --profile worker up -d

# consistența disc ↔ catalog și exporturi CSV la cerere
docker compose -f docker/compose.yaml run --rm worker python backend/tools/verify_dataset.py
docker compose -f docker/compose.yaml run --rm worker python backend/tools/export_catalog.py
```

A doua opinie pe transcrieri (Whisper large-v3), de rulat când pipeline-ul
nu procesează:

```bash
docker compose -f docker/compose.yaml --profile refiner up refiner
```

## 9. Setul de date rezultat

Pentru fiecare video, în `data/processed/{video_id}/`:

| Folder | Conținut |
|---|---|
| `mouth_crop/` | Gura, 96×96 grayscale, 25 fps: **intrarea modelului de citire pe buze** |
| `face_crop/` | Capul, 256×256: pentru revizuire și identificarea vorbitorului |
| `audio/` | Audio-ul segmentului (.wav) |
| `text/` | Transcrierea + timpii fiecărui cuvânt |

Metadatele fiecărui segment sunt în colecția `segments` din MongoDB:
- timpii;
- scorurile ASD, SyncNet și Whisper;
- cele trei variante de transcriere Whisper și consensul dintre ele;
- cuvintele;
- vorbitorul;
- nivelul de calitate;
- starea revizuirii.

## 10. Rulare fără Docker

```bash
python -m venv .venv
source .venv/Scripts/activate                 # Linux/macOS: source .venv/bin/activate

# mașina cu GPU (pipeline complet)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt -r backend/api/requirements.txt

# doar API + interfață (fără GPU)
pip install -r backend/api/requirements.txt

python backend/run_api.py                     # http://localhost:8000
python backend/run_worker.py                  # alt terminal, doar cu stack-ul complet
```

MongoDB trebuie să ruleze, iar `MONGO_URI` / `MONGO_DB` să fie setate în `.env`.

## 11. Probleme frecvente

| Simptom | Cauză / soluție |
|---|---|
| `network vsr-net declared as external, but could not be found` | vsr-mongo nu a fost pornit (pasul 4) |
| API-ul sau worker-ul nu pornesc: `ServerSelectionTimeoutError` | MongoDB oprit sau `MONGO_URI` greșit în `.env` |
| `could not select device driver "nvidia"` | Lipsește NVIDIA Container Toolkit (Linux) sau WSL2 (Windows) |
| Build-ul worker-ului pică la `COPY TalkNet-ASD/` | TalkNet-ASD nu e clonat în rădăcină (pasul 2) |
| Diarizarea eșuează | `HF_TOKEN` lipsă sau termenii pyannote neacceptați (pasul 3) |
| Interfața nu se deschide pe 8080 | Portul e ocupat: schimbă `VSR_UI_PORT` în `.env` |
| Un job rămâne „running” după o pană | Repornește worker-ul (îl marchează `interrupted`), apoi *Resume* |
| Containerele rulează cod vechi | Lipsește `build` după `git pull` |
| `module 'mediapipe' has no attribute 'solutions'` | mediapipe prea nou: imaginea fixează `mediapipe==0.10.21`; refă build-ul worker-ului |
| `module 'torchaudio' has no attribute 'AudioMetaData'` | torchaudio prea nou: imaginea fixează `torch 2.7.1` / `torchaudio 2.7.1`; refă build-ul |
| `Weights only load failed ... ListConfig` la Whisper | Imaginea setează `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`; refă build-ul worker-ului |
| `Could not load library libcudnn_ops_infer.so.8` | ctranslate2 cere cuDNN 8; imaginea îl instalează în `/opt/cudnn8`; refă build-ul |
