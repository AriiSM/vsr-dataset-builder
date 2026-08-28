# Romanian VSR Dataset Pipeline

Pipeline pentru construirea unui dataset de **Visual Speech Recognition**
(citire pe buze) în limba română cu accent moldovenesc, din video-uri de pe
YouTube: descarcă, taie pe **propoziții** (niciodată în mijlocul unui cuvânt),
urmărește fețele, alege vorbitorul activ, decupează gura (96×96, grayscale,
aliniată), transcrie cu timpi per cuvânt și scrie totul într-un catalog SQLite.
Un UI web (React) controlează pipeline-ul și servește la curatoriat (review).

```
┌─────────────┐   joburi    ┌─────────────┐   citește/scrie   ┌──────────────────┐
│  UI (React)  │ ─────────▶ │  API FastAPI │ ◀───────────────▶ │ data/catalog/     │
│  browser     │ ◀───────── │  fără GPU    │                   │   dataset.db      │
└─────────────┘   status    └─────────────┘                   └──────────────────┘
                                                                     ▲
                             ┌─────────────┐   rulează pipeline      │
                             │   WORKER     │ ────────────────────────┘
                             │ (procesul GPU)│──▶ data/processed/{video}/
                             └─────────────┘     face_crop/ mouth_crop/ audio/ text/
```

Nu există separare dev/producție: aceeași configurare, aceleași comenzi, peste tot.

---

## 1. Cerințe preliminare

| Ce | Notă |
|---|---|
| **git** | |
| **Docker** | Windows: Docker Desktop cu backend WSL2 (are suport CUDA nativ). Linux: docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| **GPU NVIDIA** (≥4 GB VRAM) + driver la zi | doar pentru worker (procesare); `nvidia-smi` trebuie să meargă pe gazdă |
| cont Hugging Face + token | diarizare (pyannote) — vezi pasul 3 |

> **Windows/cmd — două reguli care scutesc erori ciudate:** rulează comenzile
> **una pe rând** (nu lipi două pe aceeași linie) și exact așa cum sunt scrise
> — toate sunt deja pe o singură linie, compatibile cmd/PowerShell/bash.
> La comenzi Windows native (`del`, `copy`) folosește backslash: `data\catalog\...`.

**Windows**: plafonează RAM-ul mașinii WSL2 (altfel își rezervă jumătate din
RAM) — creează `%UserProfile%\.wslconfig` cu:

```ini
[wsl2]
memory=8GB
```

---

## 2. Clonare

```bash
git clone <URL-repo> vsr
cd vsr

# TalkNet-ASD (detecția vorbitorului activ) — repo extern, clonat în rădăcină.
# ATÂT: nu se instalează cu pip (nu are setup.py) — codul îl ia de pe disc;
# imaginea de worker îl copiază la build.
git clone https://github.com/TaoRuijie/TalkNet-ASD.git
```

---

## 3. Configurarea

```bash
cp .env.example .env        # macOS/Linux/Git Bash
copy .env.example .env      # Windows (cmd)
```

apoi completează:

- **`HF_TOKEN`** în `.env` — pentru diarizare (cine vorbește când):
  1. cont pe [huggingface.co](https://huggingface.co) → deschide pagina
     `pyannote/speaker-diarization-3.1` → **accepți termenii**;
  2. Settings → Access Tokens → creează un token *read*;
  3. pune-l și în `config/config.yaml` la `segmentation.diarization.hf_token`.
- Restul parametrilor pipeline-ului stau în **`config/config.yaml`**
  (praguri, modele Whisper, tot) — montat read-only în containere; îl poți
  edita oricând, workerul îl recitește la fiecare job, fără restart.

---

## 4. Build-ul imaginilor (pe mașina de procesare)

```bash
docker compose -f docker/compose.yaml build
```

Trei imagini: `frontend` (nginx, ~50 MB), `api` (python slim, ~250 MB),
`worker` (CUDA + tot stack-ul ML, ~8 GB — durează la primul build; layerele
sunt ordonate ca rebuild-urile ulterioare să fie rapide).

> Datele NU trăiesc în imagini: `./data`, `./models`, `./config` sunt volume
> locale — supraviețuiesc oricărui rebuild.
>
> Opțional, pentru instalare rapidă pe alte mașini: pune `IMAGE_PREFIX` +
> `TAG` în `.env` și `docker compose -f docker/compose.yaml push` (Docker
> Hub / GHCR); pe cealaltă mașină doar `pull` în loc de `build`.

---

## 5. Modelele — un singur folder portabil: `models/`

Manifestul greutăților e în **`config/models.yaml`**; toate cache-urile
(Whisper, Silero, insightface) se redirecționează automat sub `models/` —
folderul e complet și portabil (copiabil pe stick, pe mașini fără internet).

```bash
# descarcă tot ce se poate automat (buffalo_l, Whisper, Silero):
docker compose -f docker/compose.yaml run --rm worker python backend/tools/fetch_models.py
```

- **`talknet_asd.pth` + `syncnet_v2.pth`** nu au URL public stabil: copiază-le
  manual în `models/` (sau urcă-le o dată în repo-ul tău Hugging Face și pune
  URL-ul în `config/models.yaml` — de atunci devin și ele automate). Apoi:

```bash
# fixează hash-urile fișierelor prezente (integritate garantată de-acum):
docker compose -f docker/compose.yaml run --rm worker python backend/tools/fetch_models.py --pin
```

---

## 6. Verificarea mașinii — `doctor`

```bash
docker compose -f docker/compose.yaml run --rm worker python backend/tools/doctor.py
```

Raport verde/roșu: sistem, GPU + VRAM, pachete, modele (prezență + hash),
catalog, config (inclusiv token-ul pyannote), frontend. **Nu porni
pipeline-ul până nu e verde tot ce e blocant** (exit 1 = mai ai de rezolvat).

---

## 7. Inițializare + pornire

```bash
# structura data/ + catalogul SQLite (o singură dată):
docker compose -f docker/compose.yaml run --rm worker python backend/orchestrator/cli.py init ./data

# pornirea aplicației:
docker compose -f docker/compose.yaml --profile ui --profile worker up -d
```

Deschide **http://localhost:8080** (portul din `VSR_UI_PORT`):

- **Process** — adaugi video-uri prin *Bulk import*:
  - normal: lipești URL-uri (unul pe linie), primesc id-uri `md_001`, … și
    se descarcă;
  - **cu bifa „Videos already downloaded"**: lipești perechi
    `md_001 https://...` — pentru corpusuri PRE-descărcate în `data/raw/`:
    nu se descarcă nimic, se iau doar metadatele de la link (titlu, canal,
    durată, licență), iar fișierul raw e verificat cu ffprobe (durata față
    de metadate) înainte de înregistrarea în catalog (dataset.db);

  și pornești joburi: batch / single / resume; log live; *Stop* = anulare
  curată, reluabilă;
- **Review** — aprobi / respingi / editezi segmente (taste: A/R/S/E + săgeți);
- **Explorer** — cauți prin segmente; **Stats** — statisticile corpusului.

Din LAN: `http://<ip-ul-mașinii>:8080`. De oriunde: [Tailscale](https://tailscale.com)
pe ambele mașini — aceeași adresă, criptat, zero configurare de router.

Rulare utilă separat (noaptea, când pipeline-ul stă) — a doua opinie pe
transcripturi:

```bash
docker compose -f docker/compose.yaml --profile refiner up refiner
```

Catalogul se deschide cu [DB Browser for SQLite](https://sqlitebrowser.org):
`data/catalog/dataset.db` — **OBLIGATORIU cu „Open Database Read Only" cât
timp aplicația rulează**: deschis normal, DB Browser ține un lock de scriere
care, peste puntea Windows→WSL2, blochează complet workerul
(`unable to open database file`, crash-loop). Verificat pe viu la pilot.

---

## 8. Operare zilnică

```bash
docker compose -f docker/compose.yaml ps                  # ce rulează
docker compose -f docker/compose.yaml logs -f worker      # logul workerului
docker compose -f docker/compose.yaml down                # oprire (datele rămân)
# update după schimbări de cod — TOATE cele 3 verigi, în ordine:
git pull
docker compose -f docker/compose.yaml build
docker compose -f docker/compose.yaml --profile ui --profile worker up -d
# (codul e COPIAT în imagini: fără `build`, containerele rulează senin codul vechi)

# sănătatea datelor + exporturi CSV la cerere:
docker compose -f docker/compose.yaml run --rm worker python backend/tools/verify_dataset.py
docker compose -f docker/compose.yaml run --rm worker python backend/tools/export_catalog.py
```

Rezultatul per video: `data/processed/{video_id}/` cu `face_crop/` (capul
256×256), `mouth_crop/` (gura 96×96 — datele de antrenare), `audio/`,
`text/` (transcript + timpi per cuvânt). Toate metadatele: `dataset.db`.

---

## 9. Rulare fără Docker (alternativa nativă)

Aceleași comenzi, același comportament — utilă pe o mașină de dezvoltare
fără GPU (doar API + UI) sau dacă preferi venv:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# varianta completă (mașina cu GPU):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt -r backend/api/requirements.txt

# varianta doar-API (fără GPU, ~50 MB):
pip install -r backend/api/requirements.txt

python backend/tools/fetch_models.py     # + doctor.py, cli.py init — ca mai sus
python backend/run_api.py                # → http://localhost:8000
python backend/run_worker.py             # alt terminal (doar cu stack-ul complet)
```

---

## 10. Probleme frecvente

| Simptom | Cauza / soluția |
|---|---|
| `could not select device driver "nvidia"` | NVIDIA Container Toolkit lipsă (Linux) / Docker Desktop fără WSL2 (Windows); verifică `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi` |
| build-ul workerului pică la `COPY TalkNet-ASD/` | n-ai clonat TalkNet-ASD în rădăcină (pasul 2) |
| Diarizarea eșuează zgomotos | token HF lipsă / termeni neacceptați — pasul 3 |
| UI-ul nu se deschide pe 8080 | portul ocupat → schimbă `VSR_UI_PORT` în `.env` |
| Job blocat pe „running" după o pană | pornește workerul — îl marchează `interrupted`; apoi *Resume* |
| `doctor` roșu la modele | rulează fetch_models (pasul 5); talknet/syncnet se copiază manual |
| WSL2 mănâncă RAM | `.wslconfig` cu `memory=8GB` (pasul 1) + restart Docker Desktop |
| `zsh: command not found: python` (nativ, macOS) | folosește `python3` sau activează venv-ul |
| worker crash-loop: `unable to open database file` | închide DB Browser (redeschide-l DOAR „Read Only"); dacă persistă: `docker compose ... down`, șterge `data\catalog\dataset.db-wal` și `.db-shm` dacă există, pornește workerul primul |
| `attempt to write a readonly database` | imagine veche (api rula ne-root) — `git pull` + `build` |
| aceeași eroare și după „fix" | ai sărit o verigă: `git pull` → verifică fix-ul în fișier (`findstr`) → `build` → `up` |
| jurnalul DB e `delete`, nu `wal` (doctor) | normal pe bind-mount Windows — fallback automat, zero impact |
| venv stricat după redenumirea/mutarea folderului (nativ) | venv-urile țin calea absolută: șterge `.venv` și recreează-l |
| `address already in use` la pornirea API (nativ) | API-ul rulează deja; sau pornește cu `--port 9000` |
| toate clipurile pică la export: `module 'mediapipe' has no attribute 'solutions'` | mediapipe prea nou (API-ul clasic `mp.solutions` a fost scos) — `git pull` (pin `mediapipe==0.10.21`) + `build`; apoi șterge `data\clips\{video_id}\.checkpoint.json` și dă *Resume* — altfel clipurile picate rămân marcate „procesate" și sunt sărite |

Documentație internă: `plan.html` (arhitectura completă + roadmap),
`PIPELINE_NOTES.md` (jurnalul deciziilor), `CLAUDE.md` (ghid pentru lucrul
cu Claude Code în acest repo).
