# docker/

Imaginile Docker și orchestrarea lor. Toate comenzile se rulează din
rădăcina repository-ului.

| Fișier | Imagine / rol |
|---|---|
| `frontend.Dockerfile` | `vsr-frontend`: compilează React (Node) și îl servește cu nginx |
| `nginx.conf` | nginx: aplicația statică, `/media/` direct de pe disc (cu Range), `/api/` trimis către API, `/_health` |
| `api.Dockerfile` | `vsr-api`: FastAPI pe `python:3.11-slim`, fără PyTorch și fără GPU |
| `worker.Dockerfile` | `vsr-pipeline-worker`: CUDA + PyTorch + tot stack-ul ML. Folosit și de `vsr-transcript-refiner`. La build cere `TalkNet-ASD/` clonat în rădăcină |
| `compose.yaml` | Serviciile, volumele și rețelele (proiectul `vsr-dataset-builder`) |

## Profiluri

| Profil | Containere | Comandă |
|---|---|---|
| `ui` | `vsr-frontend`, `vsr-api` | `docker compose -f docker/compose.yaml --profile ui up -d` |
| `worker` | `vsr-pipeline-worker` | `docker compose -f docker/compose.yaml --profile ui --profile worker up -d` |
| `refiner` | `vsr-transcript-refiner` | `docker compose -f docker/compose.yaml --profile refiner up refiner` |

## Volume și rețele

- Datele **nu** se află în imagini. Rămân pe disc și se montează în containere:
  - `../data`: read-only pentru frontend;
  - `../models`: modelele;
  - `../config`: read-only pentru API.
- **MongoDB** e un proiect Compose separat (`../vsr-mongo`). Containerele îl
  găsesc ca `vsr-mongo:27017` prin rețeaua externă `vsr-net`, deci vsr-mongo
  trebuie pornit primul.
- Variabilele vin din `../.env`: `MONGO_URI`, `MONGO_DB`, `HF_TOKEN`,
  `VSR_UI_PORT`, `IMAGE_PREFIX`, `TAG`, `GIT_SHA`.

## Reconstruire după modificări de cod

Codul e copiat în imagini, deci după orice modificare trebuie refăcut build-ul:

```bash
docker compose -f docker/compose.yaml --profile ui build api frontend
docker compose -f docker/compose.yaml --profile ui up -d api frontend
docker compose -f docker/compose.yaml --profile worker build worker   # doar dacă s-a schimbat pipeline-ul
```

## Registry (opțional)

Setează `IMAGE_PREFIX` și `TAG` în `.env`, apoi:
- pe mașina de build: `docker compose -f docker/compose.yaml push`;
- pe celelalte mașini: `docker compose -f docker/compose.yaml pull`, în loc de `build`.
