# config/

| Fișier | Rol |
|---|---|
| `config.yaml` | **Toți** parametrii pipeline-ului. Worker-ul îl recitește la fiecare job |
| `models.yaml` | Manifestul modelelor: cale, sursă (URL sau copiere manuală), hash SHA-256, dacă e obligatoriu. Îl folosesc `fetch_models.py` și `doctor.py` |
| `dialect_prompt.txt` | Text în graiul moldovenesc, dat ca prompt inițial lui Whisper de `whisper_dialectat_mediu.py` și `whisper_dialectat_large.py` |

## Secțiunile din `config.yaml`

| Secțiune | Ce controlează |
|---|---|
| `paths` | Directoarele de date |
| `download` | yt-dlp: format, filtru de licență, cookies |
| `audio` | Whisper: model (`medium` … `large-v3`), dispozitiv, limbă |
| `segmentation` | Strategia (`sentence` implicit, `vad` pentru comportamentul vechi), diarizarea pyannote |
| `vad_splitting` | Pragurile Silero VAD, ștergerea clipurilor intermediare |
| `video` | Detecția fețelor (RetinaFace) |
| `asd` / `syncnet` | Detecția vorbitorului activ și verificarea sincronizării |
| `clip_filters` | Filtrele de clipuri (de ex. procentul minim de cadre cu fața vizibilă) |
| `performance` | Loturi, fire de execuție, folosirea memoriei GPU |
| `export` / `mouth_roi` | Dimensiunile și encodarea decupajelor, regiunea gurii |
| `speaker_identity` | Gruparea pe persoane (DBSCAN) și re-identificarea între videoclipuri |
| `quality_tiers` | Pragurile pentru nivelurile A/B/C (după o schimbare: `backend/tools/rebuild_segments_index.py`) |
| `logging` | Nivelul și formatul logurilor |

Secretele nu stau aici: token-ul Hugging Face și conexiunea MongoDB sunt în
`.env` (vezi README-ul principal).
