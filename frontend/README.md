# frontend/

Interfața web: o aplicație React cu o singură pagină, construită cu Vite.
Comunică doar cu API-ul FastAPI (`/api/*`); fișierele video vin direct de pe
disc (`/media/*`).

## Taburi

| Tab | Folder | Rol |
|---|---|---|
| **Dashboard** | `src/tabs/stats/` | Indicatori, progresul colectării, calitatea, filtrele de antrenare, consensul transcrierilor, echilibrul vorbitorilor |
| **Data** | `src/tabs/stats/` | Tabelele cu videoclipuri (coloanele goale sunt ascunse), vorbitori și vocabular; documentul MongoDB al fiecărui video |
| **Explorer** | `src/tabs/explorer/` | Galeria segmentelor, cu căutare, filtre de calitate și detalii |
| **Review** | `src/tabs/review/` | Revizuirea segment cu segment: aprobare, respingere, editare, tăiere |
| **Process** | `src/tabs/process/` | Pornirea joburilor (batch / single / resume), importul de videoclipuri, log live |

## Structura

```
frontend/
├── index.html, vite.config.js, package.json
├── dist/                 # build-ul de producție (inclus în git, ca să ruleze fără Node)
└── src/
    ├── main.jsx          # punctul de intrare
    ├── App.jsx           # header-ul, navigarea, taburile
    ├── api.js            # toate apelurile HTTP
    ├── config.js         # ținta setului de date (regiunea MD, 200 h)
    ├── styles.css        # tema (întunecată)
    ├── components/       # componente comune: paginare, liste, notificări, arborele unui document
    ├── hooks/            # useLocalStorage, useDebouncedValue
    ├── utils/            # formatare, praguri de calitate, parserul logului, diff de cuvinte
    └── tabs/             # câte un folder pentru fiecare tab
```

Toate taburile rămân montate când treci de la unul la altul. De exemplu,
tabul Process continuă să urmărească un job care rulează chiar dacă te uiți
la Dashboard.

## Dezvoltare

```bash
python backend/run_api.py        # terminal 1: API-ul, http://localhost:8000
cd frontend
npm install                      # prima dată
npm run dev                      # terminal 2: http://localhost:5173 (trimite /api către :8000)
```

## Producție

În Docker, imaginea `vsr-frontend` compilează aplicația și o servește cu nginx
(vezi [`docker/`](../docker/README.md)):

```bash
docker compose -f docker/compose.yaml --profile ui build frontend
docker compose -f docker/compose.yaml --profile ui up -d frontend
```

Fără Docker: `npm run build` scrie bundle-ul în `dist/`, iar
`python backend/run_api.py` îl servește la http://localhost:8000.

## Teste

```bash
npm test          # Vitest: parserul logului, fixat pe mesajele exacte ale backend-ului
```

## Scurtături în Review

| Tastă | Acțiune |
|---|---|
| `A` | Aprobă |
| `R` | Respinge. Se apasă de două ori, pentru că respingerea șterge fișierele de pe disc |
| `S` / `→` | Sari la următorul |
| `←` | Înapoi |
| `U` | Anulează |
| `E` | Editează transcrierea |
| `Space` | Play / pauză |
| `Ctrl+Enter` | Salvează în timpul editării |
| `?` | Lista scurtăturilor |
