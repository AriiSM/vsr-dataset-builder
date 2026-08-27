# Frontend — Romanian VSR Dataset Builder

React (Vite) single-page app + a small Flask API server (`app.py`) that wraps
the dataset pipeline. The old vanilla-JS UI has been fully replaced by this app.

## Structure

```
frontend/
├── app.py                  # Flask API server (also serves the production build)
├── index.html              # Vite entry page
├── vite.config.js          # dev server + /api proxy to Flask (port 5000)
├── package.json
└── src/
    ├── main.jsx            # React entry point
    ├── App.jsx             # header, tab switching, stats sub-nav
    ├── api.js              # every HTTP call to the Flask API lives here
    ├── styles.css          # the app stylesheet (dark "factory" theme)
    ├── utils/
    │   ├── format.js       # duration/metric formatting, quality thresholds
    │   └── logParser.js    # pure functions that turn the pipeline log into UI state
    ├── hooks/
    │   ├── useLocalStorage.js
    │   └── useDebouncedValue.js
    ├── components/         # shared UI (Pagination, MetricRows, RankList)
    └── tabs/
        ├── process/        # run pipeline, live log, stage "machines"
        ├── stats/          # KPI strip, insights cards, videos/speakers/vocab tables
        ├── explorer/       # segment gallery with lazy-loaded videos + detail modal
        └── review/         # segment-by-segment curation (approve/reject/edit/trim)
```

## Development

Run the processes in separate terminals:

```bash
# terminal 1 — FastAPI backend (from the project root)
python backend/run_api.py         # http://localhost:8000

# terminal 2 — queue worker (only needed to actually run the pipeline)
python backend/run_worker.py

# terminal 3 — React dev server with hot reload
cd frontend
npm install                       # first time only
npm run dev                       # http://localhost:5173  (proxies /api to :8000)
```

Open **http://localhost:5173** while developing.

## Production

```bash
cd frontend && npm run build && cd ..
python backend/run_api.py         # serves the built app at http://localhost:8000
```

`npm run build` writes the bundle to `frontend/dist/`; FastAPI serves it from
there, so the API process + the worker are enough in production.
(`frontend/app.py`, the old Flask server, is DEPRECATED — fallback only.)

## Tests

```bash
cd frontend
npm test          # Vitest — pins the log parser to the backend's exact messages
```

## Notes

- All tabs stay mounted while you switch between them, so the Process tab
  keeps polling a running pipeline even when you are looking at Stats.
- Explorer videos load lazily (IntersectionObserver) — only cards that enter
  the viewport fetch video metadata.
- Review keyboard shortcuts: `A` approve · `R` reject (press twice — reject
  deletes files from disk) · `S`/`→` skip · `←` previous · `U` revert ·
  `E` edit transcript · `Space` play/pause · `Ctrl+Enter` save while editing ·
  `?` shortcut cheat-sheet.
- Large JSON responses (e.g. `/api/segments`) are gzip-compressed by Flask;
  the expensive vocabulary/Conf disk scans are cached on file mtimes.
