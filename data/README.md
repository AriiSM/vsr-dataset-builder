# data/

Fișierele media ale setului de date. Conținutul nu intră în git; din folder
se păstrează doar structura. Metadatele (videoclipuri, segmente, cuvinte,
vorbitori, joburi) **nu** stau aici, ci în catalogul MongoDB
(`vsr-dataset-md`).

```
data/
├── raw/                      # videoclipurile sursă descărcate
│   └── {video_id}.mp4
├── clips/                    # clipuri intermediare (șterse dacă cleanup_clips e activ)
│   └── {video_id}/
│       ├── clip_XXX.mp4      # doar video
│       ├── clip_XXX.wav      # doar audio
│       ├── clips.json        # manifestul: cuvinte, timpi, vorbitori
│       └── .checkpoint.json  # starea de reluare, per clip
├── processed/                # SETUL DE DATE: câte un folder autonom per video
│   └── {video_id}/
│       ├── mouth_crop/{segment_id}.mp4   # gura 96×96 grayscale, 25 fps (intrarea modelului)
│       ├── face_crop/{segment_id}.mp4    # capul 256×256 (revizuire, identitate)
│       ├── audio/{segment_id}.wav        # audio-ul segmentului
│       └── text/{segment_id}.txt         # transcrierea + timpii cuvintelor (format LRS2)
├── catalog/
│   ├── backups/              # backup-uri automate ale catalogului (gzip, ultimele 5)
│   └── exports/              # exporturi CSV la cerere (backend/tools/export_catalog.py)
└── logs/                     # logurile joburilor (job_{id}.log)
```

## Formatul fișierului `text/{segment_id}.txt`

```
Text:  ACESTE ZILE CÂND GĂTEȘTI
Conf:  2
WORD START END ASDSCORE
ACESTE 0.00 0.42 8.1
...
```

- `Conf`: încrederea transcrierii (1 = scăzută, 2 = medie, 3 = ridicată).
- Timpii sunt relativi la începutul segmentului.

## Identificatori

| Identificator | Format | Exemplu |
|---|---|---|
| video | `md_NNN` | `md_087` |
| segment | `{video_id}_clip_{clip}_{index}` | `md_087_clip_012_00034` |
| vorbitor | `{video_id}_spk{N}`, păstrat și în alte videoclipuri dacă persoana e recunoscută | `md_087_spk0` |
