# Venom Analysis – Event Builder (Offline-first)

This tool takes your scouting CSVs and builds a single DuckDB database file (`.duckdb`) for an event.
That database contains:
- raw scouting rows (`raw`)
- optional schedule (`schedule`)
- optional pit data (`pit`)
- optional cached TBA schedule (`tba_matches`) (prefetch online before event)
- SQL views for team averages, consistency, and audits

## Quick start (Windows / Git Bash)
1) Put your CSV files in a folder, e.g. `data/`
   - `raw.csv` (required) – one row per team per match
   - `schedule.csv` (optional)
   - `pit.csv` (optional)

2) Install deps
```bash
python -m pip install -r requirements.txt
```

3) Build event DB
```bash
python event_builder.py build --raw data/raw.csv --out event_2026myevent.duckdb
```

Optional schedule/pit:
```bash
python event_builder.py build --raw data/raw.csv --schedule data/schedule.csv --pit data/pit.csv --out event_2026myevent.duckdb
```

Optional TBA prefetch (run at home with internet):
```bash
python event_builder.py prefetch-tba --event 2026myevent --out event_2026myevent.duckdb --tba-key YOUR_KEY
```

## Notes
- Column mapping can be automatic, but you’ll get best results by editing `mapping.json`.
- All analytics live in SQL views inside the `.duckdb` file.
