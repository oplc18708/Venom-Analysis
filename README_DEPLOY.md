# Venom Analysis Deployment

## Data Directory

Venom Analysis stores persistent runtime data in `VENOM_DATA_DIR`.

- Default (local): `./data`
- Docker default: `/data`

Stored here:
- Active DB pointer: `.venom_active_db`
- Empty DB fallback: `empty.duckdb`
- Imported event DBs: `event_<event_key>_<timestamp>.duckdb`
- Notes / auto path assets

DB resolution order:
1. `VENOM_DB_PATH` (explicit override)
2. `<VENOM_DATA_DIR>/.venom_active_db`
3. `<VENOM_DATA_DIR>/empty.duckdb` (empty state)

## Local Run

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m streamlit run streamlit_app.py
```

Optional:

```bash
export VENOM_DATA_DIR=./data
```

## Docker Run

Build:

```bash
docker build -t venom-analysis:latest .
```

Run:

```bash
docker run --rm -p 8501:8501 -e VENOM_DATA_DIR=/data -v "$(pwd)/data:/data" venom-analysis:latest
```

The app listens on `0.0.0.0:8501`.

