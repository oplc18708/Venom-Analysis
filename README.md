# Venom Analysis

Venom Analysis is a Streamlit + DuckDB scouting analytics app with unified CSV/ZIP import for MatchRobot, Super Scout, Pit, and Schedule exports.

## Local Run

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m streamlit run streamlit_app.py
```

Then import data in **Admin → Unified Import**.

## Docker Run

Build image:

```bash
docker build -t venom-analysis:latest .
```

Run container:

```bash
docker run --rm -p 8501:8501 -e VENOM_DATA_DIR=/data -v "$(pwd)/server_data:/data" venom-analysis:latest
```

## Unraid Deployment

See `README_UNRAID.md` for the exact template settings (path mapping, port mapping, and environment variable).

## Sample Data

Use files in `sample_data/` to validate auto-detect + normalization:
- `sample_data/matchrobot_sample.csv`
- `sample_data/superscout_sample.csv`
- `sample_data/pit_sample.csv`
- `sample_data/schedule_sample.csv`

## Important

- Do not commit DuckDB files (`*.duckdb`, `*.duckd`) or runtime data directories.
- Runtime persistent data should live under `VENOM_DATA_DIR` (defaults to `./data` locally).
