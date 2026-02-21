# Venom Analysis on Unraid

Use these exact Unraid template settings.

## Template Settings

- Network Type: `bridge` (or custom, but bridge is fine)
- Port mapping:
  - Host `8501` -> Container `8501`
- Path mapping:
  - Host: `/mnt/user/appdata/venom-analysis`
  - Container: `/data`
- Environment variable:
  - `VENOM_DATA_DIR=/data`

## Notes

- DuckDB files and `.venom_active_db` will be created under `/mnt/user/appdata/venom-analysis`
- Cloudflare tunnel should point to `http://<unraid-ip>:8501`
- Container restarts won’t wipe data because `/data` is persistent

