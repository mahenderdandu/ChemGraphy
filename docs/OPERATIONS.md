# Operations Guide

## Daily commands

### Start full stack

- macOS/Linux: `./start.sh`
- Windows PowerShell: `.\start.ps1`
- Windows CMD: `start.bat`

### Bootstrap only

- macOS/Linux: `./bootstrap.sh`
- Windows PowerShell: `.\bootstrap.ps1`

### Run backend only

```bash
python -m uvicorn helios.ui.api_server_v2:app --host 0.0.0.0 --port 8000 --reload
```

### Run frontend only

```bash
cd frontend
npm run dev -- --host 0.0.0.0 --port 3000
```

## Pre-push cleanup

Preview cleanup:

```bash
python scripts/prepare_github_repo.py --dry-run
```

Apply cleanup:

```bash
python scripts/prepare_github_repo.py --apply
```

## GitHub publishing checklist

1. Run cleanup script
2. Confirm ignored files are not staged: `git status`
3. Validate startup scripts work on your machine
4. Push source code and docs

## Production hardening suggestions

- Restrict `CORS` origins in `api_server_v2.py`
- Add authentication before deploying publicly
- Add environment-based config (`.env` + pydantic settings)
- Run behind a reverse proxy (Nginx/Caddy)
- Add CI checks for lint/tests
