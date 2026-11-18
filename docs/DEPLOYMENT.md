# Deployment Notes

This project can run in production behind a process manager and reverse proxy.

## Backend deployment

Recommended production command:

```bash
python -m uvicorn helios.ui.api_server_v2:app --host 0.0.0.0 --port 8000 --workers 2
```

For larger workloads, consider `gunicorn` with `uvicorn.workers.UvicornWorker`.

## Frontend deployment

Build static assets:

```bash
cd frontend
npm run build
```

Serve `frontend/dist/` via Nginx/Caddy or a static hosting platform.

## Required hardening before internet exposure

- Replace permissive CORS (`allow_origins=["*"]`) with trusted origins
- Add authentication and rate limiting
- Add request/body size limits for uploads
- Add structured logging and monitoring
- Keep model/data artifacts outside Git and provision them at deploy time

## Suggested infrastructure

- Reverse proxy: Nginx or Caddy
- TLS termination: proxy layer
- Process manager: systemd, supervisord, or container orchestration
- Secrets: environment variables / secret manager
