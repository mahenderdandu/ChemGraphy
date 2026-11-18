# Setup Guide

This guide gets Helios running on macOS, Linux, and Windows.

## Prerequisites

- Python 3.10+ (3.11 recommended)
- Node.js 18+ and npm (required for `frontend/`)
- Git
- Optional: CUDA-compatible GPU for model acceleration

## One-command startup

### macOS / Linux

```bash
./start.sh
```

### Windows PowerShell

```powershell
.\start.ps1
```

### Windows CMD

```bat
start.bat
```

On first run, startup scripts install Python + frontend dependencies.

## Manual setup

### 1) Clone and move into project

```bash
git clone <repo-url>
cd Helios
```

### 2) Python environment

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
```

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 3) Frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 4) Run backend + frontend

Backend:

```bash
python -m uvicorn helios.ui.api_server_v2:app --host 0.0.0.0 --port 8000 --reload
```

Frontend (new terminal):

```bash
cd frontend
npm run dev -- --host 0.0.0.0 --port 3000
```

## Verify installation

- Open `http://localhost:3000`
- API health check: `http://localhost:8000/health`

## Troubleshooting

- `ModuleNotFoundError: helios`: run `python -m pip install -e .`
- Frontend cannot start: ensure Node.js 18+ and run `npm install` in `frontend/`
- RDKit install issues: use the Python version supported by your platform package wheels
