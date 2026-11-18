# ChemGraphy

ChemGraphy is a chemistry AI workspace that combines molecular analysis utilities, retrieval pipelines, and chat-style interfaces for research workflows.

## What this repository includes

- Python package under `src/helios` for data, inference, UI, and chemistry utilities
- FastAPI backend (`helios.ui.api_server_v2`) for the React frontend
- React + Vite frontend under `frontend/`
- Data and model tooling scripts under `scripts/`

## Quick start (single command)

### macOS / Linux

```bash
./start.sh
```

### Windows (PowerShell)

```powershell
.\start.ps1
```

### Windows (CMD)

```bat
start.bat
```

These commands automatically bootstrap dependencies on first run, then start:

- Download required datasets (MoleculeNet + ChEMBL)
- Build RAG artifacts (`data/rag/chembl_corpus.json`, `data/rag/embeddings.npy`)
- Warm-download embedding + LLM model caches
- Train Graphormer (if `models/checkpoints/graphormer/best_model.pt` is missing)
- Keep only the best GNN checkpoint (`best_model.pt`)

Set `HELIOS_PREPARE_ASSETS=0` to skip the preparation phase.
On first run, this may take a long time depending on network/CPU.

- API: `http://localhost:8000`
- Frontend: `http://localhost:3000`

## Documentation

- `docs/SETUP.md` — complete setup for macOS, Linux, and Windows
- `docs/DATASETS.md` — dataset locations, expected files, and generated artifacts
- `docs/ARCHITECTURE.md` — backend/frontend and pipeline architecture
- `docs/PROJECT_STRUCTURE.md` — repository structure and ownership map
- `docs/OPERATIONS.md` — daily commands and release workflow
- `docs/DEPLOYMENT.md` — production deployment and hardening notes
- `CONTRIBUTING.md` — team collaboration and PR workflow

## Preparing a clean GitHub push

Preview what would be removed:

```bash
python scripts/prepare_github_repo.py --dry-run
```

Apply cleanup:

```bash
python scripts/prepare_github_repo.py --apply
```

Then push:

```bash
git init
git add .
git commit -m "Initial clean commit"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

## Notes

- Large data/model artifacts are intentionally ignored via `.gitignore`.
- Keep secrets in `.env` and never commit them.
