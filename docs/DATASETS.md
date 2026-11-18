# Datasets and Artifacts

This repository separates source code from large data/model artifacts.

## Data directories

- `data/raw/` — downloaded raw sources (ignored in Git)
- `data/processed/` — intermediate processed files (ignored in Git)
- `data/final/` — curated final datasets for training/evaluation
- `data/moleculenet/` — benchmark CSV files (`bace.csv`, `esol.csv`, `freesolv.csv`, `lipophilicity.csv`)
- `data/rag/` — RAG index artifacts (`chembl_corpus.json`, `embeddings.npy`, `metadata.json`) (ignored in Git)
- `data/uploads/` — runtime user-uploaded files (ignored in Git)

## Models and outputs

- `models/checkpoints/` — training checkpoints (ignored in Git)
- `outputs/` — generated outputs/reports/plots (ignored in Git)
- `logs/` — runtime logs (ignored in Git)

## Recommended Git strategy

- Keep source code, configs, and docs in Git
- Exclude generated data and checkpoints (already configured in `.gitignore`)
- If you need to share large artifacts, use object storage (S3/GCS/Azure) or Git LFS

## Rebuilding artifacts

Common commands:

```bash
python scripts/run_data_pipeline.py
python scripts/build_rag_index.py
python scripts/train_graphormer.py
```

Use `scripts/prepare_github_repo.py` before your first public push to remove local-only artifacts.
