# Project Structure

```text
Helios/
├── app.py                     # Streamlit chemistry demo app
├── configs/                   # YAML configs (data/model/training)
├── data/                      # Datasets and generated artifacts
├── docs/                      # Project documentation
├── frontend/                  # React + Vite frontend
├── scripts/                   # Operational and training scripts
├── src/helios/                # Main Python package
│   ├── data/                  # Downloaders and processors
│   ├── inference/             # Inference pipelines
│   ├── models/                # Model wrappers/components
│   ├── training/              # Training modules
│   ├── ui/                    # API server + chatbot UI
│   └── utils/                 # Chemistry, metrics, visualization helpers
├── requirements.txt
└── setup.py
```

## Ownership guide

- Product/UI changes: `frontend/src/components/`
- API contract changes: `src/helios/ui/api_server_v2.py`
- Inference logic: `src/helios/inference/`
- Data pipeline changes: `scripts/run_data_pipeline.py` + `src/helios/data/`
- Chemistry utility functions: `src/helios/utils/`

## Keep in Git vs ignore

Tracked:

- source code (`src/`, `frontend/src/`, `scripts/`)
- configs (`configs/`)
- docs (`docs/`, `README.md`)

Ignored:

- large data/model artifacts (`data/raw`, `models/checkpoints`, `data/rag`, etc.)
- local environments (`venv`, `node_modules`)
- logs and runtime outputs
