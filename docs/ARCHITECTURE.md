# Architecture

## High-level

Helios has a two-tier runtime:

1. Backend API (`FastAPI`) in `src/helios/ui/api_server_v2.py`
2. Frontend web client (`React + Vite`) in `frontend/`

The backend orchestrates retrieval + molecular analysis + LLM response generation through the fusion pipeline.

## Backend components

- API server: `helios.ui.api_server_v2`
- Pipeline: `helios.inference.rag_llm_pipeline_fusion.FusionRAGLLMPipeline`
- Chemistry utilities: `src/helios/utils/`
- Data tooling: `src/helios/data/` and `scripts/`

Core backend responsibilities:

- Session management and chat orchestration
- Molecule parsing/visualization and property endpoints
- Query execution through RAG + model fusion
- Returning structured payloads for UI rendering

## Frontend components

- App entry: `frontend/src/main.jsx`, `frontend/src/App.jsx`
- State: `frontend/src/store/chatStore.js`
- Main UI: `frontend/src/components/ModernChatInterface.jsx`

Core frontend responsibilities:

- Chat UX and session navigation
- Rendering sources, molecule cards, and visual outputs
- Calling backend `/api/*` endpoints

## Request flow

1. User submits query from frontend
2. Frontend sends request to backend (`/api/...`) via Vite proxy
3. Backend runs fusion pipeline and optional molecule analysis
4. Backend returns answer, sources, and optional structured chemistry data
5. Frontend renders chat response and visualization blocks

## Runtime ports

- Backend: `8000`
- Frontend: `3000`

Both are configurable through environment variables in startup scripts:

- `BACKEND_PORT`
- `FRONTEND_PORT`
