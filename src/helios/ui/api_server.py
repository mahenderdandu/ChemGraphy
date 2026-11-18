"""
FastAPI backend for Helios Chat Interface
Provides REST API for the React frontend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import numpy as np
from pathlib import Path

from helios.inference.rag_llm_pipeline_fusion import FusionRAGLLMPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="ChemGraphy Chemistry AI API",
    description="Multimodal AI for Chemistry: GNN + RAG + LLM with Fusion",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = None


class QueryRequest(BaseModel):
    question: str
    smiles: Optional[str] = None
    num_results: int = 5
    temperature: float = 0.3  # Lower for more focused responses


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    molecule_info: Optional[Dict[str, Any]] = None
    attention_weights: Optional[Dict[str, Any]] = None


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup"""
    global pipeline
    
    logger.info("Initializing ChemGraphy Pipeline...")
    
    pipeline = FusionRAGLLMPipeline(
        rag_dir="data/rag",
        llm_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        graphormer_checkpoint="models/checkpoints/graphormer/best_model.pt",
        device="cpu",
        use_fusion=True
    )
    
    logger.info("Pipeline initialized successfully!")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "ChemGraphy Chemistry AI API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "pipeline_loaded": pipeline is not None,
        "models": {
            "llm": "TinyLlama-1.1B-Chat-v1.0",
            "gnn": "Graphormer",
            "rag": "sentence-transformers/all-MiniLM-L6-v2"
        }
    }


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a chemistry query with optional molecule
    
    Args:
        request: Query request with question, optional SMILES, and parameters
    
    Returns:
        QueryResponse with answer, sources, and molecule analysis
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        logger.info(f"Processing query: {request.question[:50]}...")
        
        # Run pipeline
        result = pipeline.query(
            question=request.question,
            smiles=request.smiles,
            num_results=request.num_results,
            max_tokens=1024,
            temperature=request.temperature
        )
        
        # Format molecule info
        molecule_info = None
        if result['molecule_prediction'] is not None:
            molecule_info = {
                "smiles": result['smiles'],
                "predicted_property": float(result['molecule_prediction']),
                "embedding_shape": list(result['molecule_embedding'].shape) if result['molecule_embedding'] is not None else None
            }
        
        # Format attention weights for JSON serialization
        attention_weights = None
        if result['attention_weights']:
            attention_weights = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in result['attention_weights'].items()
            }
        
        return QueryResponse(
            answer=result['answer'],
            sources=result['sources'],
            molecule_info=molecule_info,
            attention_weights=attention_weights
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/examples")
async def get_examples():
    """Get example queries"""
    return {
        "examples": [
            {
                "question": "What is the mechanism of action of aspirin?",
                "smiles": "CC(=O)Oc1ccccc1C(=O)O"
            },
            {
                "question": "What are EGFR inhibitors used for?",
                "smiles": None
            },
            {
                "question": "Explain kinase inhibitors for cancer treatment",
                "smiles": None
            },
            {
                "question": "What is the solubility of this molecule?",
                "smiles": "CCO"
            },
            {
                "question": "Tell me about acetaminophen",
                "smiles": "CC(=O)Nc1ccc(O)cc1"
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
