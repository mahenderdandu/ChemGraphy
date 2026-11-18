"""
Helios API Server v2 - Enhanced with streaming, file uploads, and visualizations
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, AsyncGenerator
import logging
import numpy as np
import json
import asyncio
import uuid
import os
from pathlib import Path
from datetime import datetime
import base64
from io import BytesIO

# Chemistry imports
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors, AllChem
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
    print("RDKit loaded successfully")
except ImportError:
    RDKIT_AVAILABLE = False
    print("RDKit not available - molecule visualizations disabled")

from helios.inference.rag_llm_pipeline_fusion import FusionRAGLLMPipeline

# Drug discovery toolkit
try:
    from helios.utils.drug_discovery import (
        predict_admet,
        assess_drug_likeness,
        get_structural_alerts,
        identify_pharmacophores,
        suggest_lead_optimization,
        estimate_synthetic_accessibility,
        generate_full_report,
    )
    from helios.utils.figure_generator import (
        generate_all_figures,
        generate_admet_radar,
        generate_drug_likeness_scorecard,
        generate_bioavailability_radar,
        generate_property_comparison,
        generate_alerts_visual,
        generate_pharmacophore_chart,
        generate_optimization_chart,
        generate_annotated_molecule,
    )
    DRUG_DISCOVERY_AVAILABLE = True
except ImportError as _e:
    DRUG_DISCOVERY_AVAILABLE = False
    print(f"Drug discovery toolkit not available: {_e}")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Helios Chemistry AI API v2",
    description="Advanced Chemistry AI with streaming, visualizations, and research tools",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
pipeline = None
chat_sessions: Dict[str, Dict] = {}  # session_id -> {messages, documents, created_at}
uploaded_documents: Dict[str, Dict] = {}  # doc_id -> {filename, content, embeddings}

# ============== MODELS ==============

class QueryRequest(BaseModel):
    question: str
    smiles: Optional[str] = None
    session_id: Optional[str] = None
    num_results: int = 3  # Fewer results for speed
    temperature: float = 0.3
    reasoning_mode: str = "draft"  # Faster mode
    stream: bool = False  # Streaming disabled

class ChatMessage(BaseModel):
    role: str  # user, assistant, system
    content: str
    timestamp: Optional[str] = None
    smiles: Optional[str] = None
    sources: Optional[List[Dict]] = None

class SessionInfo(BaseModel):
    session_id: str
    title: str
    created_at: str
    message_count: int
    questions: List[Dict[str, str]]  # [{id, question, timestamp}]

class MoleculeRequest(BaseModel):
    smiles: str
    highlight_atoms: Optional[List[int]] = None
    show_3d: bool = False

class ReactionRequest(BaseModel):
    reactants: List[str]  # List of SMILES
    products: List[str]   # List of SMILES
    show_atom_mapping: bool = True

# ============== STARTUP ==============

@app.on_event("startup")
async def startup_event():
    global pipeline
    logger.info("Initializing Helios Pipeline v2...")
    
    pipeline = FusionRAGLLMPipeline(
        rag_dir="data/rag",
        llm_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        graphormer_checkpoint="models/checkpoints/graphormer/best_model.pt",
        device="cpu",
        use_fusion=True
    )
    
    # Create uploads directory
    os.makedirs("data/uploads", exist_ok=True)
    logger.info("Helios Pipeline v2 initialized!")

# ============== CHEMISTRY VISUALIZATION ==============

@app.post("/api/molecule/visualize")
async def visualize_molecule(request: MoleculeRequest):
    """Generate molecule visualization with RDKit"""
    if not RDKIT_AVAILABLE:
        raise HTTPException(status_code=503, detail="RDKit not available")
    
    try:
        mol = Chem.MolFromSmiles(request.smiles)
        if mol is None:
            raise HTTPException(status_code=400, detail="Invalid SMILES string")
        
        # Generate 2D coordinates
        AllChem.Compute2DCoords(mol)
        
        # Create drawer with nice styling
        drawer = rdMolDraw2D.MolDraw2DSVG(400, 300)
        drawer.drawOptions().addStereoAnnotation = True
        drawer.drawOptions().addAtomIndices = False
        
        # Highlight atoms if specified
        if request.highlight_atoms:
            colors = {i: (0.2, 0.8, 0.2) for i in request.highlight_atoms}
            drawer.DrawMolecule(mol, highlightAtoms=request.highlight_atoms, highlightAtomColors=colors)
        else:
            drawer.DrawMolecule(mol)
        
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        
        # Calculate molecular properties
        properties = {
            "molecular_weight": round(Descriptors.MolWt(mol), 2),
            "logp": round(Descriptors.MolLogP(mol), 2),
            "hbd": rdMolDescriptors.CalcNumHBD(mol),
            "hba": rdMolDescriptors.CalcNumHBA(mol),
            "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 2),
            "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "formula": rdMolDescriptors.CalcMolFormula(mol),
            "num_atoms": mol.GetNumAtoms(),
            "num_bonds": mol.GetNumBonds(),
        }
        
        # Lipinski's Rule of 5
        properties["lipinski_violations"] = sum([
            properties["molecular_weight"] > 500,
            properties["logp"] > 5,
            properties["hbd"] > 5,
            properties["hba"] > 10
        ])
        properties["drug_like"] = properties["lipinski_violations"] <= 1
        
        return {
            "svg": svg,
            "smiles": request.smiles,
            "properties": properties
        }
        
    except Exception as e:
        logger.error(f"Error visualizing molecule: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/molecule/image")
async def molecule_image(request: MoleculeRequest):
    """Generate molecule PNG image"""
    if not RDKIT_AVAILABLE:
        raise HTTPException(status_code=503, detail="RDKit not available")
    
    try:
        mol = Chem.MolFromSmiles(request.smiles)
        if mol is None:
            raise HTTPException(status_code=400, detail="Invalid SMILES")
        
        img = Draw.MolToImage(mol, size=(400, 300))
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        return {"image": f"data:image/png;base64,{img_base64}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reaction/visualize")
async def visualize_reaction(request: ReactionRequest):
    """Visualize a chemical reaction"""
    if not RDKIT_AVAILABLE:
        raise HTTPException(status_code=503, detail="RDKit not available")
    
    try:
        # Parse molecules
        reactant_mols = [Chem.MolFromSmiles(s) for s in request.reactants]
        product_mols = [Chem.MolFromSmiles(s) for s in request.products]
        
        if None in reactant_mols or None in product_mols:
            raise HTTPException(status_code=400, detail="Invalid SMILES in reaction")
        
        # Create reaction SMILES
        reactant_smiles = ".".join(request.reactants)
        product_smiles = ".".join(request.products)
        rxn_smiles = f"{reactant_smiles}>>{product_smiles}"
        
        # Draw reaction
        rxn = AllChem.ReactionFromSmarts(rxn_smiles, useSmiles=True)
        if rxn is None:
            # Fallback: draw molecules separately
            all_mols = reactant_mols + product_mols
            img = Draw.MolsToGridImage(all_mols, molsPerRow=len(all_mols), 
                                       subImgSize=(200, 200),
                                       legends=request.reactants + ["→"] + request.products)
        else:
            img = Draw.ReactionToImage(rxn, subImgSize=(250, 200))
        
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        return {
            "image": f"data:image/png;base64,{img_base64}",
            "reaction_smiles": rxn_smiles
        }
        
    except Exception as e:
        logger.error(f"Error visualizing reaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/molecule/similarity/{smiles1}/{smiles2}")
async def molecule_similarity(smiles1: str, smiles2: str):
    """Calculate molecular similarity using fingerprints"""
    if not RDKIT_AVAILABLE:
        raise HTTPException(status_code=503, detail="RDKit not available")
    
    try:
        from rdkit import DataStructs
        
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            raise HTTPException(status_code=400, detail="Invalid SMILES")
        
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        
        tanimoto = DataStructs.TanimotoSimilarity(fp1, fp2)
        dice = DataStructs.DiceSimilarity(fp1, fp2)
        
        return {
            "tanimoto_similarity": round(tanimoto, 4),
            "dice_similarity": round(dice, 4),
            "smiles1": smiles1,
            "smiles2": smiles2
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============== STREAMING QUERY ==============

async def stream_response(question: str, smiles: str, session_id: str, 
                          num_results: int, temperature: float) -> AsyncGenerator[str, None]:
    """Stream LLM response token by token"""
    global pipeline
    
    try:
        # Build context from session
        context_messages = []
        if session_id and session_id in chat_sessions:
            context_messages = chat_sessions[session_id].get("messages", [])[-10:]  # Last 10 messages
        
        # Add context to query
        context_text = ""
        if context_messages:
            context_text = "\n\nPrevious conversation:\n"
            for msg in context_messages[-6:]:  # Last 6 for context
                role = "User" if msg["role"] == "user" else "Assistant"
                context_text += f"{role}: {msg['content'][:200]}...\n"
        
        # Get answer (simulated streaming since TinyLlama doesn't natively stream)
        # Using "draft" mode (single LLM call) for speed - "verify" mode is 2x slower
        result = pipeline.query(
            question=context_text + question if context_text else question,
            smiles=smiles,
            num_results=num_results,
            max_tokens=256,  # Shorter responses for faster generation
            temperature=temperature,
            reasoning_mode="draft",  # Single LLM call for speed
            add_citations=False  # Skip citation pass for speed
        )
        
        answer = result['answer']
        
        # Stream the answer word by word with typing effect
        words = answer.split(' ')
        buffer = ""
        
        for i, word in enumerate(words):
            buffer += word + " "
            
            # Yield every few words for smoother streaming
            if i % 3 == 0 or i == len(words) - 1:
                yield f"data: {json.dumps({'type': 'token', 'content': buffer})}\n\n"
                buffer = ""
                await asyncio.sleep(0.02)  # Small delay for typing effect
        
        # Send sources and molecule info at the end
        yield f"data: {json.dumps({'type': 'sources', 'content': result['sources'][:5]})}\n\n"
        
        if result['molecule_prediction'] is not None:
            mol_info = {
                "smiles": result['smiles'],
                "predicted_property": float(result['molecule_prediction']),
            }
            yield f"data: {json.dumps({'type': 'molecule', 'content': mol_info})}\n\n"
        
        # Send inline molecules for visualization
        inline_molecules = result.get('inline_molecules', [])
        if inline_molecules:
            # Serialize inline molecules (images are already base64 strings)
            inline_data = [
                {
                    "name": m.get('name', ''),
                    "smiles": m.get('smiles', ''),
                    "image_base64": m.get('image_base64'),
                    "svg": m.get('svg'),
                    "gnn_prediction": m.get('gnn_prediction'),
                }
                for m in inline_molecules
            ]
            yield f"data: {json.dumps({'type': 'inline_molecules', 'content': inline_data})}\n\n"
        
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

@app.post("/api/query/stream")
async def stream_query(request: QueryRequest):
    """Stream query response"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return StreamingResponse(
        stream_response(
            request.question,
            request.smiles,
            request.session_id,
            request.num_results,
            request.temperature
        ),
        media_type="text/event-stream"
    )

@app.post("/api/query")
async def process_query(request: QueryRequest):
    """Process query with chat context"""
    global pipeline, chat_sessions
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Create or get session
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in chat_sessions:
        chat_sessions[session_id] = {
            "messages": [],
            "documents": [],
            "created_at": datetime.now().isoformat(),
            "title": request.question[:50] + "..."
        }
    
    session = chat_sessions[session_id]
    
    # Build context from previous messages
    context_messages = session["messages"][-6:]
    context_text = ""
    if context_messages:
        context_text = "Previous conversation context:\n"
        for msg in context_messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            context_text += f"{role}: {msg['content'][:150]}...\n"
        context_text += "\nCurrent question: "
    
    try:
        # Run pipeline with context - use draft mode for speed unless explicitly set
        effective_mode = request.reasoning_mode if request.reasoning_mode else "draft"
        result = pipeline.query(
            question=context_text + request.question if context_text else request.question,
            smiles=request.smiles,
            num_results=request.num_results,
            max_tokens=256,  # Shorter for faster generation
            temperature=request.temperature,
            reasoning_mode=effective_mode,
            add_citations=effective_mode == "verify"  # Only cite in verify mode
        )
        
        # Store messages
        timestamp = datetime.now().isoformat()
        session["messages"].append({
            "role": "user",
            "content": request.question,
            "smiles": request.smiles,
            "timestamp": timestamp
        })
        session["messages"].append({
            "role": "assistant", 
            "content": result['answer'],
            "sources": result['sources'][:5],
            "timestamp": timestamp
        })
        
        # Format response
        molecule_info = None
        if result['molecule_prediction'] is not None:
            molecule_info = {
                "smiles": result['smiles'],
                "predicted_property": float(result['molecule_prediction']),
            }
        
        # Include inline molecule visualizations
        inline_molecules = result.get('inline_molecules', [])
        # Clean up for JSON serialization (images are already base64 strings)
        inline_mols_response = []
        for mol_data in inline_molecules:
            inline_mols_response.append({
                "name": mol_data.get('name', ''),
                "smiles": mol_data.get('smiles', ''),
                "image_base64": mol_data.get('image_base64'),
                "svg": mol_data.get('svg'),
                "gnn_prediction": mol_data.get('gnn_prediction'),
            })
        
        # Include extracted chemicals info
        extracted_chemicals = result.get('extracted_chemicals', [])
        
        return {
            "answer": result['answer'],
            "sources": result['sources'][:5],
            "molecule_info": molecule_info,
            "session_id": session_id,
            "inline_molecules": inline_mols_response,
            "extracted_chemicals": extracted_chemicals,
            "auto_extracted_smiles": result.get('smiles'),
            "drug_discovery": result.get('drug_discovery', {}),
            "figures": result.get('figures', {}),
        }
        
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ============== SESSION MANAGEMENT ==============

@app.get("/api/sessions")
async def list_sessions():
    """List all chat sessions"""
    sessions = []
    for sid, session in chat_sessions.items():
        questions = [
            {"id": str(i), "question": msg["content"][:60], "timestamp": msg.get("timestamp", "")}
            for i, msg in enumerate(session["messages"]) if msg["role"] == "user"
        ]
        sessions.append({
            "session_id": sid,
            "title": session.get("title", "Untitled"),
            "created_at": session.get("created_at", ""),
            "message_count": len(session["messages"]),
            "questions": questions
        })
    
    return {"sessions": sorted(sessions, key=lambda x: x["created_at"], reverse=True)}

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session details and messages"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = chat_sessions[session_id]
    questions = [
        {"id": str(i), "question": msg["content"][:60], "timestamp": msg.get("timestamp", "")}
        for i, msg in enumerate(session["messages"]) if msg["role"] == "user"
    ]
    
    return {
        "session_id": session_id,
        "title": session.get("title", "Untitled"),
        "created_at": session.get("created_at", ""),
        "messages": session["messages"],
        "documents": session.get("documents", []),
        "questions": questions
    }

@app.post("/api/sessions")
async def create_session():
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    chat_sessions[session_id] = {
        "messages": [],
        "documents": [],
        "created_at": datetime.now().isoformat(),
        "title": "New Research Session"
    }
    return {"session_id": session_id}

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    return {"status": "deleted"}

# ============== DOCUMENT UPLOAD ==============

@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    session_id: Optional[str] = None
):
    """Upload a research document (PDF, TXT, MD)"""
    allowed_extensions = {'.pdf', '.txt', '.md', '.csv'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"File type not supported. Allowed: {allowed_extensions}")
    
    try:
        doc_id = str(uuid.uuid4())
        content = await file.read()
        
        # Extract text based on file type
        if file_ext == '.pdf':
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(BytesIO(content))
                text = "\n".join([page.extract_text() for page in pdf_reader.pages])
            except ImportError:
                text = "[PDF parsing requires PyPDF2. Install with: pip install PyPDF2]"
        else:
            text = content.decode('utf-8', errors='ignore')
        
        # Store document
        uploaded_documents[doc_id] = {
            "id": doc_id,
            "filename": file.filename,
            "content": text[:50000],  # Limit content size
            "uploaded_at": datetime.now().isoformat(),
            "size": len(content)
        }
        
        # Associate with session if provided
        if session_id and session_id in chat_sessions:
            chat_sessions[session_id].setdefault("documents", []).append(doc_id)
        
        return {
            "doc_id": doc_id,
            "filename": file.filename,
            "size": len(content),
            "preview": text[:500] + "..." if len(text) > 500 else text
        }
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents")
async def list_documents():
    """List uploaded documents"""
    return {
        "documents": [
            {
                "id": doc["id"],
                "filename": doc["filename"],
                "uploaded_at": doc["uploaded_at"],
                "size": doc["size"]
            }
            for doc in uploaded_documents.values()
        ]
    }

@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete an uploaded document"""
    if doc_id in uploaded_documents:
        del uploaded_documents[doc_id]
    return {"status": "deleted"}

# ============== HEALTH & INFO ==============

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Helios Chemistry AI API v2",
        "version": "3.0.0",
        "features": [
            "streaming", "visualizations", "chat_context", "document_upload",
            "drug_discovery", "admet_prediction", "drug_likeness",
            "structural_alerts", "pharmacophores", "lead_optimization",
            "textbook_figures", "bioavailability_radar"
        ]
    }

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "pipeline_loaded": pipeline is not None,
        "rdkit_available": RDKIT_AVAILABLE,
        "drug_discovery_available": DRUG_DISCOVERY_AVAILABLE,
        "active_sessions": len(chat_sessions),
        "uploaded_documents": len(uploaded_documents)
    }

# ============== DRUG DISCOVERY ENDPOINTS ==============

class DrugDiscoveryRequest(BaseModel):
    smiles: str
    molecule_name: Optional[str] = None

class CompareRequest(BaseModel):
    molecules: List[Dict[str, str]]  # [{"name": "...", "smiles": "..."}]

@app.post("/api/drug-discovery/full-report")
async def full_drug_report(request: DrugDiscoveryRequest):
    """Generate comprehensive drug discovery report with all figures."""
    if not DRUG_DISCOVERY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Drug discovery toolkit not available")
    try:
        report = generate_full_report(request.smiles)
        if report is None:
            raise HTTPException(status_code=400, detail="Invalid SMILES string")

        name = request.molecule_name or request.smiles
        figures = generate_all_figures(
            smiles=request.smiles,
            molecule_name=name,
            admet_data=report.get("admet"),
            drug_likeness_data=report.get("drug_likeness"),
            alerts_data=report.get("structural_alerts"),
            pharm_data=report.get("pharmacophores"),
            opt_data=report.get("lead_optimization"),
        )
        return {"report": report, "figures": figures}
    except Exception as e:
        logger.error(f"Drug discovery report error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/drug-discovery/admet")
async def admet_endpoint(request: DrugDiscoveryRequest):
    """Predict ADMET properties for a molecule."""
    if not DRUG_DISCOVERY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Drug discovery toolkit not available")
    try:
        admet = predict_admet(request.smiles)
        if admet is None:
            raise HTTPException(status_code=400, detail="Invalid SMILES")
        figure = generate_admet_radar(admet, request.molecule_name or "")
        return {"admet": admet, "figure": figure}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/drug-discovery/drug-likeness")
async def drug_likeness_endpoint(request: DrugDiscoveryRequest):
    """Assess drug-likeness across multiple filter systems."""
    if not DRUG_DISCOVERY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Drug discovery toolkit not available")
    try:
        dl = assess_drug_likeness(request.smiles)
        if dl is None:
            raise HTTPException(status_code=400, detail="Invalid SMILES")
        figure = generate_drug_likeness_scorecard(dl, request.molecule_name or "")
        return {"drug_likeness": dl, "figure": figure}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/drug-discovery/alerts")
async def alerts_endpoint(request: DrugDiscoveryRequest):
    """Screen for PAINS, Brenk, and toxicophore structural alerts."""
    if not DRUG_DISCOVERY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Drug discovery toolkit not available")
    try:
        alerts = get_structural_alerts(request.smiles)
        figure = generate_alerts_visual(alerts, request.molecule_name or "")
        return {"alerts": alerts, "figure": figure}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/drug-discovery/pharmacophores")
async def pharmacophore_endpoint(request: DrugDiscoveryRequest):
    """Identify pharmacophoric features in a molecule."""
    if not DRUG_DISCOVERY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Drug discovery toolkit not available")
    try:
        pharm = identify_pharmacophores(request.smiles)
        if pharm is None:
            raise HTTPException(status_code=400, detail="Invalid SMILES")
        figure = generate_pharmacophore_chart(pharm, request.molecule_name or "")
        return {"pharmacophores": pharm, "figure": figure}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/drug-discovery/lead-optimization")
async def lead_opt_endpoint(request: DrugDiscoveryRequest):
    """Suggest lead optimization strategies."""
    if not DRUG_DISCOVERY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Drug discovery toolkit not available")
    try:
        opt = suggest_lead_optimization(request.smiles)
        if opt is None:
            raise HTTPException(status_code=400, detail="Invalid SMILES")
        figure = generate_optimization_chart(opt, request.molecule_name or "")
        return {"optimization": opt, "figure": figure}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/drug-discovery/bioavailability")
async def bioavailability_endpoint(request: DrugDiscoveryRequest):
    """Generate bioavailability radar for a molecule."""
    if not DRUG_DISCOVERY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Drug discovery toolkit not available")
    try:
        figure = generate_bioavailability_radar(request.smiles, request.molecule_name or "")
        if not figure:
            raise HTTPException(status_code=400, detail="Invalid SMILES")
        return {"figure": figure}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/drug-discovery/synthetic-accessibility")
async def sa_endpoint(request: DrugDiscoveryRequest):
    """Estimate synthetic accessibility score."""
    if not DRUG_DISCOVERY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Drug discovery toolkit not available")
    try:
        sa = estimate_synthetic_accessibility(request.smiles)
        if sa is None:
            raise HTTPException(status_code=400, detail="Invalid SMILES")
        return {"synthetic_accessibility": sa}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/drug-discovery/compare")
async def compare_molecules(request: CompareRequest):
    """Compare properties of multiple molecules with chart."""
    if not DRUG_DISCOVERY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Drug discovery toolkit not available")
    try:
        reports = []
        for m in request.molecules[:6]:
            r = generate_full_report(m["smiles"])
            if r:
                r["name"] = m.get("name", m["smiles"][:20])
                reports.append(r)
        figure = generate_property_comparison(request.molecules)
        return {"reports": reports, "comparison_figure": figure}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
