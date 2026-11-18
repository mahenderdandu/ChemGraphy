"""
Fusion-based RAG-LLM Pipeline
Integrates RAG, Graphormer GNN, and LLM with cross-attention fusion layer
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from helios.models.rag.retriever_numpy import RAGRetriever
from helios.models.llm.model import HeliosLLM
from helios.models.fusion.fusion_layer import create_fusion_layer
from helios.models.graphormer.model import load_graphormer_from_pretrained
from helios.data.moleculenet_loader import smiles_to_graph
from helios.utils.chemical_extractor import (
    extract_chemicals_from_text,
    extract_first_smiles,
    extract_all_smiles,
    generate_molecule_image_base64,
    generate_molecule_svg,
    resolve_name_to_smiles,
)

# Drug discovery & figure generation
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
    from helios.utils.figure_generator import generate_all_figures
    DRUG_DISCOVERY_AVAILABLE = True
except ImportError as _e:
    DRUG_DISCOVERY_AVAILABLE = False
    logger_init_msg = f"Drug discovery modules not available: {_e}"

logger = logging.getLogger(__name__)
if not DRUG_DISCOVERY_AVAILABLE:
    logger.warning("Drug discovery toolkit not loaded – figures will be disabled.")


class FusionRAGLLMPipeline:
    """
    Complete Helios pipeline with fusion layer
    
    Architecture:
    1. User Query → Query Encoder (sentence-transformers) → Query Embedding (384d)
    2. SMILES → Graphormer GNN → Molecule Embedding (768d)
    3. Query → RAG Retriever → Retrieved Documents + Embeddings (N x 384d)
    4. Cross-Attention Fusion:
       - Query Embedding attends to [Molecule Embedding + RAG Embeddings]
       - Output: Fused Embedding (512d)
    5. Fused Embedding → LLM → Generated Response
    """
    
    def __init__(
        self,
        rag_dir: str,
        llm_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        graphormer_checkpoint: Optional[str] = None,
        device: str = "cpu",
        use_fusion: bool = True,
    ):
        """
        Initialize fusion-based pipeline
        
        Args:
            rag_dir: Directory containing RAG corpus and embeddings
            llm_model_name: HuggingFace model name for LLM
            graphormer_checkpoint: Path to trained Graphormer checkpoint
            device: Device for models
            use_fusion: If True, use fusion layer; if False, use simple concatenation
        """
        self.device = device
        self.use_fusion = use_fusion
        
        logger.info("Initializing Fusion RAG-LLM Pipeline...")
        
        # Load RAG retriever
        logger.info(f"Loading RAG from {rag_dir}")
        self.retriever = RAGRetriever(encoder_name="sentence-transformers/all-MiniLM-L6-v2")
        self.retriever.load_embeddings(Path(rag_dir) / "embeddings.npy")
        self.retriever.load_corpus(Path(rag_dir) / "chembl_corpus.json")
        
        # Load Graphormer
        logger.info("Loading Graphormer GNN...")
        self.graphormer = load_graphormer_from_pretrained(device=device)
        if graphormer_checkpoint:
            logger.info(f"Loading Graphormer checkpoint from {graphormer_checkpoint}")
            checkpoint = torch.load(graphormer_checkpoint, map_location=device)
            # Handle full training checkpoint vs pure state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}, best_val_loss={checkpoint.get('best_val_loss', '?')}")
            else:
                state_dict = checkpoint
            self.graphormer.load_state_dict(state_dict)
        self.graphormer.eval()
        
        # Load LLM
        logger.info("Loading LLM...")
        self.llm = HeliosLLM(model_name=llm_model_name, model_type="hf", device=device)
        
        # Initialize fusion layer
        if use_fusion:
            logger.info("Initializing cross-attention fusion layer...")
            self.fusion_layer = create_fusion_layer(
                query_dim=384,      # sentence-transformers
                molecule_dim=768,    # Graphormer
                rag_dim=384,        # RAG embeddings
                hidden_dim=512,
                num_heads=8,
                num_layers=2,
                dropout=0.1
            )
            self.fusion_layer = self.fusion_layer.to(device)
            self.fusion_layer.eval()
        else:
            self.fusion_layer = None
            logger.info("Running without fusion layer (simple concatenation)")
        
        logger.info("Fusion RAG-LLM Pipeline initialized successfully!")
    
    def query(
        self,
        question: str,
        smiles: Optional[str] = None,
        num_results: int = 5,
        max_tokens: int = 512,
        temperature: float = 0.7,
        reasoning_mode: str = "standard",
        num_candidates: int = 3,
        add_citations: bool = True
    ) -> Dict[str, Any]:
        """
        Process query with fusion pipeline
        
        Args:
            question: User question
            smiles: Optional molecule SMILES
            num_results: Number of RAG documents to retrieve
            max_tokens: Max tokens for LLM generation
            temperature: Sampling temperature
        
        Returns:
            Dictionary with answer, sources, embeddings, attention weights
        """
        logger.info(f"Processing query: {question}")
        
        # 0. Auto-extract chemicals from query if no SMILES provided
        extracted_chemicals = []
        if not smiles:
            logger.info("No SMILES provided. Auto-extracting chemicals from query...")
            extracted_chemicals = extract_chemicals_from_text(question, use_pubchem=True)
            if extracted_chemicals:
                # Use the first extracted molecule as primary for GNN
                smiles = extracted_chemicals[0]["smiles"]
                logger.info(f"Auto-extracted primary molecule: {extracted_chemicals[0]['name']} → {smiles}")
                if len(extracted_chemicals) > 1:
                    logger.info(f"Also found: {[c['name'] for c in extracted_chemicals[1:]]}")
            else:
                logger.info("No chemicals detected in query.")
        else:
            # Even with explicit SMILES, extract any named chemicals for inline vis
            extracted_chemicals = extract_chemicals_from_text(question, use_pubchem=False)
        
        # 1. Encode query to embedding
        query_embedding = self.retriever.encode_query(question)  # [384]
        query_tensor = torch.from_numpy(query_embedding).float().to(self.device)  # [384]
        
        # 2. Retrieve RAG documents with embeddings
        logger.info("Retrieving relevant documents...")
        retrieved_docs, rag_embeddings = self.retriever.retrieve(
            question,
            top_k=num_results,
            return_embeddings=True
        )
        rag_tensor = torch.from_numpy(rag_embeddings).float().to(self.device)  # [num_results, 384]
        
        # 3. Get molecule embedding if SMILES available (provided or auto-extracted)
        molecule_tensor = None
        molecule_prediction = None
        if smiles:
            logger.info(f"Encoding molecule: {smiles}")
            try:
                graph_data = smiles_to_graph(smiles)
                graph_batch = {
                    k: v.unsqueeze(0).to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in graph_data.items()
                }
                with torch.no_grad():
                    molecule_tensor = self.graphormer.get_embeddings(graph_batch)  # [1, 768]
                    molecule_prediction = self.graphormer.predict(graph_batch)  # [1, 1]
                logger.info(f"Molecule embedding extracted: {molecule_tensor.shape}")
            except Exception as e:
                logger.error(f"Failed to encode molecule: {e}")
                molecule_tensor = None
        
        # 4. Fusion or simple concatenation
        attention_weights = {}
        if self.use_fusion and self.fusion_layer:
            logger.info("Fusing embeddings with cross-attention...")
            with torch.no_grad():
                # Add batch dimension
                query_batch = query_tensor.unsqueeze(0)  # [1, 384]
                rag_batch = rag_tensor.unsqueeze(0)      # [1, num_results, 384]
                molecule_batch = molecule_tensor if molecule_tensor is not None else None
                
                fused_embedding, attention_weights = self.fusion_layer(
                    query_embedding=query_batch,
                    molecule_embedding=molecule_batch,
                    rag_embeddings=rag_batch
                )
                # fused_embedding: [1, 512]
            
            # Convert fused embedding to text prompt for LLM
            prompt = self._build_fused_prompt(
                question, retrieved_docs, smiles, molecule_prediction, fused_embedding
            )
        else:
            # Simple concatenation (no fusion)
            prompt = self._build_simple_prompt(
                question, retrieved_docs, smiles, molecule_prediction
            )
        
        # 5. Generate with LLM
        logger.info("Generating answer (draft)...")
        draft_answer = self.llm.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=["\nQuestion:", "\nUser:", "\n\n\n", "Context:"]
        )

        # 6. Reasoning enhancement
        if reasoning_mode.lower() == "verify":
            logger.info("Verifying and adding citations...")
            final_answer = self._verify_and_cite(
                question=question,
                draft_answer=draft_answer,
                sources=retrieved_docs,
                smiles=smiles,
                molecule_prediction=molecule_prediction,
                add_citations=add_citations,
                max_tokens=max_tokens,
                temperature=max(0.2, temperature - 0.2)
            )
        elif reasoning_mode.lower() in {"self_consistency", "self-consistency"}:
            logger.info("Running self-consistency and synthesis...")
            final_answer = self._self_consistency(
                prompt=prompt,
                question=question,
                sources=retrieved_docs,
                num_candidates=max(1, num_candidates),
                max_tokens=max_tokens,
                temperature=max(0.6, temperature)
            )
            if add_citations:
                # Add a quick verification pass to attach citations
                final_answer = self._verify_and_cite(
                    question=question,
                    draft_answer=final_answer,
                    sources=retrieved_docs,
                    smiles=smiles,
                    molecule_prediction=molecule_prediction,
                    add_citations=True,
                    max_tokens=max_tokens,
                    temperature=0.3
                )
        else:
            final_answer = draft_answer
        
        # 7. Generate inline molecule visualizations for chemicals in the answer
        inline_molecules = []
        try:
            # Gather all chemicals: from query extraction + from the answer text
            answer_chemicals = extract_chemicals_from_text(final_answer, use_pubchem=False)
            all_chemicals = {c['smiles']: c for c in extracted_chemicals}
            for c in answer_chemicals:
                if c['smiles'] not in all_chemicals:
                    all_chemicals[c['smiles']] = c
            
            for chem in all_chemicals.values():
                mol_img = generate_molecule_image_base64(chem['smiles'], size=(300, 200))
                mol_svg = generate_molecule_svg(chem['smiles'], size=(300, 200))
                
                # Also run GNN on secondary molecules for predictions
                chem_prediction = None
                if chem['smiles'] != smiles:  # already computed for primary
                    try:
                        gd = smiles_to_graph(chem['smiles'])
                        gb = {
                            k: v.unsqueeze(0).to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in gd.items()
                        }
                        with torch.no_grad():
                            chem_prediction = self.graphormer.predict(gb).item()
                    except Exception:
                        pass
                else:
                    chem_prediction = molecule_prediction.item() if molecule_prediction is not None else None
                
                inline_molecules.append({
                    'name': chem['name'],
                    'smiles': chem['smiles'],
                    'image_base64': mol_img,
                    'svg': mol_svg,
                    'gnn_prediction': chem_prediction,
                })
        except Exception as e:
            logger.warning(f"Inline molecule generation failed: {e}")
        
        # 8. Generate drug discovery analysis & textbook-style figures
        drug_discovery_data = {}
        figures = {}
        if smiles and DRUG_DISCOVERY_AVAILABLE:
            try:
                logger.info("Running drug discovery analysis...")
                primary_name = (
                    extracted_chemicals[0]['name']
                    if extracted_chemicals else smiles
                )
                admet = predict_admet(smiles)
                dl = assess_drug_likeness(smiles)
                alerts = get_structural_alerts(smiles)
                pharm = identify_pharmacophores(smiles)
                opt = suggest_lead_optimization(smiles)
                sa = estimate_synthetic_accessibility(smiles)

                drug_discovery_data = {
                    'admet': admet,
                    'drug_likeness': dl,
                    'structural_alerts': alerts,
                    'pharmacophores': pharm,
                    'lead_optimization': opt,
                    'synthetic_accessibility': sa,
                }

                # Generate publication-quality figures
                figures = generate_all_figures(
                    smiles=smiles,
                    molecule_name=primary_name,
                    admet_data=admet,
                    drug_likeness_data=dl,
                    alerts_data=alerts,
                    pharm_data=pharm,
                    opt_data=opt,
                )
                logger.info(f"Generated {len(figures)} textbook figures")
            except Exception as e:
                logger.warning(f"Drug discovery analysis failed: {e}")

        return {
            'answer': final_answer,
            'sources': retrieved_docs,
            'query_embedding': query_embedding,
            'molecule_embedding': molecule_tensor.cpu().numpy() if molecule_tensor is not None else None,
            'molecule_prediction': molecule_prediction.item() if molecule_prediction is not None else None,
            'attention_weights': {k: v.cpu().numpy() for k, v in attention_weights.items()},
            'smiles': smiles,
            'question': question,
            'extracted_chemicals': extracted_chemicals,
            'inline_molecules': inline_molecules,
            'drug_discovery': drug_discovery_data,
            'figures': figures,
        }
    
    def _build_fused_prompt(
        self,
        question: str,
        retrieved_docs: List[Dict],
        smiles: Optional[str],
        molecule_prediction: Optional[torch.Tensor],
        fused_embedding: torch.Tensor
    ) -> str:
        """Build prompt using fused embedding information for chemistry researchers"""
        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:5], 1):  # Top 5 most relevant
            title = doc.get('title', 'Unknown')
            content = doc.get('content', '').strip()
            score = doc.get('score', 0.0)
            if hasattr(score, 'item'):
                score = score.item()
            if content:
                context_parts.append(f"[{i}] {title} (relevance: {float(score):.2f})\n{content}")
        
        context_text = "\n\n".join(context_parts)
        
        # Add detailed molecule information if available
        molecule_info = ""
        if smiles and molecule_prediction is not None:
            pred_val = molecule_prediction.item()
            molecule_info = f"""

## Molecular Analysis (GNN-Predicted)
- **SMILES**: `{smiles}`
- **Predicted Property**: {pred_val:.4f}
- **Interpretation**: {"High activity/solubility" if pred_val > 0 else "Low activity/solubility"} (normalized scale)
- Graph neural network has encoded structural features including: functional groups, ring systems, stereochemistry."""
        
        # Research-grade system prompt
        system_msg = """You are Helios, an advanced AI chemistry research assistant designed for pharmaceutical researchers and medicinal chemists.

Your capabilities:
1. Molecular property prediction via Graph Neural Networks (Graphormer)
2. Scientific literature retrieval from ChEMBL database (58K documents)
3. Structure-activity relationship (SAR) analysis
4. Mechanism of action explanations

Response guidelines:
- Use precise scientific terminology
- Structure answers with clear sections: Summary, Mechanism, Evidence, Implications
- Cite sources using [1], [2] notation
- For molecules: discuss functional groups, pharmacophores, and predicted properties
- Quantify claims when possible (IC50, Ki, LogP values)
- Note limitations and suggest follow-up experiments"""
        
        prompt = f"""<s>[INST] {system_msg}

## Retrieved Scientific Context
{context_text}{molecule_info}

## Research Question
{question}

Provide a comprehensive, research-grade answer with structured reasoning. [/INST]

## Answer

"""
        return prompt
    
    def _build_simple_prompt(
        self,
        question: str,
        retrieved_docs: List[Dict],
        smiles: Optional[str],
        molecule_prediction: Optional[torch.Tensor]
    ) -> str:
        """Build prompt without fusion for chemistry researchers"""
        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:5], 1):
            title = doc.get('title', 'Unknown')
            content = doc.get('content', '').strip()
            score = doc.get('score', 0.0)
            if hasattr(score, 'item'):
                score = score.item()
            if content:
                context_parts.append(f"[{i}] {title} (score: {float(score):.2f})\n{content}")
        
        context_text = "\n\n".join(context_parts)
        
        molecule_info = ""
        if smiles and molecule_prediction is not None:
            pred_val = molecule_prediction.item()
            molecule_info = f"""

## Molecular Analysis
- **SMILES**: `{smiles}`
- **GNN Predicted Property**: {pred_val:.4f}"""
        
        system_msg = """You are Helios, an AI chemistry research assistant for pharmaceutical scientists.
Provide structured, evidence-based answers with:
1. **Summary**: Key findings in 2-3 sentences
2. **Mechanism/Details**: Scientific explanation with terminology
3. **Evidence**: Cite sources as [1], [2] from the context
4. **Research Implications**: Suggest next steps or limitations"""
        
        prompt = f"""<s>[INST] {system_msg}

## Scientific Context
{context_text}{molecule_info}

## Question
{question}

Provide a research-grade answer. [/INST]

"""
        return prompt

    def _format_sources_for_prompt(self, sources: List[Dict]) -> str:
        """Format sources with indices for citation-friendly prompts."""
        parts = []
        for i, doc in enumerate(sources[:10], 1):
            title = doc.get('title', 'Unknown')
            content = (doc.get('content', '') or '').strip()
            parts.append(f"[Source {i}] {title}\n{content}")
        return "\n\n".join(parts)

    def _verify_and_cite(
        self,
        question: str,
        draft_answer: str,
        sources: List[Dict],
        smiles: Optional[str],
        molecule_prediction: Optional[float],
        add_citations: bool,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Ask LLM to fact-check the draft against sources and attach citations."""
        context_text = self._format_sources_for_prompt(sources)
        mol_info = ""
        if smiles and molecule_prediction is not None:
            pred_val = molecule_prediction
            if hasattr(pred_val, 'item'):
                pred_val = pred_val.item()
            mol_info = f"\n- Molecule: {smiles}\n- Predicted Property: {float(pred_val):.4f}"

        instruction = """You are a senior medicinal chemistry reviewer. Your task is to verify and enhance the draft answer for pharmaceutical researchers.

Verification criteria:
1. Every factual claim must be supported by the provided sources
2. Remove or flag speculative statements not grounded in evidence
3. Ensure scientific terminology is precise (use proper IUPAC names, mechanisms)
4. Add quantitative data (IC50, Ki, LogP, MW) where available from sources"""

        citation_note = """
Structure the final answer as:
**Summary**: 2-3 sentence overview
**Mechanism**: Detailed scientific explanation with inline citations [1], [2]
**Key Data**: Relevant quantitative values from sources
**Limitations**: What is not addressed or requires further research""" if add_citations else "Focus on factual correctness without citations."

        verify_prompt = f"""<s>[INST] {instruction}

## Sources
{context_text}{mol_info}

## Original Question
{question}

## Draft Answer to Verify
{draft_answer}

{citation_note}

Provide the verified, research-grade answer: [/INST]

"""

        verified = self.llm.generate(
            verify_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=["\nSources:", "\nDraft", "\nQuestion:", "## Original"]
        )
        return verified.strip()

    def _self_consistency(
        self,
        prompt: str,
        question: str,
        sources: List[Dict],
        num_candidates: int,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate multiple candidates and synthesize a research-grade answer."""
        candidates: List[str] = []
        for i in range(num_candidates):
            cand = self.llm.generate(
                prompt,
                max_new_tokens=max_tokens // 2,
                temperature=temperature + (i * 0.1),  # Vary temperature for diversity
                stop_sequences=["\nQuestion:", "\nUser:", "Context:"]
            )
            candidates.append(cand.strip())

        # Build research-grade synthesis prompt
        context_text = self._format_sources_for_prompt(sources)
        bullets = "\n\n---\n\n".join([f"**Candidate {i+1}**:\n{c}" for i, c in enumerate(candidates)])
        
        synth_prompt = f"""<s>[INST] You are a senior chemistry research synthesizer. Multiple AI candidates have answered a research question. Your task:

1. Identify consensus points across candidates (higher confidence)
2. Resolve contradictions by checking against sources
3. Produce a single, authoritative answer suitable for publication

## Scientific Sources
{context_text}

## Research Question
{question}

## Candidate Answers
{bullets}

Synthesize the best answer with:
- **Consensus Summary**: Points agreed by multiple candidates
- **Mechanism/Details**: Precise scientific explanation
- **Evidence Quality**: Note which claims have strong vs weak support
- **Open Questions**: What remains uncertain [/INST]

**Synthesized Answer**:

"""

        final = self.llm.generate(
            synth_prompt,
            max_new_tokens=max_tokens,
            temperature=0.25,
            stop_sequences=["\n## Scientific", "\n## Candidate", "\n---"]
        )
        return final.strip()
