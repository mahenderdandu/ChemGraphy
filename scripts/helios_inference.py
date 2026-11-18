"""
Complete Helios Inference Pipeline
Combines Graphormer property prediction, RAG retrieval, and LLM interpretation
"""

import logging
import torch
from typing import Dict, Optional, List
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from helios.models.graphormer.model import GraphormerModel
from helios.models.rag.retriever_numpy import load_rag_retriever
from helios.models.llm.model import create_chemistry_llm
from helios.inference.rag_llm_pipeline import RAGLLMPipeline

logger = logging.getLogger(__name__)


class HeliosPipeline:
    """
    Complete Helios pipeline: Graphormer + RAG + LLM
    """
    
    def __init__(
        self,
        graphormer_checkpoint: Optional[str] = None,
        rag_dir: str = "data/rag",
        llm_model: Optional[str] = None,
        use_api: bool = False,
        api_key: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        Args:
            graphormer_checkpoint: Path to Graphormer checkpoint
            rag_dir: RAG index directory
            llm_model: LLM model name
            use_api: Use API for LLM
            api_key: API key
            device: Device for models
        """
        self.device = device
        
        # Load Graphormer if checkpoint provided
        self.graphormer = None
        if graphormer_checkpoint and Path(graphormer_checkpoint).exists():
            logger.info(f"Loading Graphormer from {graphormer_checkpoint}")
            self.graphormer = GraphormerModel(
                d_model=768,
                num_layers=6,
                num_heads=8,
                d_ff=768*4
            ).to(device)
            
            checkpoint = torch.load(graphormer_checkpoint, map_location=device)
            if 'model_state_dict' in checkpoint:
                self.graphormer.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.graphormer.load_state_dict(checkpoint)
            
            self.graphormer.eval()
            logger.info("Graphormer loaded")
        
        # Load RAG retriever
        logger.info(f"Loading RAG from {rag_dir}")
        self.retriever = load_rag_retriever(rag_dir)
        
        # Load LLM
        logger.info("Loading LLM...")
        self.llm = create_chemistry_llm(
            model_name=llm_model,
            use_api=use_api,
            api_key=api_key,
            device=device
        )
        
        # Create RAG-LLM pipeline
        self.rag_llm = RAGLLMPipeline(
            retriever=self.retriever,
            llm=self.llm,
            top_k=5
        )
        
        logger.info("Helios pipeline ready!")
    
    def predict_properties(
        self,
        smiles: str,
        properties: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Predict molecular properties with Graphormer
        
        Args:
            smiles: SMILES string
            properties: Properties to predict (if multiple models)
        
        Returns:
            Dictionary of predictions
        """
        if self.graphormer is None:
            raise ValueError("Graphormer not loaded. Provide checkpoint path.")
        
        # Convert SMILES to graph
        graph_data = self.graphormer.smiles_to_graph(smiles)
        if graph_data is None:
            raise ValueError(f"Failed to parse SMILES: {smiles}")
        
        # Predict
        with torch.no_grad():
            # Prepare batch
            batch = {
                'x': graph_data['x'].unsqueeze(0).to(self.device),
                'edge_index': graph_data['edge_index'].to(self.device),
                'batch': torch.zeros(graph_data['x'].size(0), dtype=torch.long).to(self.device)
            }
            
            prediction = self.graphormer(batch)
            value = prediction.squeeze().item()
        
        # For now, return single property
        # TODO: Support multiple property models
        return {'predicted_value': value}
    
    def answer_chemistry_question(
        self,
        question: str,
        max_tokens: int = 300
    ) -> Dict:
        """
        Answer general chemistry question using RAG+LLM
        
        Args:
            question: Chemistry question
            max_tokens: Max response tokens
        
        Returns:
            Response with answer and sources
        """
        return self.rag_llm.query(
            question,
            max_new_tokens=max_tokens
        )
    
    def analyze_molecule(
        self,
        smiles: str,
        question: Optional[str] = None
    ) -> Dict:
        """
        Complete molecule analysis: predict properties + RAG context + LLM interpretation
        
        Args:
            smiles: SMILES string
            question: Optional specific question about the molecule
        
        Returns:
            Complete analysis
        """
        logger.info(f"Analyzing molecule: {smiles}")
        
        # Step 1: Predict properties if Graphormer available
        predictions = {}
        if self.graphormer:
            try:
                predictions = self.predict_properties(smiles)
                logger.info(f"Predictions: {predictions}")
            except Exception as e:
                logger.warning(f"Property prediction failed: {e}")
        
        # Step 2: Retrieve similar molecules
        logger.info("Retrieving similar molecules...")
        similar_mols = self.retriever.retrieve(f"SMILES {smiles}", top_k=3)
        
        # Step 3: Generate interpretation
        logger.info("Generating interpretation...")
        
        if predictions:
            # Build context with predictions
            context = f"Molecule: {smiles}\n\nPredicted properties:\n"
            for prop, val in predictions.items():
                context += f"- {prop}: {val:.3f}\n"
        else:
            context = f"Molecule: {smiles}\n\n"
        
        # Add similar molecules context
        if similar_mols:
            context += "\nSimilar known molecules:\n"
            for i, mol in enumerate(similar_mols, 1):
                context += f"[{i}] {mol['title']}\n"
                context += f"    {mol['text'][:150]}...\n"
        
        # Generate interpretation
        if question:
            prompt = f"{context}\n\nQuestion: {question}\n\nPlease answer based on the molecular structure and available data."
        else:
            prompt = f"{context}\n\nPlease provide an analysis of this molecule, including potential applications, properties, and relevant considerations."
        
        system_prompt = """You are a chemistry expert analyzing molecular structures. 
Provide clear, scientific insights about molecular properties, potential applications, 
safety considerations, and drug-like characteristics."""
        
        interpretation = self.llm.generate(
            prompt,
            system_prompt=system_prompt,
            max_new_tokens=300,
            temperature=0.7
        )
        
        return {
            'smiles': smiles,
            'predictions': predictions,
            'similar_molecules': similar_mols,
            'interpretation': interpretation,
            'question': question
        }
    
    def suggest_molecules(
        self,
        target_description: str,
        top_k: int = 5
    ) -> Dict:
        """
        Suggest molecules for a specific target or application
        
        Args:
            target_description: Description of target/application
            top_k: Number of suggestions
        
        Returns:
            Suggestions with explanations
        """
        logger.info(f"Finding molecules for: {target_description}")
        
        # Retrieve relevant molecules
        candidates = self.retriever.retrieve(target_description, top_k=top_k)
        
        # Format for LLM
        context = f"Target/Application: {target_description}\n\nCandidate molecules:\n\n"
        for i, mol in enumerate(candidates, 1):
            context += f"[{i}] {mol['title']}\n"
            context += f"    {mol['text']}\n\n"
        
        prompt = f"{context}\nBased on these candidates, please:\n1. Explain why these molecules are relevant\n2. Highlight key properties and mechanisms\n3. Suggest which candidates are most promising and why"
        
        system_prompt = """You are a drug discovery expert helping identify promising molecular candidates. 
Provide clear, practical recommendations based on molecular properties and mechanisms."""
        
        recommendation = self.llm.generate(
            prompt,
            system_prompt=system_prompt,
            max_new_tokens=400,
            temperature=0.7
        )
        
        return {
            'target': target_description,
            'candidates': candidates,
            'recommendation': recommendation
        }


def create_helios_pipeline(
    graphormer_checkpoint: Optional[str] = None,
    rag_dir: str = "data/rag",
    llm_model: Optional[str] = None,
    use_api: bool = False,
    api_key: Optional[str] = None,
    device: str = "cpu"
) -> HeliosPipeline:
    """
    Create complete Helios pipeline
    
    Args:
        graphormer_checkpoint: Graphormer checkpoint path
        rag_dir: RAG directory
        llm_model: LLM model name
        use_api: Use API
        api_key: API key
        device: Device
    
    Returns:
        HeliosPipeline instance
    """
    return HeliosPipeline(
        graphormer_checkpoint=graphormer_checkpoint,
        rag_dir=rag_dir,
        llm_model=llm_model,
        use_api=use_api,
        api_key=api_key,
        device=device
    )


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Helios Complete Pipeline')
    parser.add_argument('--graphormer', type=str,
                       help='Graphormer checkpoint path')
    parser.add_argument('--rag-dir', type=str, default='data/rag',
                       help='RAG directory')
    parser.add_argument('--llm', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                       help='LLM model')
    parser.add_argument('--api', action='store_true',
                       help='Use API')
    parser.add_argument('--mode', type=str, default='question',
                       choices=['question', 'molecule', 'suggest'],
                       help='Mode: question, molecule analysis, or suggest')
    parser.add_argument('--input', type=str, required=True,
                       help='Input (question or SMILES)')
    
    args = parser.parse_args()
    
    # Create pipeline
    print("Loading Helios pipeline...")
    pipeline = create_helios_pipeline(
        graphormer_checkpoint=args.graphormer,
        rag_dir=args.rag_dir,
        llm_model=args.llm,
        use_api=args.api
    )
    
    print("\n" + "="*70)
    
    if args.mode == 'question':
        print(f"Question: {args.input}\n")
        result = pipeline.answer_chemistry_question(args.input)
        print(f"Answer: {result['answer']}\n")
        if result['sources']:
            print("Sources:")
            for i, src in enumerate(result['sources'][:3], 1):
                print(f"  [{i}] {src['title']}")
    
    elif args.mode == 'molecule':
        print(f"Molecule: {args.input}\n")
        result = pipeline.analyze_molecule(args.input)
        if result['predictions']:
            print("Predictions:")
            for prop, val in result['predictions'].items():
                print(f"  {prop}: {val:.3f}")
        print(f"\nAnalysis:\n{result['interpretation']}")
    
    elif args.mode == 'suggest':
        print(f"Target: {args.input}\n")
        result = pipeline.suggest_molecules(args.input)
        print(f"Recommendations:\n{result['recommendation']}")
    
    print("="*70)
