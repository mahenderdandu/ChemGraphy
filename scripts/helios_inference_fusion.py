"""
Helios Inference with Fusion Layer
Complete pipeline: RAG + GNN + LLM with cross-attention fusion
"""

import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

from helios.inference.rag_llm_pipeline_fusion import FusionRAGLLMPipeline


def main():
    parser = argparse.ArgumentParser(description="Helios Inference with Fusion")
    parser.add_argument('--rag-dir', type=str, default='data/rag',
                        help='Directory with RAG corpus and embeddings')
    parser.add_argument('--graphormer', type=str, default=None,
                        help='Path to Graphormer checkpoint (optional)')
    parser.add_argument('--llm', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                        help='LLM model name')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu/cuda/mps)')
    parser.add_argument('--no-fusion', action='store_true',
                        help='Disable fusion layer (use simple concatenation)')
    
    # Query parameters
    parser.add_argument('--question', type=str, required=True,
                        help='Chemistry question')
    parser.add_argument('--smiles', type=str, default=None,
                        help='Optional molecule SMILES')
    parser.add_argument('--num-results', type=int, default=5,
                        help='Number of RAG results')
    parser.add_argument('--max-tokens', type=int, default=512,
                        help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Helios Chemistry AI - Fusion Pipeline")
    print("="*70)
    
    # Initialize pipeline
    logger.info("Loading Helios pipeline with fusion...")
    pipeline = FusionRAGLLMPipeline(
        rag_dir=args.rag_dir,
        llm_model_name=args.llm,
        graphormer_checkpoint=args.graphormer,
        device=args.device,
        use_fusion=not args.no_fusion
    )
    
    print(f"\nQuestion: {args.question}")
    if args.smiles:
        print(f"Molecule: {args.smiles}")
    print()
    
    # Run inference
    result = pipeline.query(
        question=args.question,
        smiles=args.smiles,
        num_results=args.num_results,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    # Display results
    print("="*70)
    print("ANSWER:")
    print("="*70)
    print(result['answer'])
    print()
    
    if result['molecule_prediction'] is not None:
        print("="*70)
        print("MOLECULE ANALYSIS:")
        print("="*70)
        print(f"SMILES: {result['smiles']}")
        print(f"Predicted Property: {result['molecule_prediction']:.4f}")
        if result['molecule_embedding'] is not None:
            print(f"Graph Embedding Shape: {result['molecule_embedding'].shape}")
        print()
    
    print("="*70)
    print("RETRIEVED SOURCES:")
    print("="*70)
    for i, doc in enumerate(result['sources'], 1):
        title = doc.get('title', 'Unknown')
        score = doc.get('score', 0.0)
        print(f"[{i}] {title} (score: {score:.3f})")
    print()
    
    if result['attention_weights']:
        print("="*70)
        print("FUSION LAYER ATTENTION:")
        print("="*70)
        for name, weights in result['attention_weights'].items():
            print(f"{name}: shape {weights.shape}")
        print()
    
    print("="*70)


if __name__ == "__main__":
    main()
