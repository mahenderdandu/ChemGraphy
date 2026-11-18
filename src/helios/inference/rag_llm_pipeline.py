"""
RAG-LLM Integration Pipeline
Combines retrieval with language model generation
"""

import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from helios.models.rag.retriever_numpy import RAGRetriever, load_rag_retriever
from helios.models.llm.model import HeliosLLM, create_chemistry_llm

logger = logging.getLogger(__name__)


class RAGLLMPipeline:
    """
    Complete RAG-LLM pipeline for chemistry question answering
    """
    
    def __init__(
        self,
        retriever: RAGRetriever,
        llm: HeliosLLM,
        top_k: int = 5,
        include_citations: bool = True
    ):
        """
        Args:
            retriever: RAG retriever instance
            llm: LLM instance
            top_k: Number of documents to retrieve
            include_citations: Include source citations in response
        """
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.include_citations = include_citations
        
        logger.info("RAG-LLM pipeline initialized")
    
    def _build_prompt(
        self,
        query: str,
        context_docs: List[Dict],
        query_type: str = "general"
    ) -> Tuple[str, str]:
        """
        Build prompt with retrieved context
        
        Args:
            query: User query
            context_docs: Retrieved documents
            query_type: Type of query for specialized prompts
        
        Returns:
            (system_prompt, user_prompt)
        """
        # System prompt
        system_prompt = """You are Helios, an expert chemistry and drug discovery assistant. You provide accurate, scientific answers based on retrieved chemical databases and research literature.

Guidelines:
- Answer questions using the provided context
- Be precise and scientific in your explanations
- If context is insufficient, acknowledge limitations
- For drug/compound queries, mention key properties (mechanism, structure, etc.)
- For SMILES queries, explain the molecular structure
- Always cite sources using [1], [2], etc. when referencing specific compounds or facts"""

        # Format context
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            context_parts.append(f"[{i}] {doc['title']}")
            context_parts.append(f"    {doc['text']}")
            context_parts.append("")
        
        context_str = "\n".join(context_parts)
        
        # User prompt with context
        user_prompt = f"""Context from ChEMBL database:
{context_str}

Question: {query}

Please answer the question based on the above context. Be specific and cite your sources."""
        
        return system_prompt, user_prompt
    
    def query(
        self,
        question: str,
        query_type: str = "general",
        max_new_tokens: int = 300,
        temperature: float = 0.7
    ) -> Dict:
        """
        Answer question using RAG-LLM pipeline
        
        Args:
            question: User question
            query_type: Type of query (general, drug, mechanism, etc.)
            max_new_tokens: Max tokens for LLM
            temperature: LLM temperature
        
        Returns:
            {
                'question': str,
                'answer': str,
                'sources': List[Dict],
                'context': str
            }
        """
        logger.info(f"Processing query: {question}")
        
        # Step 1: Retrieve relevant documents
        logger.info("Retrieving relevant documents...")
        retrieved_docs, context = self.retriever.retrieve_with_context(
            question,
            top_k=self.top_k
        )
        
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        
        # Step 2: Build prompt
        system_prompt, user_prompt = self._build_prompt(
            question,
            retrieved_docs,
            query_type
        )
        
        # Step 3: Generate answer
        logger.info("Generating answer...")
        answer = self.llm.generate(
            user_prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        # Step 4: Format response
        response = {
            'question': question,
            'answer': answer,
            'sources': retrieved_docs if self.include_citations else [],
            'context': context
        }
        
        return response
    
    def answer_with_property_prediction(
        self,
        smiles: str,
        property_values: Dict[str, float],
        question: Optional[str] = None
    ) -> Dict:
        """
        Interpret property predictions from Graphormer
        
        Args:
            smiles: SMILES string
            property_values: Predicted properties {'solubility': 1.23, ...}
            question: Optional specific question about the molecule
        
        Returns:
            Response dict with interpretation
        """
        # Build context from predictions
        prediction_text = f"Molecule: {smiles}\n\nPredicted properties:\n"
        for prop, value in property_values.items():
            val = value.item() if hasattr(value, 'item') else value
            prediction_text += f"- {prop}: {float(val):.2f}\n"
        
        # Retrieve similar molecules
        retrieved_docs, _ = self.retriever.retrieve_with_context(
            f"SMILES {smiles}",
            top_k=3
        )
        
        # Build prompt
        if question:
            prompt = f"{prediction_text}\n\nBased on these predictions and the molecular structure, please answer: {question}"
        else:
            prompt = f"{prediction_text}\n\nPlease interpret these molecular properties and explain what they mean for drug development or applications."
        
        system_prompt = """You are a chemistry expert explaining molecular property predictions. 
Provide clear, practical interpretations of numerical predictions in the context of drug development, 
safety, and chemical applications."""
        
        answer = self.llm.generate(
            prompt,
            system_prompt=system_prompt,
            max_new_tokens=250,
            temperature=0.7
        )
        
        return {
            'smiles': smiles,
            'predictions': property_values,
            'interpretation': answer,
            'similar_molecules': retrieved_docs[:3]
        }
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        retrieve_context: bool = True
    ) -> str:
        """
        Multi-turn chat with optional context retrieval
        
        Args:
            messages: Conversation history
            retrieve_context: Whether to retrieve context for latest message
        
        Returns:
            Assistant response
        """
        if retrieve_context and messages:
            # Get latest user message
            latest_user_msg = None
            for msg in reversed(messages):
                if msg["role"] == "user":
                    latest_user_msg = msg["content"]
                    break
            
            if latest_user_msg:
                # Retrieve and add context
                retrieved_docs, context = self.retriever.retrieve_with_context(
                    latest_user_msg,
                    top_k=3
                )
                
                # Prepend context to message
                context_msg = f"Relevant context:\n{context}\n\nUser question: {latest_user_msg}"
                messages = messages[:-1] + [{"role": "user", "content": context_msg}]
        
        return self.llm.chat(messages)


def create_rag_llm_pipeline(
    rag_dir: str = "data/rag",
    model_name: Optional[str] = None,
    use_api: bool = False,
    api_key: Optional[str] = None,
    device: str = "cpu",
    top_k: int = 5
) -> RAGLLMPipeline:
    """
    Create complete RAG-LLM pipeline with defaults
    
    Args:
        rag_dir: Directory with RAG index
        model_name: LLM model name
        use_api: Use API instead of local model
        api_key: API key
        device: Device for local models
        top_k: Documents to retrieve
    
    Returns:
        RAGLLMPipeline instance
    """
    logger.info("Creating RAG-LLM pipeline...")
    
    # Load retriever
    logger.info(f"Loading RAG retriever from {rag_dir}")
    retriever = load_rag_retriever(rag_dir)
    
    # Load LLM
    logger.info("Loading LLM...")
    llm = create_chemistry_llm(
        model_name=model_name,
        use_api=use_api,
        api_key=api_key,
        device=device
    )
    
    # Create pipeline
    pipeline = RAGLLMPipeline(
        retriever=retriever,
        llm=llm,
        top_k=top_k
    )
    
    logger.info("Pipeline ready!")
    return pipeline


if __name__ == "__main__":
    import argparse
    import json
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Test RAG-LLM Pipeline')
    parser.add_argument('--rag-dir', type=str, default='data/rag',
                       help='RAG directory')
    parser.add_argument('--model', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                       help='LLM model name')
    parser.add_argument('--api', action='store_true',
                       help='Use API model')
    parser.add_argument('--query', type=str,
                       help='Question to ask')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Documents to retrieve')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = create_rag_llm_pipeline(
        rag_dir=args.rag_dir,
        model_name=args.model,
        use_api=args.api,
        top_k=args.top_k
    )
    
    # Test queries
    if args.query:
        test_queries = [args.query]
    else:
        test_queries = [
            "What is the mechanism of action of aspirin?",
            "Suggest EGFR inhibitors for lung cancer treatment",
            "What are the key properties of acetaminophen?"
        ]
    
    print("\n" + "="*70)
    print("RAG-LLM Pipeline Test")
    print("="*70)
    
    for query in test_queries:
        print(f"\nQuestion: {query}\n")
        
        response = pipeline.query(query, max_new_tokens=200)
        
        print(f"Answer: {response['answer']}\n")
        
        if response['sources']:
            print("Sources:")
            for i, source in enumerate(response['sources'][:3], 1):
                score = source['score']
                if hasattr(score, 'item'):
                    score = score.item()
                print(f"  [{i}] {source['title']} (score: {float(score):.3f})")
        
        print("-" * 70)
