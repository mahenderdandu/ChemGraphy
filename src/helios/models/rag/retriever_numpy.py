"""
RAG retriever implementation
Uses sentence transformers and numpy for chemistry document retrieval
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    Retrieval-Augmented Generation retriever for chemistry documents
    Uses numpy-based similarity search (efficient for <100K documents)
    """
    
    def __init__(
        self,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embeddings_path: Optional[str] = None,
        corpus_path: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        Args:
            encoder_name: Sentence transformer model name
            embeddings_path: Path to embeddings .npy file
            corpus_path: Path to corpus JSON file
            device: Device for encoder
        """
        self.device = device
        
        # Load encoder
        logger.info(f"Loading encoder: {encoder_name}")
        self.encoder = SentenceTransformer(encoder_name, device=device)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        logger.info(f"Encoder loaded, embedding dim: {self.embedding_dim}")
        
        # Initialize
        self.embeddings = None
        self.documents = []
        
        if embeddings_path and Path(embeddings_path).exists():
            self.load_embeddings(embeddings_path)
        
        if corpus_path and Path(corpus_path).exists():
            self.load_corpus(corpus_path)
    
    def load_embeddings(self, embeddings_path: str):
        """Load precomputed embeddings"""
        logger.info(f"Loading embeddings from {embeddings_path}")
        self.embeddings = np.load(embeddings_path)
        logger.info(f"Loaded embeddings: {self.embeddings.shape}")
    
    def load_corpus(self, corpus_path: str):
        """Load document corpus"""
        logger.info(f"Loading corpus from {corpus_path}")
        with open(corpus_path, 'r') as f:
            self.documents = json.load(f)
        logger.info(f"Loaded {len(self.documents)} documents")
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode query text
        
        Args:
            query: Query string
        
        Returns:
            Query embedding (normalized)
        """
        embedding = self.encoder.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype('float32')
        
        return embedding
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_type: Optional[str] = None,
        return_embeddings: bool = False
    ) -> Union[List[Dict], Tuple[List[Dict], np.ndarray]]:
        """
        Retrieve top-k relevant documents
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            filter_type: Optional filter by document type
            return_embeddings: If True, return embeddings along with documents
        
        Returns:
            If return_embeddings=False: List of retrieved documents with scores
            If return_embeddings=True: (documents, embeddings array [top_k, embedding_dim])
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded. Call load_embeddings() first.")
        
        if not self.documents:
            raise ValueError("Corpus not loaded. Call load_corpus() first.")
        
        # Encode query
        query_embedding = self.encode_query(query)
        
        # Compute cosine similarities (dot product since normalized)
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top-k indices
        if filter_type:
            # Filter by type first
            filtered_indices = [
                i for i, doc in enumerate(self.documents)
                if doc.get('type') == filter_type
            ]
            if filtered_indices:
                filtered_sims = similarities[filtered_indices]
                top_filtered = np.argsort(filtered_sims)[-top_k:][::-1]
                top_indices = [filtered_indices[i] for i in top_filtered]
            else:
                top_indices = []
        else:
            top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Build results
        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc['score'] = float(similarities[idx])
            results.append(doc)
        
        if return_embeddings:
            # Return both documents and their embeddings
            retrieved_embeddings = self.embeddings[top_indices]  # [top_k, embedding_dim]
            return results, retrieved_embeddings
        
        return results
    
    def retrieve_with_context(
        self,
        query: str,
        top_k: int = 5
    ) -> Tuple[List[Dict], str]:
        """
        Retrieve documents and format as context string
        
        Args:
            query: Query string
            top_k: Number of documents
        
        Returns:
            (retrieved_documents, formatted_context)
        """
        results = self.retrieve(query, top_k)
        
        # Format context
        context_parts = []
        for i, doc in enumerate(results, 1):
            title = doc.get('title', 'Unknown')
            content = doc.get('content', doc.get('text', ''))
            context_parts.append(f"[{i}] {title}")
            context_parts.append(f"    {content}")
            context_parts.append("")
        
        context = "\n".join(context_parts)
        
        return results, context


# Convenience function for loading retriever
def load_rag_retriever(rag_dir: str = "data/rag") -> RAGRetriever:
    """
    Load RAG retriever from directory
    
    Args:
        rag_dir: Directory containing embeddings and corpus
    
    Returns:
        Initialized RAGRetriever
    """
    rag_dir = Path(rag_dir)
    
    retriever = RAGRetriever(encoder_name="sentence-transformers/all-MiniLM-L6-v2")
    retriever.load_embeddings(rag_dir / "embeddings.npy")
    retriever.load_corpus(rag_dir / "chembl_corpus.json")
    
    return retriever


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Test RAG retrieval')
    parser.add_argument('--rag-dir', type=str, default='data/rag',
                       help='RAG directory')
    parser.add_argument('--query', type=str, 
                       help='Query to test')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of results')
    
    args = parser.parse_args()
    
    # Load retriever
    print("Loading RAG retriever...")
    retriever = load_rag_retriever(args.rag_dir)
    
    # Test queries
    if args.query:
        test_queries = [args.query]
    else:
        test_queries = [
            "What is aspirin?",
            "EGFR inhibitors for cancer",
            "kinase drug targets",
            "acetaminophen mechanism",
            "SMILES CC(=O)Oc1ccccc1C(=O)O"
        ]
    
    print("\n" + "="*60)
    print("Testing RAG Retrieval")
    print("="*60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results, context = retriever.retrieve_with_context(query, top_k=args.top_k)
        
        for i, result in enumerate(results, 1):
            title = result.get('title', 'Unknown')
            content = result.get('content', result.get('text', ''))
            score = result['score']
            if hasattr(score, 'item'):
                score = score.item()
            print(f"  [{i}] {title} (score: {float(score):.3f})")
            print(f"      {content[:100]}...")
