"""
RAG Retriever implementation
Uses sentence transformers and FAISS for chemistry document retrieval
"""

import torch
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
import faiss

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    Retrieval-Augmented Generation retriever for chemistry documents
    """
    
    def __init__(
        self,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: Optional[str] = None,
        corpus_path: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        Args:
            encoder_name: Sentence transformer model name
            index_path: Path to saved FAISS index
            corpus_path: Path to corpus JSON file
            device: Device for encoder
        """
        self.device = device
        
        # Load encoder
        logger.info(f"Loading encoder: {encoder_name}")
        self.encoder = SentenceTransformer(encoder_name, device=device)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        logger.info(f"Encoder loaded, embedding dim: {self.embedding_dim}")
        
        # Initialize index and corpus
        self.index = None
        self.documents = []
        
        if index_path and Path(index_path).exists():
            self.load_index(index_path)
        
        if corpus_path and Path(corpus_path).exists():
            self.load_corpus(corpus_path)
    
    def load_corpus(self, corpus_path: str):
        """Load document corpus"""
        logger.info(f"Loading corpus from {corpus_path}")
        with open(corpus_path, 'r') as f:
            self.documents = json.load(f)
        logger.info(f"Loaded {len(self.documents)} documents")
    
    def build_index(self, corpus_path: str, index_save_path: Optional[str] = None):
        """
        Build FAISS index from corpus
        
        Args:
            corpus_path: Path to corpus JSON
            index_save_path: Optional path to save index
        """
        # Load corpus
        self.load_corpus(corpus_path)
        
        logger.info("Encoding documents...")
        texts = [doc['text'] for doc in self.documents]
        
        # Encode in smaller batches for 8GB RAM
        batch_size = 16  # Reduced from 32
        embeddings = []
        
        logger.info(f"Encoding {len(texts)} documents in batches of {batch_size}...")
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.encoder.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=16
            )
            embeddings.append(batch_embeddings)
            
            if (i // batch_size) % 100 == 0:
                logger.info(f"Encoded {i}/{len(texts)} documents")
        
        embeddings = np.vstack(embeddings).astype('float32')
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        
        # Normalize for cosine similarity
        logger.info("Normalizing embeddings...")
        faiss.normalize_L2(embeddings)
        
        # Build FAISS index
        logger.info("Building FAISS index...")
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine similarity)
        
        # Add in smaller chunks to avoid memory issues
        chunk_size = 1000
        for i in range(0, len(embeddings), chunk_size):
            chunk = embeddings[i:i+chunk_size]
            self.index.add(chunk)
            if i % 10000 == 0:
                logger.info(f"Added {i}/{len(embeddings)} vectors to index")
        
        logger.info(f"Index built with {self.index.ntotal} vectors")
        
        # Save index
        if index_save_path:
            self.save_index(index_save_path)
    
    def save_index(self, path: str):
        """Save FAISS index"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        logger.info(f"Saved index to {path}")
    
    def load_index(self, path: str):
        """Load FAISS index"""
        self.index = faiss.read_index(str(path))
        logger.info(f"Loaded index with {self.index.ntotal} vectors from {path}")
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode query text
        
        Args:
            query: Query string
        
        Returns:
            Query embedding
        """
        embedding = self.encoder.encode(
            query,
            convert_to_numpy=True
        ).astype('float32')
        
        # Normalize
        faiss.normalize_L2(embedding.reshape(1, -1))
        
        return embedding
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve relevant documents
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            filter_type: Optional document type filter
        
        Returns:
            List of retrieved documents with scores
        """
        if self.index is None:
            raise ValueError("Index not loaded. Build or load an index first.")
        
        if not self.documents:
            raise ValueError("Corpus not loaded. Load corpus first.")
        
        # Encode query
        query_embedding = self.encode_query(query)
        
        # Search
        scores, indices = self.index.search(query_embedding.reshape(1, -1), top_k * 3)  # Get more for filtering
        scores = scores[0]
        indices = indices[0]
        
        # Get documents
        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.documents):
                continue
            
            doc = self.documents[idx].copy()
            doc['score'] = float(score)
            
            # Apply filter
            if filter_type and doc['type'] != filter_type:
                continue
            
            results.append(doc)
            
            if len(results) >= top_k:
                break
        
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
            context_parts.append(f"[{i}] {doc['title']}")
            context_parts.append(f"    {doc['text']}")
            context_parts.append("")
        
        context = "\n".join(context_parts)
        
        return results, context


def build_rag_index_from_chembl(
    chembl_db_path: str,
    output_dir: str = "data/rag",
    max_compounds: int = 10000,
    max_targets: int = 5000,
    max_mechanisms: int = 5000,
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    """
    Build RAG index from ChEMBL database
    
    Args:
        chembl_db_path: Path to ChEMBL SQLite database
        output_dir: Output directory for corpus and index
        max_compounds: Max compound documents
        max_targets: Max target documents
        max_mechanisms: Max mechanism documents
        encoder_name: Sentence transformer model
    """
    from helios.data.processors.chembl_rag_processor import ChEMBLRAGProcessor
    
    output_dir = Path(output_dir)
    corpus_path = output_dir / "chembl_corpus.json"
    index_path = output_dir / "chembl_index.faiss"
    
    # Step 1: Build corpus
    logger.info("Step 1: Building corpus from ChEMBL...")
    processor = ChEMBLRAGProcessor(chembl_db_path)
    processor.build_corpus(
        str(corpus_path),
        max_compounds=max_compounds,
        max_targets=max_targets,
        max_mechanisms=max_mechanisms
    )
    
    # Step 2: Build index
    logger.info("Step 2: Building FAISS index...")
    retriever = RAGRetriever(encoder_name=encoder_name, device="cpu")
    retriever.build_index(str(corpus_path), str(index_path))
    
    logger.info("RAG system ready!")
    logger.info(f"  - Corpus: {corpus_path}")
    logger.info(f"  - Index: {index_path}")
    
    return retriever


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Build RAG index from ChEMBL')
    parser.add_argument('--chembl-db', type=str, required=True,
                       help='Path to ChEMBL SQLite database')
    parser.add_argument('--output-dir', type=str, default='data/rag',
                       help='Output directory')
    parser.add_argument('--max-compounds', type=int, default=10000,
                       help='Max compound documents')
    parser.add_argument('--max-targets', type=int, default=5000,
                       help='Max target documents')
    parser.add_argument('--max-mechanisms', type=int, default=5000,
                       help='Max mechanism documents')
    
    args = parser.parse_args()
    
    retriever = build_rag_index_from_chembl(
        args.chembl_db,
        args.output_dir,
        args.max_compounds,
        args.max_targets,
        args.max_mechanisms
    )
    
    # Test retrieval
    print("\n" + "="*60)
    print("Testing retrieval...")
    print("="*60)
    
    test_queries = [
        "What is aspirin?",
        "EGFR inhibitors",
        "kinase targets"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve(query, top_k=3)
        for i, result in enumerate(results, 1):
            print(f"  [{i}] {result['title']} (score: {result['score']:.3f})")
            print(f"      {result['text'][:100]}...")
