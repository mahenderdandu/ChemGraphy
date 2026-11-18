"""
Simple vector retrieval using numpy (no FAISS - better for M1 Mac with limited RAM)
"""

import json
import numpy as np
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_numpy_index(
    corpus_path: str,
    output_dir: str,
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 16
):
    """
    Build numpy-based index (no FAISS)
    
    Args:
        corpus_path: Path to corpus JSON
        output_dir: Directory to save embeddings
        encoder_name: Sentence transformer model
        batch_size: Encoding batch size
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load corpus
    logger.info(f"Loading corpus from {corpus_path}")
    with open(corpus_path, 'r') as f:
        documents = json.load(f)
    logger.info(f"Loaded {len(documents)} documents")
    
    # Load encoder
    logger.info(f"Loading encoder: {encoder_name}")
    encoder = SentenceTransformer(encoder_name, device="cpu")
    embedding_dim = encoder.get_sentence_embedding_dimension()
    logger.info(f"Embedding dimension: {embedding_dim}")
    
    # Encode all documents
    texts = [doc['text'] for doc in documents]
    embeddings_list = []
    
    logger.info(f"Encoding {len(texts)} documents...")
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_emb = encoder.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=batch_size,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )
        embeddings_list.append(batch_emb)
        
        if (i // batch_size + 1) % 100 == 0:
            logger.info(f"Encoded {i + len(batch)}/{len(texts)} documents")
    
    # Combine and save
    embeddings = np.vstack(embeddings_list).astype('float32')
    logger.info(f"Final embeddings shape: {embeddings.shape}")
    
    # Save embeddings as memory-mapped file for efficient loading
    embeddings_path = output_dir / "embeddings.npy"
    logger.info(f"Saving embeddings to {embeddings_path}")
    np.save(str(embeddings_path), embeddings)
    
    # Save metadata
    metadata = {
        'num_docs': len(documents),
        'embedding_dim': embedding_dim,
        'encoder': encoder_name
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    size_mb = embeddings_path.stat().st_size / 1024 / 1024
    logger.info(f"✅ Index built successfully!")
    logger.info(f"   Documents: {len(documents)}")
    logger.info(f"   Embeddings: {embeddings.shape}")
    logger.info(f"   Size: {size_mb:.2f} MB")
    
    return embeddings


class NumpyRetriever:
    """
    Simple retriever using numpy cosine similarity (no FAISS needed)
    """
    
    def __init__(self, embeddings_path: str, corpus_path: str):
        """
        Args:
            embeddings_path: Path to embeddings .npy file
            corpus_path: Path to corpus JSON
        """
        logger.info(f"Loading embeddings from {embeddings_path}")
        self.embeddings = np.load(embeddings_path)
        logger.info(f"Loaded embeddings: {self.embeddings.shape}")
        
        logger.info(f"Loading corpus from {corpus_path}")
        with open(corpus_path, 'r') as f:
            self.documents = json.load(f)
        logger.info(f"Loaded {len(self.documents)} documents")
        
        # Load encoder
        metadata_path = Path(embeddings_path).parent / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                encoder_name = metadata.get('encoder', 'sentence-transformers/all-MiniLM-L6-v2')
        else:
            encoder_name = 'sentence-transformers/all-MiniLM-L6-v2'
        
        logger.info(f"Loading encoder: {encoder_name}")
        self.encoder = SentenceTransformer(encoder_name, device="cpu")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top-k documents for query
        
        Args:
            query: Query string
            top_k: Number of results
        
        Returns:
            List of documents with scores
        """
        # Encode query
        query_emb = self.encoder.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Compute cosine similarities (dot product since normalized)
        similarities = np.dot(self.embeddings, query_emb)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Build results
        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc['score'] = float(similarities[idx])
            results.append(doc)
        
        return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Build numpy-based index')
    parser.add_argument('--corpus', type=str, required=True,
                       help='Path to corpus JSON file')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save index')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Encoding batch size')
    parser.add_argument('--test', action='store_true',
                       help='Test retrieval after building')
    
    args = parser.parse_args()
    
    # Build index
    build_numpy_index(
        args.corpus,
        args.output_dir,
        batch_size=args.batch_size
    )
    
    # Test retrieval
    if args.test:
        print("\n" + "="*60)
        print("Testing retrieval...")
        print("="*60)
        
        retriever = NumpyRetriever(
            str(Path(args.output_dir) / "embeddings.npy"),
            args.corpus
        )
        
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
