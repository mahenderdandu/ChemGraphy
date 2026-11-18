"""
Build FAISS index from existing corpus with memory-efficient processing
"""

import json
import numpy as np
import faiss
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_index_incremental(
    corpus_path: str,
    index_path: str,
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 8,
    process_every: int = 2000
):
    """
    Build FAISS index incrementally with very small chunks for M1 Mac
    
    Args:
        corpus_path: Path to corpus JSON
        index_path: Path to save index
        encoder_name: Sentence transformer model
        batch_size: Encoding batch size (small for memory)
        process_every: Process this many documents at once
    """
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
    
    # Initialize FAISS index
    logger.info("Initializing FAISS index...")
    index = faiss.IndexFlatIP(embedding_dim)
    
    # Process in very small chunks
    texts = [doc['text'] for doc in documents]
    total = len(texts)
    
    for start_idx in range(0, total, process_every):
        end_idx = min(start_idx + process_every, total)
        chunk_texts = texts[start_idx:end_idx]
        
        logger.info(f"Processing documents {start_idx}-{end_idx} ({end_idx}/{total})")
        
        # Encode in tiny batches
        chunk_embeddings = []
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i:i+batch_size]
            batch_emb = encoder.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=batch_size,
                normalize_embeddings=True
            )
            chunk_embeddings.append(batch_emb)
        
        # Combine chunk
        chunk_embeddings = np.vstack(chunk_embeddings).astype('float32')
        
        # Add to index in tiny sub-chunks (100 at a time to avoid crash)
        sub_chunk_size = 100
        for i in range(0, len(chunk_embeddings), sub_chunk_size):
            sub_chunk = chunk_embeddings[i:i+sub_chunk_size]
            try:
                index.add(sub_chunk)
            except Exception as e:
                logger.error(f"Failed to add vectors {i}-{i+len(sub_chunk)}: {e}")
                logger.info("Falling back to one-by-one addition")
                for vec in sub_chunk:
                    index.add(vec.reshape(1, -1))
        
        logger.info(f"Added {end_idx} vectors to index (total: {index.ntotal})")
        
        # Clear memory aggressively
        del chunk_embeddings
        gc.collect()
        
        # Save checkpoint every 10K
        if end_idx % 10000 == 0 or end_idx == total:
            checkpoint_path = str(index_path).replace('.faiss', f'_checkpoint_{end_idx}.faiss')
            logger.info(f"Saving checkpoint to {checkpoint_path}")
            try:
                faiss.write_index(index, checkpoint_path)
            except Exception as e:
                logger.warning(f"Failed to save checkpoint: {e}")
    
    # Save final index
    logger.info(f"Saving index to {index_path}")
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    
    logger.info(f"✅ Index built successfully with {index.ntotal} vectors")
    logger.info(f"Index size: {Path(index_path).stat().st_size / 1024 / 1024:.2f} MB")
    
    return index


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Build FAISS index from corpus')
    parser.add_argument('--corpus', type=str, required=True,
                       help='Path to corpus JSON file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save FAISS index')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Encoding batch size')
    parser.add_argument('--process-every', type=int, default=2000,
                       help='Process this many documents at a time')
    
    args = parser.parse_args()
    
    build_index_incremental(
        args.corpus,
        args.output,
        batch_size=args.batch_size,
        process_every=args.process_every
    )
