"""
Fusion layer implementation
Cross-attention mechanism for combining molecule, text, and RAG embeddings
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion layer
    
    Fuses three modalities:
    - Query embeddings (from user query via sentence-transformers)
    - Molecule embeddings (from Graphormer GNN)
    - RAG context embeddings (from retrieved passages)
    
    Uses cross-attention to let query attend to molecules and retrieved context.
    """
    
    def __init__(
        self,
        query_dim: int = 384,      # sentence-transformers dimension
        molecule_dim: int = 768,    # Graphormer embedding dimension
        rag_dim: int = 384,         # RAG passage embedding dimension
        hidden_dim: int = 512,      # Common projection dimension
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.query_dim = query_dim
        self.molecule_dim = molecule_dim
        self.rag_dim = rag_dim
        self.hidden_dim = hidden_dim
        
        # Project all modalities to common dimension
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.molecule_proj = nn.Linear(molecule_dim, hidden_dim)
        self.rag_proj = nn.Linear(rag_dim, hidden_dim)
        
        # Cross-attention: query attends to [molecule + rag context]
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Self-attention layers for refinement
        self.self_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.self_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
        self.self_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])
        
        logger.info(f"Initialized CrossAttentionFusion: query({query_dim}), mol({molecule_dim}), rag({rag_dim}) -> hidden({hidden_dim})")
    
    def forward(
        self,
        query_embedding: torch.Tensor,
        molecule_embedding: Optional[torch.Tensor] = None,
        rag_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through fusion layer
        
        Args:
            query_embedding: Query embedding [batch, query_dim] or [batch, 1, query_dim]
            molecule_embedding: Molecule graph embedding [batch, molecule_dim]
            rag_embeddings: RAG passage embeddings [batch, num_passages, rag_dim]
        
        Returns:
            fused_embedding: Fused representation [batch, hidden_dim]
            attention_weights: Dictionary of attention weights for visualization
        """
        batch_size = query_embedding.size(0)
        attention_weights = {}
        
        # Ensure query is [batch, 1, query_dim]
        if query_embedding.dim() == 2:
            query_embedding = query_embedding.unsqueeze(1)  # [batch, 1, query_dim]
        
        # Project query to common dimension
        query_proj = self.query_proj(query_embedding)  # [batch, 1, hidden_dim]
        
        # Build context: concatenate molecule and RAG embeddings
        context_list = []
        
        if molecule_embedding is not None:
            # Ensure molecule is [batch, 1, molecule_dim]
            if molecule_embedding.dim() == 2:
                molecule_embedding = molecule_embedding.unsqueeze(1)
            mol_proj = self.molecule_proj(molecule_embedding)  # [batch, 1, hidden_dim]
            context_list.append(mol_proj)
        
        if rag_embeddings is not None:
            # rag_embeddings is [batch, num_passages, rag_dim]
            rag_proj = self.rag_proj(rag_embeddings)  # [batch, num_passages, hidden_dim]
            context_list.append(rag_proj)
        
        if not context_list:
            # No context, just return projected query
            return query_proj.squeeze(1), attention_weights
        
        # Concatenate all context
        context = torch.cat(context_list, dim=1)  # [batch, num_context, hidden_dim]
        
        # Cross-attention: query attends to context (molecule + RAG)
        attended, cross_attn_weights = self.cross_attention(
            query=query_proj,
            key=context,
            value=context,
            need_weights=True,
            average_attn_weights=True
        )
        attention_weights['cross_attention'] = cross_attn_weights
        
        # Residual + norm
        query_proj = self.norm1(query_proj + attended)
        
        # FFN
        ffn_out = self.ffn(query_proj)
        fused = self.norm2(query_proj + ffn_out)
        
        # Self-attention refinement layers
        for i, (self_attn, self_ffn, self_norm) in enumerate(
            zip(self.self_attention_layers, self.self_ffns, self.self_norms)
        ):
            attn_out, self_attn_weights = self_attn(
                query=fused,
                key=fused,
                value=fused,
                need_weights=True
            )
            fused = self_norm(fused + attn_out)
            fused = fused + self_ffn(fused)
            attention_weights[f'self_attention_{i}'] = self_attn_weights
        
        # Return single vector [batch, hidden_dim]
        fused_embedding = fused.squeeze(1)
        
        return fused_embedding, attention_weights


def create_fusion_layer(
    query_dim: int = 384,
    molecule_dim: int = 768,
    rag_dim: int = 384,
    **kwargs
) -> CrossAttentionFusion:
    """
    Factory function to create fusion layer
    
    Args:
        query_dim: Query embedding dimension (sentence-transformers)
        molecule_dim: Molecule embedding dimension (Graphormer)
        rag_dim: RAG passage embedding dimension
        **kwargs: Additional arguments for CrossAttentionFusion
    
    Returns:
        Initialized fusion layer
    """
    return CrossAttentionFusion(
        query_dim=query_dim,
        molecule_dim=molecule_dim,
        rag_dim=rag_dim,
        **kwargs
    )


class HeliosFusionModel(nn.Module):
    """
    Complete Helios fusion model
    Combines Graphormer, RAG retriever, and LLM with fusion layer
    """
    
    def __init__(
        self,
        graphormer_model,
        rag_retriever,
        llm_model,
        fusion_config: Dict,
    ):
        super().__init__()
        
        self.graphormer = graphormer_model
        self.retriever = rag_retriever
        self.llm = llm_model
        self.fusion_layer = CrossAttentionFusion(**fusion_config)
        
        logger.info("Initialized HeliosFusionModel")
    
    def forward(
        self,
        query: str,
        molecule_smiles: Optional[str] = None,
        num_retrieved_passages: int = 5,
    ) -> Dict:
        """
        Complete forward pass through Helios
        
        Args:
            query: User query string
            molecule_smiles: SMILES string (if molecule query)
            num_retrieved_passages: Number of RAG passages to retrieve
        
        Returns:
            Dictionary with response and intermediate outputs
        """
        # 1. Encode query text
        # TODO: Encode query with text encoder
        text_embeddings = None  # [1, seq_len, text_dim]
        
        # 2. Process molecule (if present)
        molecule_embeddings = None
        molecule_predictions = None
        
        if molecule_smiles:
            # TODO: Convert SMILES to graph
            # TODO: Get embeddings from Graphormer
            pass
        
        # 3. Retrieve relevant passages
        rag_results = []
        rag_embeddings = None
        
        if self.retriever:
            rag_results = self.retriever.retrieve(query, top_k=num_retrieved_passages)
            # TODO: Encode retrieved passages
            pass
        
        # 4. Fuse modalities
        fused_embeddings, attention_weights = self.fusion_layer(
            text_embeddings=text_embeddings,
            molecule_embeddings=molecule_embeddings,
            rag_embeddings=rag_embeddings,
        )
        
        # 5. Generate response with LLM
        # TODO: Pass fused embeddings to LLM
        # response = self.llm.generate_with_fusion(fused_embeddings)
        response = None
        
        return {
            'response': response,
            'molecule_predictions': molecule_predictions,
            'molecule_embeddings': molecule_embeddings,
            'rag_results': rag_results,
            'attention_weights': attention_weights,
        }


# TODO: Add training loop
# TODO: Add multi-task loss functions
# TODO: Add grounding loss (force citations)
# TODO: Add evaluation metrics
