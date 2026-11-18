"""
Helios Chatbot - Complete Gradio UI with Fusion Pipeline
Supports chat, molecule visualization, reaction chemistry, and attention heatmaps
"""

import gradio as gr
from typing import List, Tuple, Optional, Dict, Any
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import io
from PIL import Image

from helios.inference.rag_llm_pipeline_fusion import FusionRAGLLMPipeline
from helios.utils.reaction_chemistry import (
    get_reaction_processor,
    visualize_molecule_with_properties,
    check_lipinski_rules,
    check_veber_rules,
    ReactionResult,
    COMMON_REACTIONS
)
from helios.utils.chemical_extractor import (
    extract_chemicals_from_text,
    generate_molecule_image_base64,
)

logger = logging.getLogger(__name__)


class HeliosChatInterface:
    """Complete Helios chatbot with visualization and reaction chemistry"""
    
    def __init__(
        self,
        rag_dir: str = "data/rag",
        llm_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        graphormer_checkpoint: Optional[str] = "models/checkpoints/graphormer/best_model.pt",
        device: str = "cpu"
    ):
        """Initialize chatbot with fusion pipeline and reaction processor"""
        logger.info("Initializing Helios Chatbot...")
        
        self.pipeline = FusionRAGLLMPipeline(
            rag_dir=rag_dir,
            llm_model_name=llm_model,
            graphormer_checkpoint=graphormer_checkpoint,
            device=device,
            use_fusion=True
        )
        
        # Initialize reaction processor
        self.reaction_processor = get_reaction_processor()
        
        logger.info("Helios Chatbot ready!")
    
    def process_query(
        self,
        question: str,
        smiles: str,
        history: List[Dict[str, str]],
        num_results: int = 5,
        temperature: float = 0.7,
        reasoning_mode: str = "verify",
        num_candidates: int = 3,
        add_citations: bool = True
    ) -> Tuple[List[Dict[str, str]], str, str, Any]:
        """
        Process user query
        
        Returns:
            (updated_history, molecule_info, sources_text, attention_plot)
        """
        if not question.strip():
            return history, "", "", None
        
        try:
            # Clean SMILES input
            smiles_input = smiles.strip() if smiles and smiles.strip() else None
            
            # Run pipeline
            result = self.pipeline.query(
                question=question,
                smiles=smiles_input,
                num_results=num_results,
                max_tokens=512,
                temperature=temperature,
                reasoning_mode=reasoning_mode,
                num_candidates=num_candidates,
                add_citations=add_citations
            )
            
            # Format answer with inline molecule visualizations
            answer = result['answer']
            
            # Append inline molecule info to the answer text
            inline_molecules = result.get('inline_molecules', [])
            if inline_molecules:
                answer += "\n\n---\n**Detected Molecules:**\n"
                for mol in inline_molecules:
                    mol_name = mol.get('name', mol.get('smiles', ''))
                    mol_smiles = mol.get('smiles', '')
                    mol_pred = mol.get('gnn_prediction')
                    answer += f"\n🧪 **{mol_name}** (`{mol_smiles}`)"
                    if mol_pred is not None:
                        answer += f" — GNN Prediction: {mol_pred:.4f}"
                    answer += "\n"
            
            # Add to history (Gradio 6.0 messages format)
            if smiles_input:
                user_msg = f"**Q:** {question}\n**Molecule:** `{smiles_input}`"
            else:
                user_msg = question
            
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": answer})
            
            # Format molecule info
            mol_info = ""
            if result['molecule_prediction'] is not None:
                # Convert tensor to float if needed
                pred_val = result['molecule_prediction']
                if hasattr(pred_val, 'item'):
                    pred_val = pred_val.item()
                mol_info = f"""### Molecule Analysis
**SMILES:** `{result['smiles']}`  
**Predicted Property:** {float(pred_val):.4f}  
**Embedding Shape:** {result['molecule_embedding'].shape if result['molecule_embedding'] is not None else 'N/A'}
"""
            
            # Format sources
            sources_text = "### Retrieved Sources\n\n"
            for i, doc in enumerate(result['sources'], 1):
                title = doc.get('title', 'Unknown')
                score = doc.get('score', 0.0)
                if hasattr(score, 'item'):
                    score = score.item()
                content = doc.get('content', '')[:200]
                sources_text += f"**[{i}] {title}** (score: {float(score):.3f})\n{content}...\n\n"
            
            # Create attention heatmap
            attention_plot = None
            if result['attention_weights'] and 'cross_attention' in result['attention_weights']:
                attention_plot = self._plot_attention(result['attention_weights'])
            
            return history, mol_info, sources_text, attention_plot
            
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            error_msg = f"❌ Error: {str(e)}"
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": error_msg})
            return history, "", "", None
    
    def _plot_attention(self, attention_weights: Dict[str, np.ndarray]) -> Image.Image:
        """Create attention heatmap visualization"""
        try:
            # Get cross-attention weights
            cross_attn = attention_weights.get('cross_attention')
            if cross_attn is None:
                return None
            
            # cross_attn shape: [batch=1, num_heads or 1, num_context]
            if cross_attn.ndim == 3:
                cross_attn = cross_attn[0]  # Remove batch dim
            
            # Average over heads if present
            if cross_attn.ndim == 2:
                cross_attn = cross_attn.mean(axis=0)  # [num_context]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 2))
            
            # Plot heatmap
            im = ax.imshow(cross_attn.reshape(1, -1), cmap='viridis', aspect='auto')
            
            # Labels
            ax.set_yticks([0])
            ax.set_yticklabels(['Query'])
            ax.set_xticks(range(len(cross_attn)))
            
            # Create labels for context (molecule + RAG passages)
            labels = ['Molecule'] + [f'RAG-{i}' for i in range(1, len(cross_attn))]
            ax.set_xticklabels(labels, rotation=45, ha='right')
            
            ax.set_title('Cross-Attention: Query attending to [Molecule + RAG Context]')
            plt.colorbar(im, ax=ax, label='Attention Weight')
            plt.tight_layout()
            
            # Convert to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            plt.close(fig)
            
            return img
            
        except Exception as e:
            logger.error(f"Error plotting attention: {e}")
            return None
    
    def visualize_molecule(self, smiles: str) -> Optional[Image.Image]:
        """
        Visualize molecule structure
        
        Args:
            smiles: SMILES string
        
        Returns:
            PIL Image of molecule or None
        """
        if not smiles or not smiles.strip():
            return None
        
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
            
            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is None:
                return None
            
            img = Draw.MolToImage(mol, size=(400, 400))
            return img
            
        except Exception as e:
            logger.error(f"Error visualizing molecule: {e}")
            return None
    
    def analyze_molecule_full(self, smiles: str) -> Tuple[Optional[Image.Image], str, str, str]:
        """
        Full molecule analysis with properties and drug-likeness
        
        Args:
            smiles: SMILES string
            
        Returns:
            (image, properties_text, lipinski_text, veber_text)
        """
        if not smiles or not smiles.strip():
            return None, "", "", ""
        
        try:
            img, props = visualize_molecule_with_properties(smiles.strip())
            if img is None:
                return None, "❌ Invalid SMILES", "", ""
            
            # Format properties
            props_text = "### Molecular Properties\n\n"
            props_text += f"**SMILES:** `{smiles}`\n\n"
            props_text += "| Property | Value |\n|----------|-------|\n"
            for key, value in props.items():
                display_key = key.replace('_', ' ').title()
                props_text += f"| {display_key} | {value} |\n"
            
            # Check Lipinski
            lipinski = check_lipinski_rules(smiles.strip())
            if lipinski.get('valid'):
                lip_text = "### Lipinski's Rule of Five\n\n"
                lip_text += f"**Status:** {lipinski['message']}\n\n"
                lip_text += "| Rule | Value | Limit | Pass |\n|------|-------|-------|------|\n"
                for rule, data in lipinski['rules'].items():
                    status = "✅" if data['pass'] else "❌"
                    lip_text += f"| {rule.replace('_', ' ').title()} | {data['value']:.2f} | ≤{data['limit']} | {status} |\n"
            else:
                lip_text = f"❌ Error: {lipinski.get('error', 'Unknown')}"
            
            # Check Veber
            veber = check_veber_rules(smiles.strip())
            if veber.get('valid'):
                veb_text = "### Veber Rules (Oral Bioavailability)\n\n"
                veb_text += f"**Status:** {veber['message']}\n\n"
                veb_text += "| Rule | Value | Limit | Pass |\n|------|-------|-------|------|\n"
                for rule, data in veber['rules'].items():
                    status = "✅" if data['pass'] else "❌"
                    veb_text += f"| {rule.replace('_', ' ').title()} | {data['value']:.2f} | ≤{data['limit']} | {status} |\n"
            else:
                veb_text = f"❌ Error: {veber.get('error', 'Unknown')}"
            
            return img, props_text, lip_text, veb_text
            
        except Exception as e:
            logger.error(f"Error in full molecule analysis: {e}")
            return None, f"❌ Error: {str(e)}", "", ""
    
    def visualize_reaction(self, reaction_smiles: str) -> Tuple[Optional[Image.Image], str]:
        """
        Visualize a reaction and analyze its components
        
        Args:
            reaction_smiles: Reaction SMILES (e.g., "A.B>>C.D")
            
        Returns:
            (reaction_image, analysis_text)
        """
        if not reaction_smiles or not reaction_smiles.strip():
            return None, ""
        
        try:
            # Visualize reaction
            img = self.reaction_processor.visualize_reaction(reaction_smiles.strip())
            
            # Analyze components
            analysis = self.reaction_processor.analyze_reaction_components(reaction_smiles.strip())
            
            if analysis is None:
                return img, "Could not parse reaction components"
            
            text = "### Reaction Analysis\n\n"
            
            # Reactants
            text += "#### Reactants\n"
            for i, r in enumerate(analysis['reactants'], 1):
                text += f"\n**Reactant {i}:** `{r['smiles']}`\n"
                props = r['properties']
                text += f"- MW: {props.get('molecular_weight', 0):.2f}, LogP: {props.get('logp', 0):.2f}\n"
            
            # Products
            text += "\n#### Products\n"
            for i, p in enumerate(analysis['products'], 1):
                text += f"\n**Product {i}:** `{p['smiles']}`\n"
                props = p['properties']
                text += f"- MW: {props.get('molecular_weight', 0):.2f}, LogP: {props.get('logp', 0):.2f}\n"
            
            # Deltas
            text += f"\n#### Changes\n"
            text += f"- ΔMW: {analysis['delta_mw']:+.2f}\n"
            text += f"- ΔLogP: {analysis['delta_logp']:+.2f}\n"
            
            return img, text
            
        except Exception as e:
            logger.error(f"Error visualizing reaction: {e}")
            return None, f"❌ Error: {str(e)}"
    
    def run_reaction_prediction(
        self, 
        reactants: str, 
        reaction_type: str,
        custom_smirks: str
    ) -> Tuple[Optional[Image.Image], str, str]:
        """
        Run a reaction prediction
        
        Args:
            reactants: Dot-separated reactant SMILES
            reaction_type: Selected reaction template or "custom"
            custom_smirks: Custom SMIRKS pattern (if reaction_type is "custom")
            
        Returns:
            (product_image, products_text, details_text)
        """
        if not reactants or not reactants.strip():
            return None, "", "Please enter reactant SMILES"
        
        try:
            reactant_list = [r.strip() for r in reactants.split(".") if r.strip()]
            
            # Determine SMIRKS to use
            if reaction_type == "custom":
                if not custom_smirks or not custom_smirks.strip():
                    return None, "", "Please enter a custom SMIRKS pattern"
                smirks = custom_smirks.strip()
                rxn_name = "Custom Reaction"
            else:
                if reaction_type not in COMMON_REACTIONS:
                    return None, "", f"Unknown reaction type: {reaction_type}"
                rxn_info = COMMON_REACTIONS[reaction_type]
                smirks = rxn_info['smirks']
                rxn_name = rxn_info['name']
            
            # Run reaction
            result = self.reaction_processor.run_reaction(reactant_list, smirks)
            
            if not result.success:
                return None, "", f"❌ {result.message}"
            
            # Format products
            products_text = f"### {rxn_name} Results\n\n"
            products_text += f"**Reactants:** `{reactants}`\n\n"
            products_text += f"**{result.message}**\n\n"
            
            for i, product_set in enumerate(result.product_sets, 1):
                products_text += f"#### Product Set {i}\n"
                for j, p in enumerate(product_set, 1):
                    products_text += f"- Product {j}: `{p}`\n"
                products_text += "\n"
            
            # Details with properties
            details_text = "### Product Properties\n\n"
            if result.product_properties:
                for i, props in enumerate(result.product_properties, 1):
                    details_text += f"**Product {i}:**\n"
                    details_text += f"- MW: {props.get('molecular_weight', 0):.2f}\n"
                    details_text += f"- LogP: {props.get('logp', 0):.2f}\n"
                    details_text += f"- HBD: {props.get('hbd', 0)}, HBA: {props.get('hba', 0)}\n"
                    details_text += f"- TPSA: {props.get('tpsa', 0):.2f}\n\n"
            
            # Create combined product image
            product_img = None
            if result.product_images:
                product_img = result.product_images[0]  # First product
            
            return product_img, products_text, details_text
            
        except Exception as e:
            logger.error(f"Error in reaction prediction: {e}")
            return None, "", f"❌ Error: {str(e)}"
    
    def suggest_reactions(self, reactants: str) -> str:
        """
        Suggest possible reactions based on functional groups
        
        Args:
            reactants: Dot-separated reactant SMILES
            
        Returns:
            Suggestions text
        """
        if not reactants or not reactants.strip():
            return "Please enter reactant SMILES"
        
        try:
            reactant_list = [r.strip() for r in reactants.split(".") if r.strip()]
            suggestions = self.reaction_processor.suggest_reaction(reactant_list)
            
            if not suggestions:
                return "No common reactions suggested for these reactants."
            
            text = "### Suggested Reactions\n\n"
            text += "Based on functional group analysis:\n\n"
            for key, desc, confidence in suggestions:
                rxn_info = COMMON_REACTIONS.get(key, {})
                text += f"**{rxn_info.get('name', key)}** (confidence: {confidence:.0%})\n"
                text += f"- {desc}\n"
                text += f"- SMIRKS: `{rxn_info.get('smirks', 'N/A')}`\n\n"
            
            return text
            
        except Exception as e:
            logger.error(f"Error suggesting reactions: {e}")
            return f"❌ Error: {str(e)}"
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface with chemistry tools"""
        
        # Build reaction type choices
        reaction_choices = ["custom"] + list(COMMON_REACTIONS.keys())
        reaction_labels = {k: v['name'] for k, v in COMMON_REACTIONS.items()}
        
        with gr.Blocks(title="Helios Chemistry AI") as interface:
            
            gr.Markdown("""
            # 🔬 Helios - Chemistry AI Assistant
            
            **Multimodal AI for Chemistry**: Ask questions, analyze molecules, predict reactions, get grounded answers with citations.
            
            **Powered by**: Graphormer (GNN) + RAG (58K documents) + Cross-Attention Fusion + LLM + RDKit Chemistry Engine
            """)
            
            with gr.Tabs():
                # ============ TAB 1: AI CHAT ============
                with gr.Tab("💬 AI Chat", id="chat"):
                    with gr.Row():
                        # Left column: Chat
                        with gr.Column(scale=2):
                            chatbot = gr.Chatbot(
                                label="Chat History",
                                height=500
                            )
                            
                            with gr.Row():
                                question_input = gr.Textbox(
                                    label="Your Question",
                                    placeholder="e.g., What is the mechanism of action of aspirin?",
                                    lines=2,
                                    scale=4
                                )
                                submit_btn = gr.Button("Submit", variant="primary", scale=1)
                            
                            with gr.Row():
                                chat_smiles_input = gr.Textbox(
                                    label="Molecule SMILES (optional)",
                                    placeholder="e.g., CC(=O)Oc1ccccc1C(=O)O",
                                    lines=1
                                )
                            
                            with gr.Accordion("Advanced Settings", open=False):
                                num_results = gr.Slider(
                                    minimum=1, maximum=10, value=5, step=1,
                                    label="Number of RAG documents to retrieve"
                                )
                                temperature = gr.Slider(
                                    minimum=0.0, maximum=1.0, value=0.7, step=0.1,
                                    label="LLM Temperature"
                                )
                                reasoning_mode = gr.Dropdown(
                                    label="Reasoning Mode",
                                    choices=["standard", "verify", "self_consistency"],
                                    value="verify"
                                )
                                num_candidates = gr.Slider(
                                    minimum=1, maximum=7, value=3, step=1,
                                    label="Self-Consistency Candidates"
                                )
                                add_citations = gr.Checkbox(
                                    label="Add Inline Citations",
                                    value=True
                                )
                            
                            with gr.Row():
                                clear_btn = gr.Button("Clear Chat")
                        
                        # Right column: Visualizations
                        with gr.Column(scale=1):
                            gr.Markdown("### 📊 Visualizations")
                            
                            with gr.Tab("Molecule"):
                                mol_structure = gr.Image(
                                    label="Molecule Structure",
                                    type="pil",
                                    height=300
                                )
                                mol_info = gr.Markdown(label="Analysis")
                            
                            with gr.Tab("Attention"):
                                attention_plot = gr.Image(
                                    label="Cross-Attention Heatmap",
                                    type="pil",
                                    height=200
                                )
                                gr.Markdown("*Query attending to Molecule + RAG context*")
                            
                            with gr.Tab("Sources"):
                                sources_display = gr.Markdown(
                                    label="Retrieved Documents",
                                    value="No sources yet"
                                )
                    
                    # Examples
                    gr.Examples(
                        examples=[
                            ["What is the mechanism of action of aspirin?", "CC(=O)Oc1ccccc1C(=O)O"],
                            ["What are EGFR inhibitors used for?", ""],
                            ["Explain the structure-activity relationship of kinase inhibitors", ""],
                            ["What is the solubility of this molecule?", "CCO"],
                            ["Tell me about acetaminophen", "CC(=O)Nc1ccc(O)cc1"],
                        ],
                        inputs=[question_input, chat_smiles_input],
                        label="Example Queries"
                    )
                
                # ============ TAB 2: MOLECULE ANALYZER ============
                with gr.Tab("🧬 Molecule Analyzer", id="molecule"):
                    gr.Markdown("""
                    ### Single Molecule Analysis
                    Enter a SMILES string to visualize the molecule and calculate its properties,
                    including drug-likeness assessment (Lipinski, Veber rules).
                    """)
                    
                    with gr.Row():
                        mol_smiles_input = gr.Textbox(
                            label="SMILES String",
                            placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O (Aspirin)",
                            lines=1,
                            scale=4
                        )
                        analyze_mol_btn = gr.Button("Analyze Molecule", variant="primary", scale=1)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            mol_image = gr.Image(
                                label="Molecule Structure",
                                type="pil",
                                height=400
                            )
                        
                        with gr.Column(scale=1):
                            mol_props_display = gr.Markdown(label="Properties")
                    
                    with gr.Row():
                        with gr.Column():
                            lipinski_display = gr.Markdown(label="Lipinski Rules")
                        with gr.Column():
                            veber_display = gr.Markdown(label="Veber Rules")
                    
                    gr.Examples(
                        examples=[
                            ["CC(=O)OC1=CC=CC=C1C(=O)O"],  # Aspirin
                            ["CC(=O)Nc1ccc(O)cc1"],  # Acetaminophen
                            ["CN1C=NC2=C1C(=O)N(C(=O)N2C)C"],  # Caffeine
                            ["CC(C)Cc1ccc(cc1)C(C)C(=O)O"],  # Ibuprofen
                            ["c1ccc2c(c1)cc3ccc4cccc5ccc2c3c45"],  # Pyrene
                        ],
                        inputs=[mol_smiles_input],
                        label="Example Molecules"
                    )
                
                # ============ TAB 3: REACTION VISUALIZER ============
                with gr.Tab("⚗️ Reaction Visualizer", id="reaction_viz"):
                    gr.Markdown("""
                    ### Reaction System Visualizer
                    Visualize a complete reaction string and analyze the transformation.
                    
                    **Format:** `Reactants >> Products` (e.g., `CC(=O)O.OCC>>CC(=O)OCC.O`)
                    """)
                    
                    with gr.Row():
                        rxn_smiles_input = gr.Textbox(
                            label="Reaction SMILES",
                            placeholder="e.g., CC(=O)O.OCC>>CC(=O)OCC.O",
                            lines=1,
                            scale=4
                        )
                        viz_rxn_btn = gr.Button("Visualize Reaction", variant="primary", scale=1)
                    
                    rxn_image = gr.Image(
                        label="Reaction Scheme",
                        type="pil",
                        height=300
                    )
                    
                    rxn_analysis = gr.Markdown(label="Reaction Analysis")
                    
                    gr.Examples(
                        examples=[
                            ["CC(=O)O.OCC>>CC(=O)OCC.O"],  # Esterification
                            ["CC(=O)O.CN>>CC(=O)NC.O"],  # Amide formation
                            ["c1ccccc1Br.c1ccc(cc1)B(O)O>>c1ccc(-c2ccccc2)cc1"],  # Suzuki
                            ["CC=O.N>>CCN"],  # Reductive amination
                        ],
                        inputs=[rxn_smiles_input],
                        label="Example Reactions"
                    )
                
                # ============ TAB 4: REACTION PREDICTOR ============
                with gr.Tab("🔮 Reaction Predictor", id="reaction_pred"):
                    gr.Markdown("""
                    ### Predict Reaction Products
                    Provide reactants and select a reaction type (or enter custom SMIRKS) to predict products.
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            pred_reactants_input = gr.Textbox(
                                label="Reactants (dot-separated SMILES)",
                                placeholder="e.g., CC(=O)O.CN",
                                lines=1
                            )
                            
                            with gr.Row():
                                reaction_type_dropdown = gr.Dropdown(
                                    label="Reaction Type",
                                    choices=reaction_choices,
                                    value="amide_coupling",
                                    scale=2
                                )
                                suggest_btn = gr.Button("🔍 Suggest Reactions", scale=1)
                            
                            custom_smirks_input = gr.Textbox(
                                label="Custom SMIRKS (if 'custom' selected)",
                                placeholder="e.g., [C:1](=[O:2])[OH1:3].[N!H0:4]>>[C:1](=[O:2])[N:4].[OH2:3]",
                                lines=1,
                                visible=True
                            )
                            
                            run_rxn_btn = gr.Button("Run Reaction", variant="primary")
                        
                        with gr.Column(scale=1):
                            suggestions_display = gr.Markdown(
                                label="Reaction Suggestions",
                                value="Enter reactants and click 'Suggest Reactions'"
                            )
                    
                    with gr.Row():
                        with gr.Column():
                            pred_product_image = gr.Image(
                                label="Product Structure",
                                type="pil",
                                height=250
                            )
                        with gr.Column():
                            pred_products_display = gr.Markdown(label="Products")
                    
                    pred_details_display = gr.Markdown(label="Product Properties")
                    
                    # Reaction template reference
                    with gr.Accordion("Available Reaction Templates", open=False):
                        template_text = "| Key | Name | Description |\n|-----|------|-------------|\n"
                        for key, info in COMMON_REACTIONS.items():
                            template_text += f"| {key} | {info['name']} | {info['description']} |\n"
                        gr.Markdown(template_text)
                    
                    gr.Examples(
                        examples=[
                            ["CC(=O)O.CN", "amide_coupling", ""],
                            ["CC(=O)O.CCO", "esterification", ""],
                            ["c1ccccc1Br.CCN", "buchwald_hartwig", ""],
                            ["CC=CC", "hydrogenation", ""],
                        ],
                        inputs=[pred_reactants_input, reaction_type_dropdown, custom_smirks_input],
                        label="Example Reactions"
                    )
            
            # ============ EVENT HANDLERS ============
            
            # Chat tab handlers
            def handle_submit(question, smiles, history, num_res, temp, mode, candidates, cite):
                new_history, mol_info_text, sources_text, attn_plot = self.process_query(
                    question, smiles, history, num_res, temp, mode, candidates, cite
                )
                # Show molecule visualization for provided SMILES or auto-extracted SMILES
                mol_img = None
                if smiles:
                    mol_img = self.visualize_molecule(smiles)
                elif not smiles:
                    # Try to auto-extract molecules for visualization
                    extracted = extract_chemicals_from_text(question, use_pubchem=False)
                    if extracted:
                        mol_img = self.visualize_molecule(extracted[0]['smiles'])
                return new_history, mol_info_text, sources_text, attn_plot, mol_img, ""
            
            submit_btn.click(
                fn=handle_submit,
                inputs=[question_input, chat_smiles_input, chatbot, num_results, temperature, reasoning_mode, num_candidates, add_citations],
                outputs=[chatbot, mol_info, sources_display, attention_plot, mol_structure, question_input]
            )
            
            question_input.submit(
                fn=handle_submit,
                inputs=[question_input, chat_smiles_input, chatbot, num_results, temperature, reasoning_mode, num_candidates, add_citations],
                outputs=[chatbot, mol_info, sources_display, attention_plot, mol_structure, question_input]
            )
            
            clear_btn.click(
                fn=lambda: ([], "", "", None, None),
                outputs=[chatbot, mol_info, sources_display, attention_plot, mol_structure]
            )
            
            # Molecule analyzer handlers
            analyze_mol_btn.click(
                fn=self.analyze_molecule_full,
                inputs=[mol_smiles_input],
                outputs=[mol_image, mol_props_display, lipinski_display, veber_display]
            )
            
            # Reaction visualizer handlers
            viz_rxn_btn.click(
                fn=self.visualize_reaction,
                inputs=[rxn_smiles_input],
                outputs=[rxn_image, rxn_analysis]
            )
            
            # Reaction predictor handlers
            run_rxn_btn.click(
                fn=self.run_reaction_prediction,
                inputs=[pred_reactants_input, reaction_type_dropdown, custom_smirks_input],
                outputs=[pred_product_image, pred_products_display, pred_details_display]
            )
            
            suggest_btn.click(
                fn=self.suggest_reactions,
                inputs=[pred_reactants_input],
                outputs=[suggestions_display]
            )
            
            gr.Markdown("""
            ---
            **Note**: This is a research prototype. Always verify critical information with domain experts.
            
            **Tech Stack**: Graphormer (GNN) • RAG (ChEMBL) • Cross-Attention Fusion • TinyLlama (1.1B) • RDKit Chemistry Engine
            """)
        
        return interface


def launch_chatbot(
    rag_dir: str = "data/rag",
    llm_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    graphormer_checkpoint: Optional[str] = None,
    device: str = "cpu",
    share: bool = False,
    server_port: int = 7860
):
    """
    Launch Helios chatbot interface
    
    Args:
        rag_dir: RAG data directory
        llm_model: LLM model name
        graphormer_checkpoint: Path to Graphormer checkpoint
        device: Device for models
        share: Create public link
        server_port: Port number
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s'
    )
    
    # Create chatbot
    chatbot = HeliosChatInterface(
        rag_dir=rag_dir,
        llm_model=llm_model,
        graphormer_checkpoint=graphormer_checkpoint,
        device=device
    )
    
    # Create and launch interface
    interface = chatbot.create_interface()
    
    interface.launch(
        share=share,
        server_port=server_port,
        server_name="0.0.0.0"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch Helios Chatbot")
    parser.add_argument('--rag-dir', type=str, default='data/rag')
    parser.add_argument('--llm', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    parser.add_argument('--graphormer', type=str, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--share', action='store_true', help='Create public link')
    parser.add_argument('--port', type=int, default=7860)
    
    args = parser.parse_args()
    
    launch_chatbot(
        rag_dir=args.rag_dir,
        llm_model=args.llm,
        graphormer_checkpoint=args.graphormer,
        device=args.device,
        share=args.share,
        server_port=args.port
    )
