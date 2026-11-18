"""
Gradio-based chatbot interface for Helios
"""

import gradio as gr
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class HeliosChatbot:
    """
    Helios chatbot interface
    
    TODO: Implement full UI
    - Chat interface with conversation history
    - Molecule input (SMILES or draw)
    - Property predictions display
    - Citation links
    - Attention visualization
    - 3D molecule viewer
    """
    
    def __init__(self, inference_pipeline):
        self.pipeline = inference_pipeline
        self.conversation_history = []
        
        logger.info("Initialized HeliosChatbot")
    
    def process_message(
        self,
        message: str,
        smiles: Optional[str],
        history: List[Tuple[str, str]]
    ) -> Tuple[List[Tuple[str, str]], str, dict]:
        """
        Process user message and return response
        
        Args:
            message: User message
            smiles: Optional SMILES input
            history: Conversation history
        
        Returns:
            Updated history, molecule info, and visualizations
        """
        logger.info(f"Processing message: {message[:50]}...")
        
        try:
            # Run inference
            result = self.pipeline.run(
                query=message,
                molecule_smiles=smiles
            )
            
            # Format response
            response = result['response']
            
            # Add citations
            if result.get('citations'):
                response += "\n\n**References:**\n"
                for i, citation in enumerate(result['citations'], 1):
                    response += f"[{i}] {citation}\n"
            
            # Add warnings
            if result.get('warnings'):
                response += "\n\n⚠️ **Warnings:**\n"
                for warning in result['warnings']:
                    response += f"- {warning}\n"
            
            # Update history
            history.append((message, response))
            
            # Format molecule info
            mol_info = result.get('molecule_info', {})
            
            return history, mol_info, {}
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            error_response = f"Sorry, I encountered an error: {str(e)}"
            history.append((message, error_response))
            return history, {}, {}
    
    def create_interface(self) -> gr.Blocks:
        """
        Create Gradio interface
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(title="Helios - Chemistry AI Assistant") as interface:
            gr.Markdown("""
            # 🔬 Helios - Chemistry AI Assistant
            
            Ask questions about molecules, properties, mechanisms, and more!
            Helios provides grounded responses with citations from scientific literature.
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Chat interface
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=500,
                    )
                    
                    with gr.Row():
                        message = gr.Textbox(
                            label="Your question",
                            placeholder="Ask about molecules, properties, mechanisms...",
                            lines=2,
                        )
                    
                    with gr.Row():
                        submit_btn = gr.Button("Send", variant="primary")
                        clear_btn = gr.Button("Clear")
                
                with gr.Column(scale=1):
                    # Molecule input
                    gr.Markdown("### Molecule Input (Optional)")
                    smiles_input = gr.Textbox(
                        label="SMILES",
                        placeholder="Enter SMILES string...",
                    )
                    
                    # TODO: Add molecule drawing tool
                    # molecule_drawer = gr.HTML(...)
                    
                    # Molecule visualization
                    gr.Markdown("### Molecule Structure")
                    molecule_viz = gr.HTML(label="Structure")
                    
                    # Property predictions
                    gr.Markdown("### Predicted Properties")
                    properties_output = gr.JSON(label="Properties")
            
            # Example queries
            gr.Markdown("### Example Queries")
            gr.Examples(
                examples=[
                    ["What is the mechanism of action of aspirin?", ""],
                    ["Predict the IC50 of this compound", "CC(=O)Oc1ccccc1C(=O)O"],
                    ["What are the key drug discovery targets in cancer?", ""],
                    ["Explain the structure-activity relationship of kinase inhibitors", ""],
                ],
                inputs=[message, smiles_input],
            )
            
            # Event handlers
            submit_btn.click(
                fn=self.process_message,
                inputs=[message, smiles_input, chatbot],
                outputs=[chatbot, properties_output, molecule_viz],
            )
            
            clear_btn.click(
                fn=lambda: ([], ""),
                outputs=[chatbot, message],
            )
            
            # Auto-submit on Enter
            message.submit(
                fn=self.process_message,
                inputs=[message, smiles_input, chatbot],
                outputs=[chatbot, properties_output, molecule_viz],
            )
        
        return interface
    
    def launch(self, **kwargs):
        """Launch the Gradio interface"""
        interface = self.create_interface()
        interface.launch(**kwargs)


def main():
    """Main entry point for chatbot"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Helios Chatbot')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--share', action='store_true', help='Create public link')
    
    args = parser.parse_args()
    
    # TODO: Load models
    # graphormer = load_graphormer(...)
    # retriever = RAGRetriever(...)
    # llm = HeliosLLM(...)
    # fusion = HeliosFusionModel(...)
    # pipeline = HeliosInferencePipeline(...)
    
    # For now, use placeholder
    pipeline = None
    
    chatbot = HeliosChatbot(pipeline)
    chatbot.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
