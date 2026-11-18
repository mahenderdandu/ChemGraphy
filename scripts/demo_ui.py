"""
Quick Test UI for Helios - Minimal version for fast loading
"""

import gradio as gr
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_demo_ui():
    """Create a quick demo UI without loading the full pipeline"""
    
    def mock_response(question, smiles, history):
        """Mock response for testing UI"""
        answer = f"**Demo Mode**: Received your question about: '{question}'"
        if smiles:
            answer += f"\n\n**Molecule**: `{smiles}`"
        answer += "\n\n*Note: This is a demo. Load the full pipeline for real answers.*"
        
        history.append((question, answer))
        return history, "", ""
    
    with gr.Blocks(title="Helios Demo") as interface:
        gr.Markdown("""
        # 🔬 Helios - Chemistry AI Assistant (Demo Mode)
        
        **Quick UI Test** - Full pipeline available with: `python scripts/launch_chatbot.py`
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Chat History", height=400)
                
                with gr.Row():
                    question = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g., What is aspirin?",
                        lines=2,
                        scale=4
                    )
                    submit = gr.Button("Submit", variant="primary", scale=1)
                
                smiles = gr.Textbox(
                    label="Molecule SMILES (optional)",
                    placeholder="e.g., CC(=O)Oc1ccccc1C(=O)O"
                )
                
                clear = gr.Button("Clear")
            
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Info")
                mol_info = gr.Markdown("No analysis yet")
                sources = gr.Markdown("No sources yet")
        
        gr.Examples(
            examples=[
                ["What is aspirin?", "CC(=O)Oc1ccccc1C(=O)O"],
                ["Explain EGFR inhibitors", ""],
            ],
            inputs=[question, smiles]
        )
        
        submit.click(
            fn=mock_response,
            inputs=[question, smiles, chatbot],
            outputs=[chatbot, mol_info, sources]
        )
        
        question.submit(
            fn=mock_response,
            inputs=[question, smiles, chatbot],
            outputs=[chatbot, mol_info, sources]
        )
        
        clear.click(lambda: [], outputs=[chatbot])
        
        gr.Markdown("""
        ---
        **To load full Helios pipeline**: `python scripts/launch_chatbot.py`
        
        **Tech**: Graphormer (GNN) • RAG (ChEMBL) • Cross-Attention Fusion • TinyLlama
        """)
    
    return interface


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--share', action='store_true')
    args = parser.parse_args()
    
    print("="*70)
    print("🔬 Helios Demo UI")
    print("="*70)
    print(f"\n🚀 Launching on http://localhost:{args.port}")
    print("\n💡 This is a demo UI for testing the interface.")
    print("   For full functionality, use: python scripts/launch_chatbot.py\n")
    
    interface = create_demo_ui()
    interface.launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0"
    )
