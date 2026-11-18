#!/usr/bin/env python3
"""
Launch Helios Chatbot UI
Simple launcher with progress indicators
"""

import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Launch Helios Chatbot")
    parser.add_argument('--rag-dir', type=str, default='data/rag')
    parser.add_argument('--llm', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    parser.add_argument('--graphormer', type=str, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--share', action='store_true', help='Create public link')
    parser.add_argument('--port', type=int, default=7860)
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔬 Helios Chemistry AI - Chatbot Interface")
    print("=" * 70)
    print("\n📦 Loading dependencies...")
    
    # Import with progress
    print("  ⏳ Loading Gradio...")
    import gradio as gr
    
    print("  ⏳ Loading PyTorch...")
    import torch
    
    print("  ⏳ Loading RDKit (this may take a moment)...")
    from rdkit import Chem
    
    print("  ⏳ Loading Helios modules...")
    from helios.ui.chatbot_app import launch_chatbot
    
    print("\n✅ All dependencies loaded!")
    print("\n🚀 Starting chatbot server...")
    print(f"   Port: {args.port}")
    print(f"   Device: {args.device}")
    print(f"   LLM: {args.llm}")
    print("\n" + "=" * 70)
    
    # Launch
    launch_chatbot(
        rag_dir=args.rag_dir,
        llm_model=args.llm,
        graphormer_checkpoint=args.graphormer,
        device=args.device,
        share=args.share,
        server_port=args.port
    )

if __name__ == "__main__":
    main()
