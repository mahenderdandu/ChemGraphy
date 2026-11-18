"""
HELIOS PROJECT IMPLEMENTATION ROADMAP
=====================================

This document outlines the complete implementation plan for the Helios Chemistry AI system.

## PROJECT STATUS

✅ Phase 0: System Architecture & Design - COMPLETE
✅ Phase 1: Data Pipeline (Partial) - IN PROGRESS
  ✅ Data downloaders (ChEMBL, PubChem)
  ✅ Molecule cleaning & standardization
  ✅ Activity data processing
  ✅ Main pipeline script
  ⏸️ RAG corpus builder (TODO)
  
□ Phase 2: Graphormer Pipeline - PENDING
□ Phase 3: RAG Pipeline - PENDING
□ Phase 4: LLM Pipeline - PENDING
□ Phase 5: Fusion Layer - PENDING
□ Phase 6: Inference Pipeline - PENDING
□ Phase 7: Safety Controls - PENDING
□ Phase 8: UI/Chatbot - PENDING

## DIRECTORY STRUCTURE (CREATED)

```
helios/
├── README.md                          ✅ Project overview
├── requirements.txt                   ✅ Dependencies
├── setup.py                           ✅ Package setup
├── .gitignore                         ✅ Git ignore rules
│
├── configs/                           ✅ Configuration files
│   ├── data_config.yaml              ✅ Data pipeline config
│   ├── model_config.yaml             ✅ Model configurations
│   └── training_config.yaml          ✅ Training parameters
│
├── src/helios/                        
│   ├── __init__.py                   ✅
│   │
│   ├── utils/                         ✅ Shared utilities
│   │   ├── __init__.py               ✅
│   │   ├── chemistry.py              ✅ RDKit molecule processing
│   │   ├── metrics.py                ✅ Evaluation metrics
│   │   └── visualization.py          ✅ Plotting & viz
│   │
│   ├── data/                          
│   │   ├── __init__.py               ✅
│   │   ├── downloaders/              ✅ Data fetchers
│   │   │   ├── __init__.py           ✅
│   │   │   ├── chembl_downloader.py  ✅
│   │   │   └── pubchem_downloader.py ✅
│   │   └── processors/               
│   │       ├── __init__.py           ✅
│   │       ├── chembl_processor.py   ✅ ChEMBL SQL extraction
│   │       └── molecule_cleaner.py   ✅ Cleaning & standardization
│   │
│   ├── models/                        □ Model implementations
│   │   ├── graphormer/               □ GNN models
│   │   │   ├── model.py              □ Graphormer architecture
│   │   │   ├── encoder.py            □ Graph encoding
│   │   │   └── predictor.py          □ Property prediction heads
│   │   │
│   │   ├── rag/                      □ Retrieval system
│   │   │   ├── retriever.py          □ Vector retriever
│   │   │   ├── indexer.py            □ Index builder
│   │   │   └── encoder.py            □ Text encoder
│   │   │
│   │   ├── llm/                      □ Language model
│   │   │   ├── model.py              □ LLM wrapper
│   │   │   └── tokenizer.py          □ Tokenization
│   │   │
│   │   └── fusion/                   □ Fusion layer
│   │       ├── fusion_layer.py       □ Cross-attention
│   │       └── model.py              □ Complete pipeline
│   │
│   ├── training/                      □ Training scripts
│   │   ├── graphormer_train.py       □ GNN training
│   │   ├── rag_train.py              □ Retriever training
│   │   ├── llm_train.py              □ LLM finetuning
│   │   └── fusion_train.py           □ End-to-end training
│   │
│   ├── inference/                     □ Runtime pipeline
│   │   ├── query_parser.py           □ Query understanding
│   │   ├── molecule_processor.py     □ Molecule handling
│   │   ├── retriever.py              □ RAG retrieval
│   │   ├── response_generator.py     □ Response generation
│   │   └── api.py                    □ API server
│   │
│   └── ui/                            □ User interface
│       ├── app.py                    □ Gradio/Streamlit app
│       └── components/               □ UI components
│
├── scripts/                           
│   └── run_data_pipeline.py          ✅ Main data script
│
└── tests/                             □ Unit tests

```

## NEXT STEPS - PRIORITY ORDER

### IMMEDIATE (Phase 1 Completion)
1. ⏸️ Create RAG corpus builder
   - Extract compound/target descriptions
   - Chunk literature abstracts
   - Format as JSONL

### HIGH PRIORITY (Phase 2 - Graphormer)
2. □ Implement Graphormer model wrapper
   - Load pretrained Microsoft Graphormer
   - Create zero-shot embedding extractor
   - Implement property prediction heads

3. □ Create Graphormer training script
   - Multi-task learning setup
   - Loss functions (MSE, BCE)
   - Evaluation metrics

4. □ Self-supervised pretraining (optional)
   - Masked node prediction
   - Contrastive learning

### HIGH PRIORITY (Phase 3 - RAG)
5. □ Implement RAG retriever
   - Load BGE/GTE embedder
   - FAISS index builder
   - Query-passage retrieval

6. □ Create RAG training script
   - Contrastive learning
   - Hard negative mining

### MEDIUM PRIORITY (Phase 4 - LLM)
7. □ Implement LLM wrapper
   - Load Mistral-7B/Llama
   - LoRA configuration
   - Generation utilities

8. □ Create instruction finetuning script
   - Format prompt templates
   - Training loop
   - Evaluation

### MEDIUM PRIORITY (Phase 5 - Fusion)
9. □ Implement fusion layer
   - Cross-attention mechanism
   - Multi-modal encoding
   - Joint training

10. □ End-to-end training script
    - Combined loss functions
    - Grounding objectives

### MEDIUM PRIORITY (Phase 6 - Inference)
11. □ Query parser implementation
    - NER for molecules/targets
    - Intent classification

12. □ Complete inference pipeline
    - Graphormer prediction
    - RAG retrieval
    - LLM generation
    - Citation formatting

### LOW PRIORITY (Phase 7 - Safety)
13. □ Hallucination controls
    - Confidence thresholding
    - Forced citation mode
    - Database cross-checking

### LOW PRIORITY (Phase 8 - UI)
14. □ Chatbot interface
    - Gradio/Streamlit UI
    - Conversation management
    - Visualization components

## CRITICAL DEPENDENCIES

### Python Packages (Already in requirements.txt)
- torch >= 2.1.0
- transformers >= 4.35.0
- rdkit >= 2023.9.1
- faiss-cpu >= 1.7.4
- sentence-transformers >= 2.2.2
- pandas, numpy, scikit-learn

### Pretrained Models (Need to Download)
- Microsoft Graphormer (PCQM4M checkpoint)
- BAAI/bge-large-en-v1.5 (RAG encoder)
- mistralai/Mistral-7B-Instruct-v0.2 (LLM)

### Data Sources
- ChEMBL 33 database (SQLite)
- PubChem BioAssay data
- (Optional) PubMed abstracts
- (Optional) ZINC unlabeled molecules

## IMPLEMENTATION ESTIMATES

- Phase 1 (Data): 80% complete, ~2-3 hours remaining
- Phase 2 (Graphormer): ~4-6 hours
- Phase 3 (RAG): ~3-4 hours
- Phase 4 (LLM): ~4-5 hours
- Phase 5 (Fusion): ~5-7 hours
- Phase 6 (Inference): ~3-4 hours
- Phase 7 (Safety): ~2-3 hours
- Phase 8 (UI): ~2-3 hours

**Total estimated time: 25-35 hours** for full implementation

## HOW TO USE THIS PROJECT

### 1. Setup Environment
```bash
cd /Volumes/One\ Touch/Helios
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Data Pipeline (Current Stage)
```bash
# Download and process data
python scripts/run_data_pipeline.py --config configs/data_config.yaml

# Or run individual steps
python scripts/run_data_pipeline.py --step download
python scripts/run_data_pipeline.py --step process
python scripts/run_data_pipeline.py --step clean
python scripts/run_data_pipeline.py --step prepare
```

### 3. Train Models (TODO - After Phases 2-5)
```bash
# Train Graphormer
python src/helios/training/graphormer_train.py

# Train RAG retriever
python src/helios/training/rag_train.py

# Finetune LLM
python src/helios/training/llm_train.py

# Train fusion layer
python src/helios/training/fusion_train.py
```

### 4. Run Inference (TODO - After Phase 6)
```bash
# Start API server
python src/helios/inference/api.py

# Or start chatbot UI
python src/helios/ui/app.py
```

## TESTING STRATEGY

1. Unit tests for each component
2. Integration tests for pipelines
3. End-to-end tests with sample queries
4. Benchmarking against ChEMBL test set

## DEPLOYMENT CONSIDERATIONS

- Model size: ~8-10 GB total (Graphormer + LLM + embeddings)
- RAM requirement: 16+ GB
- GPU recommended: 16+ GB VRAM for training, 8+ GB for inference
- Storage: 50+ GB for data and models

## NOTES

- Current focus: Complete Phase 1 (data pipeline)
- All configuration files are ready and parameterized
- Code follows modular architecture for easy testing
- Each phase can be developed and tested independently
- Comprehensive error handling and logging throughout

## CONTACT & CONTRIBUTION

For questions or contributions, please refer to the main README.md
