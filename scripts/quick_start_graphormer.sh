#!/bin/bash

# Quick Start Script for Helios Phase 2 - Graphormer Training
# Uses public MoleculeNet datasets (no preprocessing needed!)

echo "============================================================"
echo "HELIOS Phase 2 - Graphormer Training (No Preprocessing!)"
echo "============================================================"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo "Step 1: Testing dataset loading..."
python -c "
from helios.data.moleculenet_loader import get_moleculenet_loaders
import logging
logging.basicConfig(level=logging.INFO)
print('Testing ESOL dataset loading...')
train_loader, val_loader, test_loader = get_moleculenet_loaders('ESOL', batch_size=8, num_workers=0)
print(f'✅ Dataset loaded successfully!')
print(f'   Train: {len(train_loader.dataset)} samples')
print(f'   Val: {len(val_loader.dataset)} samples')
print(f'   Test: {len(test_loader.dataset)} samples')
"

if [ $? -ne 0 ]; then
    echo "❌ Dataset loading failed. Check errors above."
    exit 1
fi

echo ""
echo "Step 2: Starting Graphormer training on ESOL dataset..."
echo "This will train a molecular property prediction model."
echo ""

# Train on ESOL (smallest dataset, ~1100 samples)
python scripts/train_graphormer.py \
    --dataset ESOL \
    --batch-size 16 \
    --epochs 50 \
    --lr 0.0001 \
    --checkpoint-dir models/checkpoints/graphormer_esol

echo ""
echo "============================================================"
echo "Training Complete!"
echo "============================================================"
echo ""
echo "Trained model saved to: models/checkpoints/graphormer_esol/best_model.pt"
echo ""
echo "Next steps:"
echo "  1. Check training curves: models/checkpoints/graphormer_esol/training_curves.png"
echo "  2. Try other datasets: FreeSolv, Lipophilicity, BACE"
echo "  3. Move to Phase 3: RAG implementation"
echo ""
