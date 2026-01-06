#!/bin/bash
# Full Hebrew training run optimized for T4 GPU (16GB VRAM)
# This will train for 4 epochs on Hebrew data only

cd /home/ubuntu/workspace/smart-turn
source venv/bin/activate

echo "🚀 Starting full Hebrew training (4 epochs)..."
echo "📊 Training on 2,603 Hebrew samples"
echo "⚡ Optimized for T4 GPU (batch size 16)"
echo ""

# Run training with Hebrew data only, optimized batch sizes for T4
python train.py \
    --run-name "hebrew-full-$(date +%Y%m%d-%H%M)" \
    --batch-size 16 \
    --eval-batch-size 16 \
    --epochs 4 \
    --replace-datasets \
    --add-dataset "./datasets/output/smart-turn-hebrew-train" \
    --replace-test-datasets \
    --test-dataset "./datasets/output/smart-turn-hebrew-test" \
    --wandb-project "smart-turn-ft"

echo ""
echo "✅ Training complete!"

