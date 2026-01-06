#!/bin/bash
# Extended Hebrew training with more epochs and frequent evaluation
# This will help us see if the model is underfitting

cd /home/ubuntu/workspace/smart-turn
source venv/bin/activate

echo "🚀 Starting EXTENDED Hebrew training..."
echo "📊 Training: 2,603 Hebrew samples"
echo "⏱️  Epochs: 10 (vs previous 4)"
echo "📈 Eval every 50 steps (vs previous 500)"
echo "⚡ Optimized for T4 GPU (batch size 16)"
echo ""

# Run training with MORE epochs and FREQUENT evaluation
python train.py \
    --run-name "hebrew-extended-$(date +%Y%m%d-%H%M)" \
    --batch-size 16 \
    --eval-batch-size 16 \
    --epochs 10 \
    --eval-steps 50 \
    --save-steps 150 \
    --logging-steps 25 \
    --replace-datasets \
    --add-dataset "./datasets/output/smart-turn-hebrew-train" \
    --replace-test-datasets \
    --test-dataset "./datasets/output/smart-turn-hebrew-test" \
    --wandb-project "smart-turn-ft"

echo ""
echo "✅ Extended training complete!"
echo "Check W&B dashboard for learning curves:"
echo "https://wandb.ai/wonderful-ai/smart-turn-ft"

