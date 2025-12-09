#!/bin/bash
# Quick test script for Hebrew training
# This will run just 10 steps to verify everything works

cd /home/ubuntu/workspace/smart-turn
source venv/bin/activate

# Check if WANDB_API_KEY is set
if [ -z "$WANDB_API_KEY" ]; then
    echo "⚠️  WANDB_API_KEY not set. Please set it first:"
    echo "   export WANDB_API_KEY='your-key-here'"
    echo "   or create a .env file"
    exit 1
fi

echo "🚀 Starting Hebrew training smoke test..."
echo "📊 This will train for just a few steps to verify the setup"
echo ""

# Run training with Hebrew data only, for just a few steps
python train_local.py \
    --run-name "test-hebrew-smoke" \
    --batch-size 4 \
    --eval-batch-size 4 \
    --epochs 1 \
    --add-dataset "./datasets/output/smart-turn-hebrew-train" \
    --wandb-project "smart-turn-ft"

echo ""
echo "✅ Smoke test complete!"

