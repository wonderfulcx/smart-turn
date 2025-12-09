# Smart Turn Hebrew Training - Next Steps Guide

## Overview

This guide covers what to do once you have collected your Hebrew training data (1000+ samples recommended).

---

## Step 1: Collect More Hebrew Data 📊

### Target Dataset Size
- **Minimum**: 1000 samples (500 complete, 500 incomplete)
- **Recommended**: 2000+ samples for better performance
- **Current**: 285 samples (199 train, 86 test)

### Data Requirements

Follow the guidelines in `/home/ubuntu/workspace/smart-turn/docs/data_generation_contribution_guide.md`:

**Audio Format:**
- Format: WAV or FLAC (FLAC preferred - lossless compression)
- Sample rate: 16kHz (mandatory for Whisper)
- Channels: Mono
- Bit depth: 16-bit
- Max length: 16 seconds per sample
- Trailing silence: ~200ms at the end

**File Naming:**
- Use UUID format: `{uuid}.flac` or `{uuid}.wav`
- Example: `a1b2c3d4-5678-90ab-cdef-123456789abc.flac`

**Directory Structure:**
```
all-hebrew-training/
└── heb/
    ├── complete-nofiller/
    │   ├── {uuid1}.flac
    │   ├── {uuid2}.flac
    │   └── ...
    └── incomplete-nofiller/
        ├── {uuid3}.flac
        ├── {uuid4}.flac
        └── ...
```

**Class Balance:**
- Aim for 50:50 split between complete and incomplete
- Complete = finished thought, ready for response
- Incomplete = speaker will continue (fillers, connectives, prosody)

**Content Guidelines:**
- ✅ Natural conversational Hebrew
- ✅ Voice assistant / customer service style
- ✅ Variety in length (1-16 seconds)
- ✅ Different speakers (if possible)
- ❌ No real PII (names, addresses, etc.)
- ❌ No repeating the same sentences
- ❌ Minimize background noise

---

## Step 2: Prepare New Dataset 🔧

### 2.1 Convert Audio to FLAC (if needed)

If you have WAV files, convert them to FLAC:

```bash
cd /home/ubuntu/workspace/smart-turn
source venv/bin/activate

# Convert all WAV files to FLAC and remove originals
find all-hebrew-training -name "*.wav" -type f -exec sh -c \
  'ffmpeg -i "$1" -y "${1%.wav}.flac" 2>/dev/null && rm "$1"' _ {} \;

# Verify conversion
echo "Total FLAC files:"
find all-hebrew-training -name "*.flac" -type f | wc -l
```

### 2.2 Validate Dataset Structure

```bash
# Check directory structure
tree all-hebrew-training -L 2

# Should show:
# all-hebrew-training/
# └── heb/
#     ├── complete-nofiller/
#     └── incomplete-nofiller/
```

### 2.3 Create HuggingFace Dataset

```bash
cd /home/ubuntu/workspace/smart-turn
source venv/bin/activate

# Create new version of dataset
python datasets/scripts/raw_to_hf_dataset.py \
    smart-turn-hebrew-v2 \
    all-hebrew-training \
    ./datasets/output \
    ./datasets/tmp
```

### 2.4 Split into Train/Test (70/30)

```bash
python << 'EOF'
from datasets import load_from_disk, DatasetDict
import os

print("Loading dataset...")
dataset = load_from_disk("./datasets/output/smart-turn-hebrew-v2")["train"]

print(f"Total samples: {len(dataset)}")

# Check class balance
complete = sum(1 for x in dataset if x['endpoint_bool'])
incomplete = len(dataset) - complete
print(f"Complete: {complete} ({complete/len(dataset)*100:.1f}%)")
print(f"Incomplete: {incomplete} ({incomplete/len(dataset)*100:.1f}%)")

# Split 70/30
print("\nSplitting dataset...")
split = dataset.train_test_split(test_size=0.3, seed=42)

print(f"Train: {len(split['train'])} samples")
print(f"Test: {len(split['test'])} samples")

# Save splits
DatasetDict({"train": split['train']}).save_to_disk(
    "./datasets/output/smart-turn-hebrew-v2-train"
)
DatasetDict({"train": split['test']}).save_to_disk(
    "./datasets/output/smart-turn-hebrew-v2-test"
)

print("\n✅ Datasets saved:")
print("  - ./datasets/output/smart-turn-hebrew-v2-train")
print("  - ./datasets/output/smart-turn-hebrew-v2-test")
EOF
```

---

## Step 3: Choose Training Strategy 🎯

You have two options:

### Option A: Mixed Training (Recommended) ⭐
Train on original v3 dataset (50k samples, 23 languages) + your Hebrew data.

**Pros:**
- Leverages 50k+ samples from 23 languages
- Better generalization
- Hebrew becomes 24th language in the model
- Less prone to overfitting
- Proven to work well

**Cons:**
- Longer training time (~4-5 hours)
- Downloads ~3GB on first run

**When to use:**
- You have 500+ Hebrew samples
- You want robust performance
- This is your production model (recommended for most cases)

### Option B: Hebrew Only Training
Train ONLY on your Hebrew dataset.

**Pros:**
- Fast training (~10-20 minutes)
- 100% focused on Hebrew
- Good for rapid iteration during data collection

**Cons:**
- May overfit if dataset is small (<1000 samples)
- Loses knowledge from other languages
- Not recommended unless you have 2000+ samples
- **Requires editing train_local.py first** (see command below)

**When to use:**
- You have 2000+ Hebrew samples
- You only need Hebrew language support
- You want quick experiments

---

## Step 4: Run Training 🚀

### Set Up Environment

```bash
cd /home/ubuntu/workspace/smart-turn
source venv/bin/activate

# Set your W&B API key (for real-time monitoring)
export WANDB_API_KEY='your-wandb-api-key-here'

# Or create .env file:
echo "WANDB_API_KEY=your-key-here" > .env
```

### Option A: Mixed Training (Recommended) ⭐

This trains on the original dataset (50k samples, 23 languages) + your Hebrew data.

```bash
python train_local.py \
    --run-name "v3-hebrew-$(date +%Y%m%d)" \
    --batch-size 32 \
    --epochs 4 \
    --add-dataset "./datasets/output/smart-turn-hebrew-v2-train" \
    --wandb-project "smart-turn-ft"
```

**Expected time**: 4-5 hours  
**Note**: First run will download ~3GB original dataset automatically

---

### Option B: Hebrew Only Training

This trains ONLY on your Hebrew dataset. **Requires editing the script first**.

**Step 1: Edit train_local.py**
```bash
# Open the file
nano train_local.py

# Find line 56-58 and comment out the default dataset:
"datasets_training": [
    # "pipecat-ai/smart-turn-data-v3-train",  # ← Add # to comment out
],

# Save and exit (Ctrl+X, Y, Enter)
```

**Step 2: Run training**
```bash
python train_local.py \
    --run-name "hebrew-only-$(date +%Y%m%d)" \
    --batch-size 32 \
    --epochs 4 \
    --add-dataset "./datasets/output/smart-turn-hebrew-v2-train" \
    --wandb-project "smart-turn-ft"
```

**Expected time**: 10-30 minutes (depends on dataset size)  
**Warning**: May overfit with small datasets (<1000 samples)

### Custom Training Options

```bash
# Different learning rate
python train_local.py \
    --run-name "hebrew-custom" \
    --learning-rate 3e-5 \
    --add-dataset "./datasets/output/smart-turn-hebrew-v2-train"

# More epochs (if not overfitting)
python train_local.py \
    --run-name "hebrew-longer" \
    --epochs 6 \
    --add-dataset "./datasets/output/smart-turn-hebrew-v2-train"

# Smaller batch size (if OOM errors)
python train_local.py \
    --run-name "hebrew-small-batch" \
    --batch-size 16 \
    --eval-batch-size 8 \
    --add-dataset "./datasets/output/smart-turn-hebrew-v2-train"

# See all options
python train_local.py --help
```

---

## Step 5: Monitor Training 📈

### Option 1: Real-time with W&B

If you set `WANDB_API_KEY`, watch training live:

1. Go to: https://wandb.ai/your-username/smart-turn-ft
2. Select your run (e.g., `v3-hebrew-20251209`)
3. Watch metrics update in real-time:
   - Loss curves
   - Accuracy, F1, Precision, Recall
   - Per-language breakdown (if mixed training)
   - GPU utilization
   - Training speed

### Option 2: Console Logs

Training will print:
```
2025-12-09 12:34:56 - endpointing_training | Starting training...
{'loss': 0.4182, 'learning_rate': 5e-05, 'epoch': 0.11}
{'eval_loss': 0.1790, 'eval_accuracy': 0.85, 'eval_f1': 0.88}
...
```

### Option 3: TensorBoard (if needed)

```bash
# In another terminal
cd /home/ubuntu/workspace/smart-turn
tensorboard --logdir ./output/
# Then open http://localhost:6006
```

---

## Step 6: Evaluate Results ✅

After training completes, you'll find:

```
./output/{run_name}/
├── final_model/
│   ├── config.json                 # Model configuration
│   ├── model.safetensors          # PyTorch model weights
│   ├── quantized_model.pth        # INT8 quantized PyTorch
│   ├── preprocessor_config.json   # Feature extractor config
│   └── exports/
│       ├── model_fp32.onnx        # FP32 ONNX (for deployment)
│       └── model_int8.onnx        # INT8 ONNX (smaller, faster)
├── checkpoint-{step}/              # Intermediate checkpoints
│   └── ...
├── dataset_ids.json                # Sample IDs used
└── evaluation_plots/
    ├── confusion_matrix_eval.png
    └── probability_distribution_eval.png
```

### Check Final Metrics

Look for these in the final logs or W&B:

**Target metrics** (on test set):
- **Accuracy**: 85%+ (good), 90%+ (excellent)
- **F1 Score**: 85%+ (good), 90%+ (excellent)
- **Precision**: 85%+ (avoid false positives)
- **Recall**: 85%+ (avoid missing completions)

If metrics are low:
- Collect more data
- Check data quality (labels correct?)
- Try training longer (more epochs)
- Adjust learning rate

---

## Step 7: Test the Model 🧪

### Test on Hebrew Audio Files

```bash
# Option 1: Test with ONNX model (faster)
cd /home/ubuntu/workspace/smart-turn
source venv/bin/activate

# First, update inference.py to use your new model:
# Edit: ONNX_MODEL_PATH = "./output/{run_name}/final_model/exports/model_int8.onnx"

python predict.py path/to/hebrew/test.wav
```

### Benchmark on Full Test Set

```bash
python benchmark_hebrew.py \
    --directory ./datasets/output/smart-turn-hebrew-v2-test \
    --wandb-project "smart-turn-ft" \
    --wandb-run-name "hebrew-benchmark-$(date +%Y%m%d)"
```

This will generate:
- Detailed metrics (accuracy, F1, precision, recall)
- Confusion matrix
- ROC curve
- Per-class analysis
- Processing time statistics

---

## Step 8: Deploy Model 🌐

### For Production Use

1. **Copy ONNX model to deployment location:**
```bash
cp ./output/{run_name}/final_model/exports/model_int8.onnx /path/to/production/
```

2. **Update your inference code** to use the new model path

3. **Update `inference.py`:**
```python
# Change this line:
ONNX_MODEL_PATH = "./output/{run_name}/final_model/exports/model_int8.onnx"
```

### For Pipecat Integration

See Pipecat documentation:
- https://docs.pipecat.ai/server/utilities/smart-turn/smart-turn-overview

You can use `LocalSmartTurnAnalyzerV3` with your custom ONNX model.

---

## Troubleshooting 🔧

### Out of Memory (OOM) Errors

```bash
# Reduce batch size
python train_local.py --batch-size 16 --eval-batch-size 8 ...

# Or even smaller
python train_local.py --batch-size 8 --eval-batch-size 4 ...
```

### Training Too Slow

```bash
# Check GPU is being used
nvidia-smi

# Should show python process using GPU with ~80-90% utilization
```

### Poor Performance After Training

**If accuracy < 70%:**
1. Check label quality - are samples labeled correctly?
2. Check class balance - should be ~50/50 complete/incomplete
3. Collect more data - may need 2000+ samples
4. Try longer training - increase epochs to 6 or 8

**If overfitting (train accuracy >> test accuracy):**
1. Use mixed training instead of Hebrew-only
2. Collect more diverse data
3. Reduce epochs to 2-3
4. Check for data leakage (duplicates in train/test)

### W&B Not Logging

```bash
# Check API key is set
echo $WANDB_API_KEY

# Set it if not
export WANDB_API_KEY='your-key-here'

# Or run in offline mode (logs saved locally)
export WANDB_MODE=offline
python train_local.py ...

# Sync later
wandb sync ./wandb/offline-run-*
```

### Model Files Not Found

```bash
# Check output directory
ls -la ./output/{run_name}/final_model/

# Should see:
# - config.json
# - model.safetensors
# - quantized_model.pth
# - exports/model_fp32.onnx
# - exports/model_int8.onnx
```

---

## Iterative Improvement Loop 🔄

1. **Collect initial data** (500-1000 samples)
2. **Train model** (mixed training recommended)
3. **Evaluate on test set**
4. **Identify failure cases** (check confusion matrix, listen to misclassified samples)
5. **Collect more data** targeting failure modes
6. **Retrain with augmented dataset**
7. **Repeat** until performance satisfactory

---

## Advanced: Multiple Training Runs

### Compare Different Configurations

```bash
# Run 1: Baseline
python train_local.py \
    --run-name "hebrew-baseline" \
    --learning-rate 5e-5 \
    --epochs 4 \
    --add-dataset "./datasets/output/smart-turn-hebrew-v2-train"

# Run 2: Lower learning rate
python train_local.py \
    --run-name "hebrew-lr3e5" \
    --learning-rate 3e-5 \
    --epochs 4 \
    --add-dataset "./datasets/output/smart-turn-hebrew-v2-train"

# Run 3: More epochs
python train_local.py \
    --run-name "hebrew-ep6" \
    --learning-rate 5e-5 \
    --epochs 6 \
    --add-dataset "./datasets/output/smart-turn-hebrew-v2-train"
```

Then compare results in W&B to find the best configuration.

---

## Quick Reference Commands

```bash
# ============================================
# FULL TRAINING WORKFLOW (Mixed - Recommended)
# ============================================

cd /home/ubuntu/workspace/smart-turn
source venv/bin/activate
export WANDB_API_KEY='your-key'

# 1. Prepare data (if new files added)
find all-hebrew-training -name "*.wav" -exec sh -c \
  'ffmpeg -i "$1" -y "${1%.wav}.flac" 2>/dev/null && rm "$1"' _ {} \;

python datasets/scripts/raw_to_hf_dataset.py \
    smart-turn-hebrew-v2 all-hebrew-training ./datasets/output ./datasets/tmp

# 2. Split dataset
python -c "
from datasets import load_from_disk, DatasetDict
ds = load_from_disk('./datasets/output/smart-turn-hebrew-v2')['train']
split = ds.train_test_split(test_size=0.3, seed=42)
DatasetDict({'train': split['train']}).save_to_disk('./datasets/output/smart-turn-hebrew-v2-train')
DatasetDict({'train': split['test']}).save_to_disk('./datasets/output/smart-turn-hebrew-v2-test')
"

# 3. Train (Mixed: Original + Hebrew - Recommended)
python train_local.py \
    --run-name "v3-hebrew-$(date +%Y%m%d)" \
    --batch-size 32 \
    --epochs 4 \
    --add-dataset "./datasets/output/smart-turn-hebrew-v2-train"

# 4. Benchmark
python benchmark_hebrew.py \
    --directory ./datasets/output/smart-turn-hebrew-v2-test
```

```bash
# ============================================
# HEBREW ONLY TRAINING (If you have 2000+ samples)
# ============================================

# 1. First edit train_local.py line 57:
#    Comment out: # "pipecat-ai/smart-turn-data-v3-train",

# 2. Then run training:
python train_local.py \
    --run-name "hebrew-only-$(date +%Y%m%d)" \
    --batch-size 32 \
    --epochs 4 \
    --add-dataset "./datasets/output/smart-turn-hebrew-v2-train"
```

---

## Files Reference

| File/Directory | Description |
|----------------|-------------|
| `train_local.py` | Main training script |
| `NEXT_STEPS.md` | This file |
| `HEBREW_TRAINING_SUMMARY.md` | Initial setup summary |
| `all-hebrew-training/` | Your raw audio files |
| `datasets/output/smart-turn-hebrew-v2-train/` | Processed training dataset |
| `datasets/output/smart-turn-hebrew-v2-test/` | Processed test dataset |
| `./output/{run_name}/` | Training outputs and model files |
| `./wandb/` | W&B logs (if offline mode) |

---

## Need Help?

- **Smart Turn GitHub**: https://github.com/pipecat-ai/smart-turn
- **Data Guidelines**: `/home/ubuntu/workspace/smart-turn/docs/data_generation_contribution_guide.md`
- **Pipecat Docs**: https://docs.pipecat.ai/

---

**Last Updated**: 2025-12-09
**Ready for**: Production training with 1000+ Hebrew samples

