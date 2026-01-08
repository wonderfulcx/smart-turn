#!/usr/bin/env python3
"""
Threshold calibration script for Smart Turn Hebrew EOT detection.

Uses ROC curve analysis to find the optimal probability threshold for Hebrew data.

NOTE: The ONNX model outputs PROBABILITIES (sigmoid is applied in the model's forward method),
NOT raw logits. The training code thresholds at prob > 0.5.
"""

import argparse
import logging
import numpy as np
import onnxruntime as ort
from datasets import load_from_disk
from transformers import WhisperFeatureExtractor
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, 
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
from tqdm import tqdm

from audio_utils import truncate_audio_to_last_n_seconds

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Constants
SAMPLING_RATE = 16000
MAX_AUDIO_LENGTH = 8 * SAMPLING_RATE  # 8 seconds


def load_hebrew_dataset(dataset_path):
    """Load Hebrew test dataset."""
    log.info(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    # Handle different dataset structures
    if "train" in dataset:
        dataset = dataset["train"]
    
    log.info(f"Loaded {len(dataset)} samples")
    
    # Log class distribution
    labels = dataset["endpoint_bool"]
    n_complete = sum(labels)
    n_incomplete = len(labels) - n_complete
    log.info(f"Class distribution: {n_complete} complete ({100*n_complete/len(labels):.1f}%), "
             f"{n_incomplete} incomplete ({100*n_incomplete/len(labels):.1f}%)")
    
    return dataset


def run_inference(onnx_path, dataset, feature_extractor):
    """Run inference on all samples and collect probabilities and labels.
    
    NOTE: The ONNX model outputs PROBABILITIES (sigmoid already applied in model),
    not raw logits!
    """
    log.info(f"Loading ONNX model: {onnx_path}")
    
    # Try CUDA first, fallback to CPU
    try:
        session = ort.InferenceSession(
            onnx_path, 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        log.info(f"Using provider: {session.get_providers()[0]}")
    except Exception as e:
        log.warning(f"CUDA failed: {e}, falling back to CPU")
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    all_probs = []
    all_labels = []
    
    log.info("Running inference...")
    for sample in tqdm(dataset, desc="Inference"):
        # Get audio and label
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        label = sample["endpoint_bool"]
        
        # Truncate to last 8 seconds (or pad if shorter)
        audio = truncate_audio_to_last_n_seconds(audio)
        
        # Extract features (must match training code exactly!)
        features = feature_extractor(
            audio,
            sampling_rate=SAMPLING_RATE,
            return_tensors="np",
            max_length=MAX_AUDIO_LENGTH,
            truncation=True,
            padding="max_length",
            do_normalize=True,  # Critical: must match training!
        )
        
        input_features = features.input_features.astype(np.float32)
        
        # Run inference - model outputs PROBABILITIES (sigmoid already applied)
        outputs = session.run([output_name], {input_name: input_features})
        prob = outputs[0][0, 0]
        
        all_probs.append(prob)
        all_labels.append(label)
    
    return np.array(all_probs), np.array(all_labels)


def find_optimal_threshold(probs, labels):
    """Find optimal PROBABILITY threshold using various methods.
    
    Note: Model outputs are already probabilities (sigmoid applied in model).
    Training code uses threshold of 0.5 on these probabilities.
    """
    
    # ROC Curve Analysis
    fpr, tpr, roc_thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    
    # Youden's J statistic (maximizes TPR - FPR)
    j_scores = tpr - fpr
    best_j_idx = np.argmax(j_scores)
    best_threshold_youden = roc_thresholds[best_j_idx]
    
    # Probability threshold range (0 to 1)
    threshold_range = np.arange(0.1, 0.95, 0.01)
    
    # F1 score at each probability threshold
    f1_scores = []
    for thresh in threshold_range:
        preds = (probs > thresh).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        f1_scores.append((thresh, f1))
    
    f1_scores = np.array(f1_scores)
    best_f1_idx = np.argmax(f1_scores[:, 1])
    best_threshold_f1 = f1_scores[best_f1_idx, 0]
    
    # Macro F1 at each probability threshold (better for imbalanced data)
    macro_f1_scores = []
    for thresh in threshold_range:
        preds = (probs > thresh).astype(int)
        macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
        macro_f1_scores.append((thresh, macro_f1))
    
    macro_f1_scores = np.array(macro_f1_scores)
    best_macro_f1_idx = np.argmax(macro_f1_scores[:, 1])
    best_threshold_macro_f1 = macro_f1_scores[best_macro_f1_idx, 0]
    
    return {
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'roc_thresholds': roc_thresholds,
        'best_threshold_youden': best_threshold_youden,
        'best_j_score': j_scores[best_j_idx],
        'best_threshold_f1': best_threshold_f1,
        'best_f1': f1_scores[best_f1_idx, 1],
        'best_threshold_macro_f1': best_threshold_macro_f1,
        'best_macro_f1': macro_f1_scores[best_macro_f1_idx, 1],
        'f1_scores': f1_scores,
        'macro_f1_scores': macro_f1_scores,
        'threshold_range': threshold_range,
    }


def evaluate_at_threshold(probs, labels, threshold):
    """Evaluate metrics at a specific probability threshold."""
    preds = (probs > threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    
    return {
        'threshold': threshold,
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
        'macro_f1': f1_score(labels, preds, average='macro', zero_division=0),
        'class0_precision': precision_score(labels, preds, pos_label=0, zero_division=0),
        'class0_recall': recall_score(labels, preds, pos_label=0, zero_division=0),
        'class0_f1': f1_score(labels, preds, pos_label=0, zero_division=0),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
    }


def plot_analysis(results, probs, labels, model_name, output_path):
    """Generate analysis plots using probability thresholds."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Training default threshold is 0.5 probability
    prob_default = 0.5
    prob_youden = results['best_threshold_youden']
    prob_best_macro = results['best_threshold_macro_f1']
    
    # 1. ROC Curve
    ax1 = axes[0, 0]
    ax1.plot(results['fpr'], results['tpr'], 'b-', linewidth=2, 
             label=f"ROC (AUC = {results['roc_auc']:.3f})")
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax1.scatter([results['fpr'][np.argmax(results['tpr'] - results['fpr'])]], 
                [results['tpr'][np.argmax(results['tpr'] - results['fpr'])]], 
                c='red', s=100, zorder=5, 
                label=f"Youden's J (prob={prob_youden:.2f})")
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'ROC Curve - {model_name}')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # 2. F1 Score vs Probability Threshold
    ax2 = axes[0, 1]
    ax2.plot(results['f1_scores'][:, 0], results['f1_scores'][:, 1], 'b-', 
             linewidth=2, label='Binary F1')
    ax2.plot(results['macro_f1_scores'][:, 0], results['macro_f1_scores'][:, 1], 'g-', 
             linewidth=2, label='Macro F1')
    ax2.axvline(x=prob_default, color='gray', linestyle='--', label=f'Training default ({prob_default:.2f})')
    ax2.axvline(x=prob_best_macro, color='red', linestyle='-', 
                label=f"Best Macro F1 ({prob_best_macro:.2f})")
    ax2.set_xlabel('Probability Threshold')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score vs Probability Threshold')
    ax2.set_xlim(0.2, 0.85)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Probability Distribution
    ax3 = axes[1, 0]
    ax3.hist(probs[labels == 0], bins=50, alpha=0.6, label='Incomplete (0)', color='red')
    ax3.hist(probs[labels == 1], bins=50, alpha=0.6, label='Complete (1)', color='green')
    ax3.axvline(x=prob_default, color='gray', linestyle='--', linewidth=2, label=f'Training default ({prob_default:.2f})')
    ax3.axvline(x=prob_best_macro, color='blue', linestyle='-', 
                linewidth=2, label=f"Optimal ({prob_best_macro:.2f})")
    ax3.set_xlabel('Model Output (Probability)')
    ax3.set_ylabel('Count')
    ax3.set_title('Probability Distribution by Class')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Metrics comparison table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Compare metrics at different probability thresholds
    thresholds_to_compare = [prob_default, prob_youden, prob_best_macro]
    
    table_data = []
    headers = ['Metric', 
               f"Training Default\nprob > {thresholds_to_compare[0]:.2f}", 
               f"Youden's J\nprob > {thresholds_to_compare[1]:.2f}", 
               f"Best Macro F1\nprob > {thresholds_to_compare[2]:.2f}"]
    
    metrics_at_thresholds = [evaluate_at_threshold(probs, labels, t) for t in thresholds_to_compare]
    
    metric_names = ['Accuracy', 'Macro F1', 'Binary F1', 'Binary Recall', 
                    'Incomplete F1', 'Incomplete Recall']
    metric_keys = ['accuracy', 'macro_f1', 'f1', 'recall', 'class0_f1', 'class0_recall']
    
    for name, key in zip(metric_names, metric_keys):
        row = [name]
        for m in metrics_at_thresholds:
            row.append(f"{m[key]*100:.1f}%")
        table_data.append(row)
    
    table = ax4.table(cellText=table_data, colLabels=headers, loc='center',
                      cellLoc='center', colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Highlight best values
    for i in range(len(table_data)):
        values = [float(table_data[i][j].replace('%', '')) for j in range(1, 4)]
        best_col = np.argmax(values) + 1
        table[(i+1, best_col)].set_facecolor('#90EE90')
    
    ax4.set_title('Metrics Comparison at Different Probability Thresholds', pad=20, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    log.info(f"Saved analysis plot to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Calibrate EOT detection threshold for Hebrew (using probabilities)")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to ONNX model")
    parser.add_argument("--dataset", type=str, 
                        default="./datasets/output/smart-turn-hebrew-test",
                        help="Path to Hebrew test dataset")
    parser.add_argument("--output", type=str, default="threshold_calibration.png",
                        help="Output path for analysis plot")
    args = parser.parse_args()
    
    # Load feature extractor (must match training code!)
    log.info("Loading Whisper feature extractor...")
    feature_extractor = WhisperFeatureExtractor(chunk_length=8)  # 8 seconds, same as training
    
    # Load dataset
    dataset = load_hebrew_dataset(args.dataset)
    
    # Run inference - get PROBABILITIES (model applies sigmoid internally)
    probs, labels = run_inference(args.model, dataset, feature_extractor)
    
    # Find optimal thresholds
    log.info("\n" + "="*60)
    log.info("THRESHOLD CALIBRATION RESULTS (using PROBABILITY thresholds)")
    log.info("="*60)
    log.info("Note: Model outputs PROBABILITIES (sigmoid applied in model).")
    log.info("      Training code thresholds at prob > 0.5")
    
    results = find_optimal_threshold(probs, labels)
    
    log.info(f"\n📊 ROC AUC: {results['roc_auc']:.4f}")
    log.info(f"\n📊 Probability Distribution:")
    log.info(f"   Min: {probs.min():.4f}, Max: {probs.max():.4f}, Mean: {probs.mean():.4f}")
    
    log.info(f"\n🎯 Optimal PROBABILITY Thresholds:")
    log.info(f"   Youden's J:     prob > {results['best_threshold_youden']:.4f} (J={results['best_j_score']:.4f})")
    log.info(f"   Best Binary F1: prob > {results['best_threshold_f1']:.4f} (F1={results['best_f1']:.4f})")
    log.info(f"   Best Macro F1:  prob > {results['best_threshold_macro_f1']:.4f} (Macro F1={results['best_macro_f1']:.4f})")
    
    # Compare default vs optimal
    log.info(f"\n📈 Metrics Comparison:")
    for thresh_name, thresh_val in [("Training Default (prob=0.5)", 0.5), 
                                     ("Youden's J", results['best_threshold_youden']),
                                     ("Best Macro F1", results['best_threshold_macro_f1'])]:
        metrics = evaluate_at_threshold(probs, labels, thresh_val)
        log.info(f"\n   {thresh_name} (prob > {thresh_val:.3f}):")
        log.info(f"      Accuracy:         {metrics['accuracy']*100:.2f}%")
        log.info(f"      Macro F1:         {metrics['macro_f1']*100:.2f}%")
        log.info(f"      Complete F1:      {metrics['f1']*100:.2f}%")
        log.info(f"      Complete Recall:  {metrics['recall']*100:.2f}%")
        log.info(f"      Incomplete F1:    {metrics['class0_f1']*100:.2f}%")
        log.info(f"      Incomplete Recall:{metrics['class0_recall']*100:.2f}%")
        log.info(f"      Confusion: TP={metrics['tp']}, FP={metrics['fp']}, TN={metrics['tn']}, FN={metrics['fn']}")
    
    # Generate plots
    model_name = args.model.split('/')[-1].replace('.onnx', '')
    plot_analysis(results, probs, labels, model_name, args.output)
    
    log.info(f"\n✅ Calibration complete! See {args.output} for visualization.")
    
    # Summary recommendation
    log.info("\n" + "="*60)
    log.info("RECOMMENDATION")
    log.info("="*60)
    best_prob = results['best_threshold_macro_f1']
    default_metrics = evaluate_at_threshold(probs, labels, 0.5)
    best_metrics = evaluate_at_threshold(probs, labels, best_prob)
    
    improvement = best_metrics['macro_f1'] - default_metrics['macro_f1']
    
    if abs(improvement) < 0.01:
        log.info("Training default (prob > 0.5) is near-optimal for this dataset.")
        log.info(f"   Default Macro F1: {default_metrics['macro_f1']*100:.2f}%")
    else:
        log.info(f"Optimal: prob > {best_prob:.3f}")
        log.info(f"Improvement over default: {improvement*100:+.1f}% Macro F1")
        log.info(f"\nDeployment code:")
        log.info(f"  # Model outputs probabilities, just threshold directly")
        log.info(f"  is_complete = model_output > {best_prob:.3f}")


if __name__ == "__main__":
    main()
