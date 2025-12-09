#!/usr/bin/env python3
"""
Smart Turn v3 Hebrew Benchmark Script

This script evaluates the existing Smart Turn v3 model on your Hebrew dataset.
It supports multiple input formats and provides comprehensive metrics.

Setup:
    1. Create a .env file in the project root with:
       WANDB_API_KEY=your_wandb_api_key_here
    2. Install dependencies: pip install -r requirements.txt

Usage:
    python benchmark_hebrew.py --files file1.wav:1 file2.wav:0 file3.wav:1
    python benchmark_hebrew.py --csv labels.csv
    python benchmark_hebrew.py --directory path/to/hebrew/audio/
    python benchmark_hebrew.py --directory path/to/hebrew/audio/ --threshold 0.3
    
Input formats:
    1. Command line: file1.wav:1 file2.wav:0 (filename:label pairs)
    2. CSV file: filename,label columns
    3. Directory structure: complete/ and incomplete/ subdirectories
    4. JSON file: [{"file": "path", "label": 1, "metadata": {...}}]
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(override=True)

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# Try to import inference - handle missing dependencies gracefully
try:
    from inference import predict_endpoint
    INFERENCE_AVAILABLE = True
    print("✅ Smart Turn v3 inference available")
except ImportError as e:
    INFERENCE_AVAILABLE = False
    print(f"❌ Smart Turn v3 inference not available: {e}")
    print("   Make sure you have the model files and dependencies installed")


class HebrewBenchmark:
    """Benchmark Smart Turn v3 model on Hebrew audio files."""
    
    def __init__(self, output_dir: str = "benchmark_results", use_wandb: bool = True, 
                 wandb_project: str = "eot-evaluations", wandb_run_name: Optional[str] = None,
                 wandb_tags: Optional[List[str]] = None, wandb_config: Optional[Dict] = None,
                 threshold: float = 0.5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = []
        self.predictions = []
        self.true_labels = []
        self.probabilities = []
        self.filenames = []
        self.processing_times = []
        
        # EOT detection threshold
        self.threshold = threshold
        print(f"📊 Using EOT detection threshold: {threshold}")
        
        # WandB initialization
        self.use_wandb = use_wandb
        if self.use_wandb:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            run_name = wandb_run_name or f"smart_turn_v3_eval_{timestamp}"
            tags = wandb_tags or ["smart-turn-v3", "hebrew", "audio-native-eot"]
            
            config = wandb_config or {}
            config.update({
                "model": "smart-turn-v3",
                "model_type": "audio-native",
                "language": "hebrew",
                "threshold": self.threshold,
                "output_dir": str(self.output_dir),
            })
            
            wandb.init(
                project=wandb_project,
                name=run_name,
                config=config,
                tags=tags
            )
            print(f"✅ Initialized WandB run: {wandb.run.name}")
            print(f"   Project: {wandb_project}")
            print(f"   URL: {wandb.run.url}")
        
        print(f"📊 Hebrew Smart Turn Benchmark initialized")
        print(f"   Output directory: {self.output_dir}")
    
    def load_files_from_args(self, file_args: List[str]) -> List[Dict]:
        """Load files from command line arguments (file:label format)."""
        files = []
        
        for file_arg in file_args:
            if ':' not in file_arg:
                raise ValueError(f"Invalid format: {file_arg}. Use 'filename:label' format")
            
            filepath, label_str = file_arg.rsplit(':', 1)
            try:
                label = int(label_str)
                if label not in [0, 1]:
                    raise ValueError(f"Label must be 0 or 1, got: {label}")
            except ValueError:
                raise ValueError(f"Invalid label: {label_str}. Must be 0 or 1")
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
            
            files.append({
                'file': filepath,
                'label': label,
                'source': 'command_line'
            })
        
        return files
    
    def load_files_from_csv(self, csv_path: str) -> List[Dict]:
        """Load files from CSV file with columns: filename, label."""
        files = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            # Auto-detect delimiter
            sample = f.read(1024)
            f.seek(0)
            delimiter = ',' if ',' in sample else '\t'
            
            reader = csv.DictReader(f, delimiter=delimiter)
            
            # Flexible column name detection
            fieldnames = reader.fieldnames
            file_col = None
            label_col = None
            
            for col in fieldnames:
                col_lower = col.lower()
                if col_lower in ['file', 'filename', 'path', 'audio', 'audio_file']:
                    file_col = col
                elif col_lower in ['label', 'eot', 'endpoint', 'complete', 'target']:
                    label_col = col
            
            if not file_col or not label_col:
                raise ValueError(f"CSV must have file and label columns. Found: {fieldnames}")
            
            print(f"📄 Reading CSV: file column='{file_col}', label column='{label_col}'")
            
            for i, row in enumerate(reader, 1):
                filepath = row[file_col].strip()
                label_str = row[label_col].strip()
                
                # Handle relative paths
                if not os.path.isabs(filepath):
                    csv_dir = os.path.dirname(csv_path)
                    filepath = os.path.join(csv_dir, filepath)
                
                if not os.path.exists(filepath):
                    print(f"⚠️  Warning: File not found (row {i}): {filepath}")
                    continue
                
                try:
                    # Handle various label formats
                    if label_str.lower() in ['complete', 'true', 'yes', '1']:
                        label = 1
                    elif label_str.lower() in ['incomplete', 'false', 'no', '0']:
                        label = 0
                    else:
                        label = int(float(label_str))  # Handle "1.0" format
                        
                    if label not in [0, 1]:
                        raise ValueError(f"Label must be 0 or 1, got: {label}")
                        
                except ValueError:
                    print(f"⚠️  Warning: Invalid label (row {i}): {label_str}")
                    continue
                
                files.append({
                    'file': filepath,
                    'label': label,
                    'source': 'csv',
                    'row': i,
                    **{k: v for k, v in row.items() if k not in [file_col, label_col]}
                })
        
        return files
    
    def load_files_from_directory(self, dir_path: str) -> List[Dict]:
        """Load files from directory structure (complete/ and incomplete/ subdirs)."""
        files = []
        base_path = Path(dir_path)
        
        if not base_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        # Look for complete/incomplete subdirectories
        complete_dir = base_path / "complete"
        incomplete_dir = base_path / "incomplete"
        
        # Also check for other common naming patterns
        alt_patterns = [
            ("complete-nofiller", "incomplete-nofiller"),
            ("complete-midfiller", "incomplete-midfiller"), 
            ("complete-endfiller", "incomplete-endfiller"),
            ("1", "0"),
            ("positive", "negative"),
            ("true", "false")
        ]
        
        found_pattern = None
        
        # Check standard pattern first
        if complete_dir.exists() and incomplete_dir.exists():
            found_pattern = (complete_dir, incomplete_dir)
        else:
            # Check alternative patterns
            for pos_name, neg_name in alt_patterns:
                pos_dir = base_path / pos_name
                neg_dir = base_path / neg_name
                if pos_dir.exists() and neg_dir.exists():
                    found_pattern = (pos_dir, neg_dir)
                    break
        
        if not found_pattern:
            # Look for any subdirectories and guess
            subdirs = [d for d in base_path.iterdir() if d.is_dir()]
            if len(subdirs) == 2:
                print(f"⚠️  Found 2 subdirectories, guessing labels:")
                print(f"   {subdirs[0].name} → Complete (1)")
                print(f"   {subdirs[1].name} → Incomplete (0)")
                found_pattern = (subdirs[0], subdirs[1])
            else:
                raise ValueError(f"Could not find complete/incomplete directory structure in {dir_path}")
        
        pos_dir, neg_dir = found_pattern
        print(f"📁 Loading from directories:")
        print(f"   Complete (1): {pos_dir.name}")
        print(f"   Incomplete (0): {neg_dir.name}")
        
        # Load complete files (label=1)
        for audio_file in pos_dir.glob("*"):
            if audio_file.is_file() and audio_file.suffix.lower() in ['.wav', '.flac', '.mp3', '.m4a']:
                files.append({
                    'file': str(audio_file),
                    'label': 1,
                    'source': 'directory',
                    'category': pos_dir.name
                })
        
        # Load incomplete files (label=0)  
        for audio_file in neg_dir.glob("*"):
            if audio_file.is_file() and audio_file.suffix.lower() in ['.wav', '.flac', '.mp3', '.m4a']:
                files.append({
                    'file': str(audio_file),
                    'label': 0,
                    'source': 'directory',
                    'category': neg_dir.name
                })
        
        return files
    
    def load_files_from_json(self, json_path: str) -> List[Dict]:
        """Load files from JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of objects")
        
        files = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"JSON item {i} must be an object")
            
            if 'file' not in item or 'label' not in item:
                raise ValueError(f"JSON item {i} must have 'file' and 'label' fields")
            
            filepath = item['file']
            if not os.path.isabs(filepath):
                json_dir = os.path.dirname(json_path)
                filepath = os.path.join(json_dir, filepath)
            
            if not os.path.exists(filepath):
                print(f"⚠️  Warning: File not found (item {i}): {filepath}")
                continue
            
            label = item['label']
            if label not in [0, 1]:
                raise ValueError(f"Label must be 0 or 1, got: {label}")
            
            files.append({
                'file': filepath,
                'label': label,
                'source': 'json',
                'index': i,
                **{k: v for k, v in item.items() if k not in ['file', 'label']}
            })
        
        return files
    
    def process_audio_file(self, filepath: str) -> Tuple[Optional[Dict], float]:
        """
        Process a single audio file and return prediction + processing time.
        
        Note: We only use the 'probability' from inference.py.
        The 'prediction' field is ignored since inference.py uses a hardcoded 0.5 threshold.
        We apply our custom threshold in run_benchmark().
        """
        if not INFERENCE_AVAILABLE:
            raise RuntimeError("Smart Turn inference not available")
        
        try:
            # Load audio file
            start_time = time.perf_counter()
            
            # Use librosa to load with automatic resampling
            audio, sr = librosa.load(filepath, sr=None, mono=True)
            
            # Resample to 16kHz if needed (same as predict.py)
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
            # Ensure float32 and proper range
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / np.max(np.abs(audio))
            
            # Run Smart Turn inference (we only use the 'probability', not 'prediction')
            result = predict_endpoint(audio)
            
            processing_time = (time.perf_counter() - start_time) * 1000  # ms
            
            return result, processing_time
            
        except Exception as e:
            print(f"❌ Error processing {filepath}: {e}")
            return None, 0.0
    
    def run_benchmark(self, files: List[Dict]) -> Dict:
        """Run benchmark on list of files."""
        print(f"\n🚀 Starting benchmark on {len(files)} files...")
        
        if not INFERENCE_AVAILABLE:
            raise RuntimeError("Cannot run benchmark: Smart Turn inference not available")
        
        results = []
        failed_files = []
        
        for i, file_info in enumerate(files, 1):
            filepath = file_info['file']
            true_label = file_info['label']
            
            print(f"Processing {i}/{len(files)}: {os.path.basename(filepath)}", end=" ")
            
            prediction_result, proc_time = self.process_audio_file(filepath)
            
            if prediction_result is None:
                failed_files.append(filepath)
                print("❌ FAILED")
                continue
            
            # Get probability from inference (ignore the hardcoded 0.5 prediction)
            probability = prediction_result['probability']
            
            # Apply our custom threshold to get the prediction
            predicted_label = 1 if probability >= self.threshold else 0
            
            # Store results
            result = {
                'filename': os.path.basename(filepath),
                'filepath': filepath,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'probability': probability,
                'processing_time_ms': proc_time,
                'correct': predicted_label == true_label,
                **{k: v for k, v in file_info.items() if k not in ['file', 'label']}
            }
            
            results.append(result)
            self.results.append(result)
            self.predictions.append(predicted_label)
            self.true_labels.append(true_label)
            self.probabilities.append(probability)
            self.filenames.append(os.path.basename(filepath))
            self.processing_times.append(proc_time)
            
            status = "✅" if predicted_label == true_label else "❌"
            print(f"{status} P={probability:.3f} ({proc_time:.1f}ms)")
        
        print(f"\n📊 Processed {len(results)}/{len(files)} files successfully")
        if failed_files:
            print(f"⚠️  {len(failed_files)} files failed to process")
        
        return self.calculate_metrics()
    
    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive metrics using only classification_report."""
        if not self.predictions:
            raise ValueError("No predictions available")
        
        y_true = np.array(self.true_labels)
        y_pred = np.array(self.predictions)  # Already thresholded during run_benchmark
        y_prob = np.array(self.probabilities)
        
        processing_times_s = np.array(self.processing_times) / 1000.0  # Convert ms to seconds
        
        # Convert to boolean for classification_report to match wonderful-ml format
        # This makes sklearn use "True"/"False" string keys instead of "0"/"1"
        y_true_bool = y_true.astype(bool)
        y_pred_bool = y_pred.astype(bool)
        
        # Get full classification report - preserve structure exactly as-is
        # Note: classification_report converts boolean labels to string keys "True"/"False"
        report_dict = classification_report(
            y_true_bool, 
            y_pred_bool, 
            output_dict=True, 
            zero_division=0
        )
        
        # Confusion matrix (for tp, tn, fp, fn - not calculated elsewhere)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # ROC AUC (not in classification_report)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.0  # Handle case where only one class is present
        
        # Performance stats
        avg_processing_time = np.mean(self.processing_times)
        total_audio_duration = len(self.results) * 4.0  # Assume 4s average
        
        # Build metrics dict - include classification_report as-is, plus additional metrics
        metrics = {
            # Include the entire classification_report dict as-is
            **report_dict,
            
            # EOT detection threshold used
            'threshold': float(self.threshold),
            
            # Confusion matrix components (needed for wonderful-ml compatibility)
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            
            # AUC (not in classification_report)
            'auc': float(auc),
            
            # Latency metrics
            'latency_mean': float(np.mean(processing_times_s)),
            'latency_median': float(np.median(processing_times_s)),
            'latency_p95': float(np.percentile(processing_times_s, 95)),
            'latency_p99': float(np.percentile(processing_times_s, 99)),
            
            # For backward compatibility - legacy structure
            'total_files': len(self.results),
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            },
            'performance': {
                'avg_processing_time_ms': avg_processing_time,
                'total_processing_time_s': sum(self.processing_times) / 1000,
                'estimated_audio_duration_s': total_audio_duration,
                'processing_speed_ratio': total_audio_duration / (sum(self.processing_times) / 1000)
            },
            'class_distribution': {
                'incomplete_count': int(np.sum(y_true == 0)),
                'complete_count': int(np.sum(y_true == 1)),
                'incomplete_percentage': float(np.mean(y_true == 0) * 100),
                'complete_percentage': float(np.mean(y_true == 1) * 100)
            }
        }
        
        return metrics
    
    def generate_visualizations(self, metrics: Dict):
        """Generate visualization plots."""
        print("\n📈 Generating visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Smart Turn v3 Hebrew Benchmark Results (Threshold: {metrics["threshold"]:.2f})', 
                     fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        cm = np.array([[metrics['confusion_matrix']['true_negatives'], 
                       metrics['confusion_matrix']['false_positives']],
                      [metrics['confusion_matrix']['false_negatives'], 
                       metrics['confusion_matrix']['true_positives']]])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0],
                   xticklabels=['Incomplete', 'Complete'],
                   yticklabels=['Incomplete', 'Complete'])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('True')
        
        # 2. Metrics Bar Chart
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        metric_values = [
            metrics['accuracy'], 
            metrics['weighted avg']['precision'], 
            metrics['weighted avg']['recall'], 
            metrics['weighted avg']['f1-score'], 
            metrics['auc']
        ]
        
        bars = axes[0,1].bar(metric_names, metric_values, color='skyblue', edgecolor='navy')
        axes[0,1].set_title('Performance Metrics')
        axes[0,1].set_ylim(0, 1)
        axes[0,1].set_ylabel('Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{value:.3f}', ha='center', va='bottom')
        
        plt.setp(axes[0,1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Class Distribution
        class_counts = [metrics['class_distribution']['incomplete_count'],
                       metrics['class_distribution']['complete_count']]
        class_labels = ['Incomplete (0)', 'Complete (1)']
        
        wedges, texts, autotexts = axes[0,2].pie(class_counts, labels=class_labels, autopct='%1.1f%%',
                                                startangle=90, colors=['lightcoral', 'lightgreen'])
        axes[0,2].set_title('Class Distribution')
        
        # 4. Probability Distribution
        incomplete_probs = [self.probabilities[i] for i, label in enumerate(self.true_labels) if label == 0]
        complete_probs = [self.probabilities[i] for i, label in enumerate(self.true_labels) if label == 1]
        
        axes[1,0].hist(incomplete_probs, bins=20, alpha=0.7, label='Incomplete (True)', color='lightcoral', density=True)
        axes[1,0].hist(complete_probs, bins=20, alpha=0.7, label='Complete (True)', color='lightgreen', density=True)
        axes[1,0].axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
        axes[1,0].set_title('Probability Distribution by True Class')
        axes[1,0].set_xlabel('Predicted Probability')
        axes[1,0].set_ylabel('Density')
        axes[1,0].legend()
        
        # 5. ROC Curve
        if len(set(self.true_labels)) > 1:  # Only if we have both classes
            fpr, tpr, _ = roc_curve(self.true_labels, self.probabilities)
            axes[1,1].plot(fpr, tpr, color='darkorange', lw=2, 
                          label=f'ROC Curve (AUC = {metrics["auc"]:.3f})')
            axes[1,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            axes[1,1].set_xlim([0.0, 1.0])
            axes[1,1].set_ylim([0.0, 1.05])
            axes[1,1].set_xlabel('False Positive Rate')
            axes[1,1].set_ylabel('True Positive Rate')
            axes[1,1].set_title('ROC Curve')
            axes[1,1].legend(loc="lower right")
        else:
            axes[1,1].text(0.5, 0.5, 'ROC Curve\n(Single class only)', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('ROC Curve')
        
        # 6. Processing Time Distribution
        axes[1,2].hist(self.processing_times, bins=20, color='gold', edgecolor='orange', alpha=0.7)
        axes[1,2].axvline(x=np.mean(self.processing_times), color='red', linestyle='--', 
                         label=f'Mean: {np.mean(self.processing_times):.1f}ms')
        axes[1,2].set_title('Processing Time Distribution')
        axes[1,2].set_xlabel('Processing Time (ms)')
        axes[1,2].set_ylabel('Frequency')
        axes[1,2].legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "benchmark_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved visualizations to: {plot_path}")
        
        plt.show()
        
        return plot_path
    
    def save_results(self, metrics: Dict):
        """Save detailed results to files and log to WandB."""
        print("\n💾 Saving results...")
        
        # Save metrics summary
        metrics_path = self.output_dir / "metrics_summary.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"📋 Saved metrics summary to: {metrics_path}")
        
        # Save detailed results
        results_path = self.output_dir / "detailed_results.csv"
        with open(results_path, 'w', newline='', encoding='utf-8') as f:
            if self.results:
                writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
                writer.writeheader()
                writer.writerows(self.results)
        print(f"📄 Saved detailed results to: {results_path}")
        
        # Save classification report
        report_path = None
        if self.true_labels and self.predictions:
            report = classification_report(self.true_labels, self.predictions, 
                                         target_names=['Incomplete', 'Complete'])
            report_path = self.output_dir / "classification_report.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("Smart Turn v3 Hebrew Benchmark - Classification Report\n")
                f.write("=" * 60 + "\n")
                f.write(f"Threshold: {self.threshold:.2f}\n")
                f.write("=" * 60 + "\n\n")
                f.write(report)
            print(f"📊 Saved classification report to: {report_path}")
        
        # Log to WandB
        if self.use_wandb:
            print("\n📤 Logging to WandB...")
            
            # Log metrics exactly like wonderful-ml repo (matching unified_eot_streaming.py)
            # Spread the entire classification report dict directly
            wandb_metrics = {
                **metrics,  # This includes "True", "False", "accuracy", "macro avg", "weighted avg", tp, tn, fp, fn, auc, latency metrics
            }
            
            # Log to both metrics and summary
            wandb.log(wandb_metrics)
            for key, value in wandb_metrics.items():
                if not isinstance(value, (wandb.Image, wandb.Table)):
                    wandb.summary[key] = value
            
            # Create and log confusion matrix image
            cm = np.array([
                [metrics['tn'], metrics['fp']],
                [metrics['fn'], metrics['tp']]
            ])
            cm_fig = self._create_wandb_confusion_matrix(cm)
            wandb.log({"confusion_matrix": wandb.Image(cm_fig)})
            plt.close(cm_fig)
            
            # Log CSV artifact
            artifact = wandb.Artifact(
                name=f"smart_turn_v3_results_{wandb.run.id}",
                type="results",
                description="Smart Turn v3 Hebrew evaluation results"
            )
            artifact.add_file(str(results_path))
            artifact.add_file(str(metrics_path))
            if report_path:
                artifact.add_file(str(report_path))
            wandb.log_artifact(artifact)
            
            # Log results table
            if self.results:
                results_table = wandb.Table(dataframe=pd.DataFrame(self.results))
                wandb.log({"results_sample": results_table})
            
            print(f"✅ Logged to WandB: {wandb.run.url}")
        
        return {
            'metrics': metrics_path,
            'results': results_path,
            'report': report_path
        }
    
    def _create_wandb_confusion_matrix(self, cm: np.ndarray) -> plt.Figure:
        """Create confusion matrix figure for WandB (matching wonderful-ml format)."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Incomplete (Non-EOT)', 'Complete (EOT)'],
            yticklabels=['Incomplete (Non-EOT)', 'Complete (EOT)'],
            ax=ax,
            cbar_kws={'label': 'Count'}
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix: Smart Turn v3 Hebrew', fontsize=14, fontweight='bold', pad=20)
        
        # Add accuracy annotation
        accuracy = np.trace(cm) / np.sum(cm)
        ax.text(
            0.5, -0.15, 
            f'Accuracy: {accuracy:.2%}',
            transform=ax.transAxes,
            ha='center',
            fontsize=11,
            style='italic'
        )
        
        plt.tight_layout()
        return fig
    
    def print_summary(self, metrics: Dict):
        """Print a summary of results to console."""
        print("\n" + "=" * 60)
        print("🎯 SMART TURN V3 HEBREW BENCHMARK SUMMARY")
        print("=" * 60)
        
        print(f"⚙️  Configuration:")
        print(f"   • Threshold:   {metrics['threshold']:.2f}")
        
        print(f"\n📊 Dataset: {metrics['total_files']} files")
        print(f"   • Non-EOT (False): {metrics['class_distribution']['incomplete_count']} "
              f"({metrics['class_distribution']['incomplete_percentage']:.1f}%)")
        print(f"   • EOT (True): {metrics['class_distribution']['complete_count']} "
              f"({metrics['class_distribution']['complete_percentage']:.1f}%)")
        
        print(f"\n🎯 Overall Metrics:")
        print(f"   • Accuracy:    {metrics['accuracy']:.3f}")
        print(f"   • AUC:         {metrics['auc']:.3f}")
        
        print(f"\n📊 Weighted Average Metrics:")
        print(f"   • Precision:   {metrics['weighted avg']['precision']:.3f}")
        print(f"   • Recall:      {metrics['weighted avg']['recall']:.3f}")
        print(f"   • F1-Score:    {metrics['weighted avg']['f1-score']:.3f}")
        
        print(f"\n📉 Per-Class Metrics:")
        print(f"   False (Non-EOT): P={metrics['False']['precision']:.3f}, "
              f"R={metrics['False']['recall']:.3f}, F1={metrics['False']['f1-score']:.3f}")
        print(f"   True (EOT):      P={metrics['True']['precision']:.3f}, "
              f"R={metrics['True']['recall']:.3f}, F1={metrics['True']['f1-score']:.3f}")
        
        print(f"\n📈 Confusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"   • True Negatives:  {cm['true_negatives']}")
        print(f"   • False Positives: {cm['false_positives']}")
        print(f"   • False Negatives: {cm['false_negatives']}")
        print(f"   • True Positives:  {cm['true_positives']}")
        
        print(f"\n⚡ Latency:")
        print(f"   • Mean:   {metrics['latency_mean']*1000:.1f}ms")
        print(f"   • Median: {metrics['latency_median']*1000:.1f}ms")
        print(f"   • P95:    {metrics['latency_p95']*1000:.1f}ms")
        print(f"   • P99:    {metrics['latency_p99']*1000:.1f}ms")
        
        print(f"\n⚡ Performance:")
        perf = metrics['performance']
        print(f"   • Total processing:    {perf['total_processing_time_s']:.1f}s")
        print(f"   • Speed ratio:         {perf['processing_speed_ratio']:.1f}x real-time")
        
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Smart Turn v3 model on Hebrew audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Command line file:label pairs
  python benchmark_hebrew.py --files audio1.wav:1 audio2.wav:0 audio3.wav:1
  
  # CSV file with filename,label columns  
  python benchmark_hebrew.py --csv labels.csv
  
  # Directory with complete/ and incomplete/ subdirectories
  python benchmark_hebrew.py --directory path/to/hebrew/audio/
  
  # JSON file with list of {file, label} objects
  python benchmark_hebrew.py --json dataset.json
  
  # Custom threshold (recommended 0.3 for Hebrew based on benchmark analysis)
  python benchmark_hebrew.py --directory path/to/hebrew/audio/ --threshold 0.3
  
  # Custom output directory
  python benchmark_hebrew.py --csv labels.csv --output results_hebrew
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--files', nargs='+', 
                           help='Audio files with labels (format: file1.wav:1 file2.wav:0)')
    input_group.add_argument('--csv', 
                           help='CSV file with filename,label columns')
    input_group.add_argument('--directory', 
                           help='Directory with complete/ and incomplete/ subdirectories')
    input_group.add_argument('--json', 
                           help='JSON file with list of {file, label} objects')
    
    # Output options
    parser.add_argument('--output', default='benchmark_results',
                       help='Output directory for results (default: benchmark_results)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip generating visualizations')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimize console output')
    
    # WandB options
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable WandB logging')
    parser.add_argument('--wandb-project', default='eot-evaluations',
                       help='WandB project name (default: eot-evaluations)')
    parser.add_argument('--wandb-run-name', type=str,
                       help='WandB run name (default: auto-generated with timestamp)')
    parser.add_argument('--wandb-tags', nargs='+',
                       help='Additional WandB tags (default: smart-turn-v3, hebrew, audio-native-eot)')
    
    # Model options
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='EOT detection threshold (default: 0.5, recommended: 0.3 for Hebrew)')
    
    args = parser.parse_args()
    
    if not INFERENCE_AVAILABLE:
        print("❌ Error: Smart Turn inference not available")
        print("   Make sure you have:")
        print("   1. The smart-turn-v3.0.onnx model file")
        print("   2. Required dependencies (onnxruntime, transformers)")
        print("   3. The inference.py file")
        sys.exit(1)
    
    try:
        # Prepare WandB config
        wandb_config = {
            "input_method": None,
            "input_path": None,
            "total_files": 0,
        }
        
        # Determine input method and path for config
        if args.files:
            wandb_config["input_method"] = "command_line"
            wandb_config["total_files"] = len(args.files)
        elif args.csv:
            wandb_config["input_method"] = "csv"
            wandb_config["input_path"] = args.csv
        elif args.directory:
            wandb_config["input_method"] = "directory"
            wandb_config["input_path"] = args.directory
        elif args.json:
            wandb_config["input_method"] = "json"
            wandb_config["input_path"] = args.json
        
        # Initialize benchmark with WandB parameters
        wandb_tags = args.wandb_tags or ["smart-turn-v3", "hebrew", "audio-native-eot"]
        benchmark = HebrewBenchmark(
            output_dir=args.output,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            wandb_tags=wandb_tags,
            wandb_config=wandb_config,
            threshold=args.threshold
        )
        
        # Load files based on input method
        if args.files:
            files = benchmark.load_files_from_args(args.files)
        elif args.csv:
            files = benchmark.load_files_from_csv(args.csv)
        elif args.directory:
            files = benchmark.load_files_from_directory(args.directory)
        elif args.json:
            files = benchmark.load_files_from_json(args.json)
        
        if not files:
            print("❌ No valid files found to process")
            sys.exit(1)
        
        print(f"✅ Loaded {len(files)} files for benchmarking")
        
        # Update WandB config with actual file count
        if benchmark.use_wandb:
            wandb.config.update({"total_files": len(files)}, allow_val_change=True)
        
        # Run benchmark
        metrics = benchmark.run_benchmark(files)
        
        # Generate outputs
        if not args.no_viz:
            benchmark.generate_visualizations(metrics)
        
        file_paths = benchmark.save_results(metrics)
        
        if not args.quiet:
            benchmark.print_summary(metrics)
        
        print(f"\n✅ Benchmark completed successfully!")
        print(f"   Results saved to: {benchmark.output_dir}")
        
        # Finish WandB run
        if benchmark.use_wandb:
            wandb.finish()
            print(f"✅ WandB run finished")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        if not args.no_wandb:
            wandb.finish(exit_code=1)
        sys.exit(1)


if __name__ == "__main__":
    main()

