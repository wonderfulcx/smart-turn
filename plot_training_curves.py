#!/usr/bin/env python3
"""Plot training and validation curves from trainer_state.json"""

import json
import sys
import matplotlib.pyplot as plt
from pathlib import Path

def plot_training_curves(checkpoint_dir):
    """Plot training and validation metrics."""
    trainer_state_path = Path(checkpoint_dir) / "trainer_state.json"
    
    if not trainer_state_path.exists():
        print(f"❌ Error: {trainer_state_path} not found")
        return
    
    print(f"📊 Loading metrics from: {trainer_state_path}")
    
    with open(trainer_state_path) as f:
        state = json.load(f)
    
    # Extract metrics
    train_steps = []
    train_losses = []
    eval_steps = []
    eval_losses = []
    eval_f1 = []
    eval_acc = []
    
    for entry in state['log_history']:
        if 'loss' in entry and 'eval_loss' not in entry:
            train_steps.append(entry['step'])
            train_losses.append(entry['loss'])
        elif 'eval_loss' in entry:
            eval_steps.append(entry['step'])
            eval_losses.append(entry['eval_loss'])
            eval_f1.append(entry.get('eval_f1', 0))
            eval_acc.append(entry.get('eval_accuracy', 0))
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Hebrew Model Training Curves', fontsize=16, fontweight='bold')
    
    # Plot 1: Training & Validation Loss
    ax1 = axes[0, 0]
    ax1.plot(train_steps, train_losses, 'b-', alpha=0.7, label='Training Loss', linewidth=2)
    ax1.plot(eval_steps, eval_losses, 'r-o', label='Validation Loss', linewidth=2, markersize=8)
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation F1 Score
    ax2 = axes[0, 1]
    ax2.plot(eval_steps, eval_f1, 'g-o', label='F1 Score', linewidth=2, markersize=8)
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title('Validation F1 Score', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Plot 3: Validation Accuracy
    ax3 = axes[1, 0]
    ax3.plot(eval_steps, eval_acc, 'm-o', label='Accuracy', linewidth=2, markersize=8)
    ax3.set_xlabel('Training Steps', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # Plot 4: Learning Rate Schedule
    ax4 = axes[1, 1]
    lr_steps = []
    learning_rates = []
    for entry in state['log_history']:
        if 'learning_rate' in entry:
            lr_steps.append(entry['step'])
            learning_rates.append(entry['learning_rate'])
    
    if learning_rates:
        ax4.plot(lr_steps, learning_rates, 'orange', linewidth=2)
        ax4.set_xlabel('Training Steps', fontsize=12)
        ax4.set_ylabel('Learning Rate', fontsize=12)
        ax4.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(checkpoint_dir).parent / "training_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_path}")
    
    # Print summary
    print("\n📈 Training Summary:")
    print(f"   Total training steps: {max(train_steps)}")
    print(f"   Number of evaluations: {len(eval_steps)}")
    print(f"   Final training loss: {train_losses[-1]:.4f}")
    if eval_losses:
        print(f"   Final validation loss: {eval_losses[-1]:.4f}")
        print(f"   Final F1 score: {eval_f1[-1]:.4f}")
        print(f"   Final accuracy: {eval_acc[-1]:.4f}")
    
    # Check for underfitting/overfitting
    print("\n🔍 Analysis:")
    if len(eval_losses) < 3:
        print("   ⚠️  Warning: Only {} evaluation(s) - need more frequent eval for proper analysis".format(len(eval_losses)))
    
    if len(eval_losses) >= 2:
        if eval_losses[-1] < eval_losses[-2]:
            print("   📉 Validation loss is decreasing - model is still learning!")
        elif eval_losses[-1] > eval_losses[-2]:
            print("   📈 Validation loss is increasing - possible overfitting")
    
    if train_losses[-1] < train_losses[0] * 0.9:
        print("   ✅ Training loss decreased significantly (good)")
    
    if len(eval_losses) >= 1 and train_losses[-1] > eval_losses[-1]:
        print("   ⚠️  Training loss > Validation loss - possible underfitting")
    
    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_training_curves.py <checkpoint_directory>")
        print("\nExample:")
        print("  python plot_training_curves.py ./output/v3.1-hebrew-full-20260106-0846/checkpoint-588")
        print("\nOr find the latest checkpoint:")
        print("  python plot_training_curves.py $(find ./output -name 'checkpoint-*' -type d | tail -1)")
        sys.exit(1)
    
    checkpoint_dir = sys.argv[1]
    plot_training_curves(checkpoint_dir)

