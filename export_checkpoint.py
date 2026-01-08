#!/usr/bin/env python3
"""Export a checkpoint to ONNX format."""

import argparse
from pathlib import Path

from train import SmartTurnV3Model, CONFIG, export_to_onnx_fp32
from logger import log


def main():
    parser = argparse.ArgumentParser(description="Export checkpoint to ONNX")
    parser.add_argument("checkpoint_path", help="Path to checkpoint directory")
    parser.add_argument("--output", "-o", help="Output ONNX file path (default: checkpoint name + .onnx)")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Determine output path - use version from CONFIG
    if args.output:
        output_path = args.output
    else:
        version = CONFIG["run_name_prefix"]
        output_path = f"smart-turn-{version}-hebrew-{checkpoint_path.name}.onnx"

    log.info(f"Loading checkpoint from: {checkpoint_path}")
    log.info(f"Output ONNX file: {output_path}")

    # Load the model from checkpoint using from_pretrained
    # This automatically loads the config and weights
    model = SmartTurnV3Model.from_pretrained(
        str(checkpoint_path),
        num_labels=1,
        ignore_mismatched_sizes=True
    )
    log.info(f"Loaded model from {checkpoint_path}")

    model.eval()

    # Export to ONNX
    export_to_onnx_fp32(model, output_path, CONFIG)
    
    # Print file size
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    log.info(f"📦 ONNX file size: {file_size_mb:.2f} MB")


if __name__ == "__main__":
    main()

