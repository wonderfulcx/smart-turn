#!/usr/bin/env python3
"""Export a checkpoint to ONNX format."""

import argparse
import torch
import onnx
import numpy as np
from pathlib import Path
from torch.export import Dim

from train import SmartTurnV3Model, CONFIG
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def export_to_onnx_fp32(model, output_path, config):
    """Export model to ONNX FP32 format"""
    log.info("Exporting model to ONNX FP32...")

    class ONNXExportWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_features):
            out = self.inner(input_features)
            logits = out["logits"] if isinstance(out, dict) else out
            batch_size = logits.shape[0]
            return logits.reshape(batch_size, 1)

    export_model = ONNXExportWrapper(model).eval().cpu()

    example_input_b1 = torch.randn(1, 80, 800)
    example_input_b2 = torch.randn(2, 80, 800)

    with torch.no_grad():
        test_output_1 = export_model(example_input_b1)
        test_output_2 = export_model(example_input_b2)
        assert test_output_1.shape == (1, 1), f"Expected (1, 1), got {test_output_1.shape}"
        assert test_output_2.shape == (2, 1), f"Expected (2, 1), got {test_output_2.shape}"

    dynamic_shapes = {
        'input_features': {0: Dim.DYNAMIC},
    }

    torch.onnx.export(
        model=export_model,
        args=(example_input_b2,),
        f=output_path,
        export_params=True,
        opset_version=config["onnx_opset_version"],
        do_constant_folding=False,
        input_names=['input_features'],
        output_names=['logits'],
        dynamic_shapes=dynamic_shapes,
        verbose=False,
        external_data=False,
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    log.info(f"FP32 ONNX model saved to {output_path}")

    # Verify the exported model
    import onnxruntime as ort
    session = ort.InferenceSession(output_path)

    example_input_1_np = example_input_b1.numpy().astype(np.float32)
    outputs_1 = session.run(None, {'input_features': example_input_1_np})
    assert outputs_1[0].shape == (1, 1), f"Expected (1, 1), got {outputs_1[0].shape}"

    example_input_2_np = example_input_b2.numpy().astype(np.float32)
    outputs_2 = session.run(None, {'input_features': example_input_2_np})
    assert outputs_2[0].shape == (2, 1), f"Expected (2, 1), got {outputs_2[0].shape}"

    log.info("✅ ONNX model verification passed!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Export checkpoint to ONNX")
    parser.add_argument("checkpoint_path", help="Path to checkpoint directory")
    parser.add_argument("--output", "-o", help="Output ONNX file path (default: checkpoint name + .onnx)")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = f"smart-turn-v3.1-hebrew-{checkpoint_path.name}.onnx"

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

