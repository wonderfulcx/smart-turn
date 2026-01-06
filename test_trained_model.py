#!/usr/bin/env python3
"""Quick test script for the trained Hebrew model."""

import onnxruntime as ort
import numpy as np
from transformers import WhisperFeatureExtractor
import librosa

# Path to your trained model
MODEL_PATH = "./output/v3.1-hebrew-full-20260106-0846/final_model/exports/model_fp32.onnx"

def test_model(audio_path):
    """Test the trained model on an audio file."""
    print(f"Loading model from: {MODEL_PATH}")
    
    # Create ONNX session with GPU
    session = ort.InferenceSession(
        MODEL_PATH,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    
    print(f"Testing on: {audio_path}")
    
    # Load and preprocess audio
    feature_extractor = WhisperFeatureExtractor(chunk_length=8)
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Extract features
    features = feature_extractor(
        audio, 
        sampling_rate=16000, 
        return_tensors="np",
        padding="max_length",
        max_length=128000  # 8 seconds at 16kHz
    )
    
    # Run inference
    input_features = features['input_features']
    outputs = session.run(None, {'input_features': input_features})
    
    # Get prediction
    logit = outputs[0][0][0]
    probability = 1 / (1 + np.exp(-logit))  # Sigmoid
    
    print(f"\n🎯 Results:")
    print(f"   Logit: {logit:.4f}")
    print(f"   Probability: {probability:.4f}")
    print(f"   Prediction: {'✅ COMPLETE' if probability > 0.5 else '⏸️  INCOMPLETE'}")
    
    return probability

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_trained_model.py <audio_file.wav>")
        print("\nExample:")
        print("  python test_trained_model.py ./datasets/output/smart-turn-hebrew-test/heb_complete-nofiller_*.flac")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    test_model(audio_file)

