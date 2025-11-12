#!/usr/bin/env python3
"""
Quick test script to verify adapter system works with existing DBN models.
"""

import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_refactored.core.adapters import create_adapter, validate_adapter
from pipeline_refactored.core.model_manager import ModelManager

def test_adapter_system():
    """Test adapter system with real DBN model."""

    print("=" * 70)
    print("ADAPTER SYSTEM TEST")
    print("=" * 70)

    # Path to one of your models
    model_path = PROJECT_ROOT / "networks/uniform/dataset_10_10/uniform_i_/dbn_trained_uniform_1500_500.pkl"

    if not model_path.exists():
        print(f"❌ Model not found at: {model_path}")
        print("Please update the path in test_adapters.py")
        return False

    print(f"\n1. Loading model from: {model_path}")

    # Test ModelManager with adapters
    print("\n2. Testing ModelManager with adapter support...")
    mm = ModelManager(adapter_type="auto")

    try:
        model = mm.load_model(str(model_path), label="test_model")
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

    # Get adapter
    print("\n3. Getting adapter...")
    try:
        adapter = mm.get_adapter("test_model")
        print(f"✓ Adapter created: {adapter}")
    except Exception as e:
        print(f"❌ Failed to get adapter: {e}")
        return False

    # Test adapter metadata
    print("\n4. Adapter metadata:")
    try:
        metadata = adapter.get_metadata()
        for key, value in metadata.items():
            print(f"   {key}: {value}")
    except Exception as e:
        print(f"❌ Failed to get metadata: {e}")
        return False

    # Test encode
    print("\n5. Testing encode...")
    try:
        device = adapter.get_device()
        dummy_input = torch.randn(4, 10000).to(device)  # Batch of 4, input size 10000
        embeddings = adapter.encode(dummy_input)
        print(f"✓ Encode successful: {dummy_input.shape} -> {embeddings.shape}")
    except Exception as e:
        print(f"❌ Encode failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test layerwise
    print("\n6. Testing layerwise extraction...")
    try:
        layerwise_embeddings = adapter.encode_layerwise(dummy_input)
        print(f"✓ Layerwise successful: extracted {len(layerwise_embeddings)} layers")
        for i, emb in enumerate(layerwise_embeddings, 1):
            print(f"   Layer {i}: {emb.shape}")
    except Exception as e:
        print(f"❌ Layerwise failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test decode
    print("\n7. Testing decode (reconstruction)...")
    try:
        reconstructed = adapter.decode(embeddings)
        print(f"✓ Decode successful: {embeddings.shape} -> {reconstructed.shape}")

        # Check reconstruction error
        recon_error = torch.mean((dummy_input - reconstructed) ** 2).item()
        print(f"   Reconstruction MSE: {recon_error:.6f}")
    except Exception as e:
        print(f"❌ Decode failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Validate adapter
    print("\n8. Running adapter validation...")
    is_valid = validate_adapter(adapter, verbose=True)

    if is_valid:
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        return True
    else:
        print("\n" + "=" * 70)
        print("⚠️  SOME TESTS FAILED")
        print("=" * 70)
        return False

if __name__ == "__main__":
    success = test_adapter_system()
    sys.exit(0 if success else 1)
