#!/usr/bin/env python3
"""
Simple test to verify core components work correctly.

This tests the new DatasetManager, ModelManager, and EmbeddingExtractor
without requiring actual models or data.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_imports():
    """Test that all modules can be imported."""
    print("=" * 60)
    print("TEST 1: Imports")
    print("=" * 60)

    try:
        from pipeline_refactored.core import (
            DatasetManager,
            ModelManager,
            EmbeddingExtractor,
            AnalysisContext,
            StageResult,
        )
        print("✓ All core modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_context_creation():
    """Test creating context and result objects."""
    print("\n" + "=" * 60)
    print("TEST 2: Context Creation")
    print("=" * 60)

    try:
        from pipeline_refactored.core import AnalysisContext, StageResult
        import numpy as np

        # Create context
        context = AnalysisContext(
            embeddings={"uniform": np.random.randn(100, 500)},
            features={"labels": np.arange(100)},
            architecture="iDBN_1500_500",
            distribution="uniform",
        )

        print(f"✓ Created context: {context}")
        print(f"  - Has embedding 'uniform': {context.has_embedding('uniform')}")
        print(f"  - Has feature 'labels': {context.has_feature('labels')}")

        # Create result
        result = StageResult(stage_name="test_stage")
        result.add_metric("accuracy", 0.95)
        result.add_artifact("test_data", [1, 2, 3])
        result.add_metadata("duration", 1.5)

        print(f"✓ Created result: {result}")
        print(f"  - Metrics: {result.metrics}")

        return True
    except Exception as e:
        print(f"✗ Context creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_manager():
    """Test ModelManager basic functionality."""
    print("\n" + "=" * 60)
    print("TEST 3: ModelManager")
    print("=" * 60)

    try:
        from pipeline_refactored.core import ModelManager

        mm = ModelManager()
        print(f"✓ Created ModelManager: {mm}")
        print(f"  - Loaded models: {mm.list_models()}")
        print(f"  - Number of models: {len(mm)}")

        # Test info for non-existent model
        try:
            mm.get_model("nonexistent")
            print("✗ Should have raised KeyError for non-existent model")
            return False
        except KeyError:
            print("✓ Correctly raised KeyError for non-existent model")

        return True
    except Exception as e:
        print(f"✗ ModelManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_manager():
    """Test DatasetManager basic functionality."""
    print("\n" + "=" * 60)
    print("TEST 4: DatasetManager")
    print("=" * 60)

    try:
        from pipeline_refactored.core import DatasetManager

        # Note: This will fail if dataset doesn't exist, but tests the API
        dm = DatasetManager("dummy/path", "dummy_dataset.npz")
        print(f"✓ Created DatasetManager: {dm}")

        # Test that methods exist and have correct signatures
        assert hasattr(dm, "get_dataloader"), "Missing get_dataloader method"
        assert hasattr(dm, "get_features"), "Missing get_features method"
        assert hasattr(dm, "get_info"), "Missing get_info method"

        print("✓ DatasetManager has all required methods")

        return True
    except Exception as e:
        print(f"✗ DatasetManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embedding_extractor():
    """Test EmbeddingExtractor basic functionality."""
    print("\n" + "=" * 60)
    print("TEST 5: EmbeddingExtractor")
    print("=" * 60)

    try:
        from pipeline_refactored.core import EmbeddingExtractor, ModelManager

        mm = ModelManager()
        extractor = EmbeddingExtractor(mm)
        print(f"✓ Created EmbeddingExtractor: {extractor}")

        # Test that methods exist
        assert hasattr(extractor, "extract"), "Missing extract method"
        assert hasattr(extractor, "extract_aligned_pair"), "Missing extract_aligned_pair method"
        assert hasattr(extractor, "extract_layerwise"), "Missing extract_layerwise method"
        assert hasattr(extractor, "reconstruct"), "Missing reconstruct method"

        print("✓ EmbeddingExtractor has all required methods")

        return True
    except Exception as e:
        print(f"✗ EmbeddingExtractor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("GROUNDEEP CORE COMPONENTS - Simple Tests")
    print("=" * 60)

    tests = [
        test_imports,
        test_context_creation,
        test_model_manager,
        test_dataset_manager,
        test_embedding_extractor,
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test {test_func.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print(f"✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
