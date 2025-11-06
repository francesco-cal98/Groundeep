#!/usr/bin/env python3
"""
Quick test script to validate GROUNDEEP pipeline setup.

Run this after installation to check if everything works.
"""

import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_imports():
    """Test that all required modules can be imported."""
    print("="*70)
    print("Testing imports...")
    print("="*70)

    tests = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("torch", "PyTorch"),
        ("sklearn", "scikit-learn"),
        ("scipy", "SciPy"),
        ("matplotlib", "Matplotlib"),
        ("hydra", "Hydra"),
        ("omegaconf", "OmegaConf"),
    ]

    failed = []
    for module, name in tests:
        try:
            __import__(module)
            print(f"✓ {name:20s} OK")
        except ImportError as e:
            print(f"✗ {name:20s} FAILED: {e}")
            failed.append(name)

    # Test optional dependencies
    optional_tests = [
        ("wandb", "WandB (optional)"),
        ("umap", "UMAP (optional)"),
        ("skimage", "scikit-image (optional)"),
    ]

    print("\nOptional dependencies:")
    for module, name in optional_tests:
        try:
            __import__(module)
            print(f"✓ {name:20s} OK")
        except ImportError:
            print(f"○ {name:20s} Not installed (optional)")

    if failed:
        print(f"\n❌ {len(failed)} required dependencies missing: {', '.join(failed)}")
        return False
    else:
        print("\n✅ All required dependencies installed!")
        return True


def test_pipeline_stages():
    """Test that pipeline stages can be imported."""
    print("\n" + "="*70)
    print("Testing pipeline stages...")
    print("="*70)

    try:
        from pipeline_refactored.stages import (
            PowerLawStage,
            LinearProbesStage,
            GeometryStage,
            ReconstructionStage,
            DimensionalityStage,
            CKAStage,
            BehavioralStage,
        )

        stages = [
            ("PowerLawStage", PowerLawStage),
            ("LinearProbesStage", LinearProbesStage),
            ("GeometryStage", GeometryStage),
            ("ReconstructionStage", ReconstructionStage),
            ("DimensionalityStage", DimensionalityStage),
            ("CKAStage", CKAStage),
            ("BehavioralStage", BehavioralStage),
        ]

        for name, stage_class in stages:
            stage = stage_class()
            print(f"✓ {name:25s} OK")

        print("\n✅ All pipeline stages loaded successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Failed to load pipeline stages: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test that configuration files exist and are valid."""
    print("\n" + "="*70)
    print("Testing configuration...")
    print("="*70)

    config_path = PROJECT_ROOT / "src" / "configs" / "analysis.yaml"

    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False

    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(config_path)

        # Check required fields
        required_fields = ["seed", "output_root", "models"]
        missing = [f for f in required_fields if f not in cfg]

        if missing:
            print(f"❌ Missing required fields: {', '.join(missing)}")
            return False

        print(f"✓ Config file found: {config_path}")
        print(f"✓ Config has {len(cfg.models)} model(s) configured")
        print(f"✓ Output directory: {cfg.output_root}")
        print("\n✅ Configuration valid!")
        return True

    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return False


def test_directory_structure():
    """Test that required directories exist."""
    print("\n" + "="*70)
    print("Testing directory structure...")
    print("="*70)

    required_dirs = [
        "src",
        "src/main_scripts",
        "src/configs",
        "src/analyses",
        "pipeline_refactored",
        "pipeline_refactored/stages",
        "pipeline_refactored/core",
    ]

    missing = []
    for dir_path in required_dirs:
        full_path = PROJECT_ROOT / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} MISSING")
            missing.append(dir_path)

    if missing:
        print(f"\n❌ {len(missing)} required directories missing")
        return False
    else:
        print("\n✅ Directory structure OK!")
        return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("GROUNDEEP Pipeline Test Suite")
    print("="*70 + "\n")

    results = {
        "Imports": test_imports(),
        "Directory Structure": test_directory_structure(),
        "Configuration": test_config(),
        "Pipeline Stages": test_pipeline_stages(),
    }

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:25s} {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED! Your setup is ready.")
        print("="*70)
        print("\nNext steps:")
        print("  1. Check your config: src/configs/analysis.yaml")
        print("  2. Run analysis: python src/main_scripts/analyze_modular.py")
        print("  3. See README.md for more information")
        return 0
    else:
        print("\n" + "="*70)
        print("⚠️  SOME TESTS FAILED. Please fix the issues above.")
        print("="*70)
        print("\nTroubleshooting:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Check Python version: python --version (need >= 3.8)")
        print("  - See README.md for detailed setup instructions")
        return 1


if __name__ == "__main__":
    sys.exit(main())
