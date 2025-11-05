#!/usr/bin/env python3
"""
Migration Example: Old Embedding_analysis → New 3-Class System

This shows how to replace the monolithic Embedding_analysis class
with the cleaner DatasetManager + ModelManager + EmbeddingExtractor pattern.

The new system offers:
- Lazy loading (only load what you need)
- Clear separation of concerns
- Better testability
- More flexible (can extract from single model, aligned pairs, layer-wise, etc.)
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def old_way_example():
    """
    OLD WAY: Using Embedding_analysis (monolithic class)

    Problems:
    - Loads everything in constructor (side effects)
    - Creates 4 dataloaders even if you only need 1
    - Hard to test individual components
    - Does too much (data + models + embeddings)
    """
    print("=" * 60)
    print("OLD WAY: Embedding_analysis")
    print("=" * 60)

    from src.analyses.embedding_analysis import Embedding_analysis

    # Setup (you MUST have actual model files for this to work)
    path2data = "path/to/data"
    data_name = "dataset.npz"
    model_uniform = "path/to/uniform_model.pkl"
    model_zipfian = "path/to/zipfian_model.pkl"
    arch_name = "iDBN_1500_500"

    # Everything happens in constructor!
    # - Creates 4 dataloaders (uniform train/val/test + zipfian train/val/test)
    # - Loads 2 models
    # - Even if you only need embeddings from val split
    analyzer = Embedding_analysis(
        path2data, data_name,
        model_uniform, model_zipfian,
        arch_name,
        val_size=0.05
    )

    # Extract embeddings (hidden in private method)
    output_dict = analyzer._get_encodings()

    # Access results
    Z_uniform = output_dict['Z_uniform']
    Z_zipfian = output_dict['Z_zipfian']
    labels = output_dict['labels_uniform']
    cumArea = output_dict['cumArea_uniform']
    CH = output_dict['CH_uniform']

    print(f"✓ Extracted embeddings:")
    print(f"  Z_uniform: {Z_uniform.shape}")
    print(f"  Z_zipfian: {Z_zipfian.shape}")
    print(f"  labels: {labels.shape}")
    print(f"  cumArea: {cumArea.shape}")
    print(f"  CH: {CH.shape}")

    # For reconstruction
    original, reconstructed = analyzer.reconstruct_input(input_type="uniform")
    print(f"\n✓ Reconstruction:")
    print(f"  Original: {original.shape}")
    print(f"  Reconstructed: {reconstructed.shape}")


def new_way_example():
    """
    NEW WAY: Using DatasetManager + ModelManager + EmbeddingExtractor

    Benefits:
    - Lazy loading (only create what you need)
    - Clear separation of concerns
    - Testable components
    - More flexible (extract from single model, aligned pairs, layer-wise, etc.)
    """
    print("\n" + "=" * 60)
    print("NEW WAY: 3-Class System")
    print("=" * 60)

    from pipeline_refactored.core import (
        DatasetManager,
        ModelManager,
        EmbeddingExtractor,
        AnalysisContext,
    )

    # Setup (you MUST have actual model files for this to work)
    path2data = "path/to/data"
    data_name = "dataset.npz"
    model_uniform = "path/to/uniform_model.pkl"
    model_zipfian = "path/to/zipfian_model.pkl"

    # Step 1: Create managers (NO side effects, NO loading yet)
    print("\n[1/4] Creating managers...")
    dataset_mgr = DatasetManager(path2data, data_name, default_val_size=0.05)
    model_mgr = ModelManager()
    print(f"  ✓ Created: {dataset_mgr}")
    print(f"  ✓ Created: {model_mgr}")

    # Step 2: Load models (lazy, only when needed)
    print("\n[2/4] Loading models...")
    model_mgr.load_model(model_uniform, label='uniform')
    model_mgr.load_model(model_zipfian, label='zipfian')
    print(f"  ✓ Loaded models: {model_mgr.list_models()}")

    # Step 3: Get dataloader (lazy, only creates uniform val split)
    print("\n[3/4] Getting dataloader...")
    val_loader = dataset_mgr.get_dataloader('uniform', split='val')
    print(f"  ✓ Created val_loader with {len(val_loader.dataset)} samples")

    # Step 4: Extract embeddings (aligned pair for fair comparison)
    print("\n[4/4] Extracting embeddings...")
    extractor = EmbeddingExtractor(model_mgr)
    Z_uniform, Z_zipfian = extractor.extract_aligned_pair(
        'uniform', 'zipfian', val_loader, verbose=True
    )

    # Extract features
    features = dataset_mgr.get_features('uniform', split='val')
    labels = features['labels']
    cumArea = features['cum_area']
    CH = features['convex_hull']

    print(f"\n✓ Results:")
    print(f"  Z_uniform: {Z_uniform.shape}")
    print(f"  Z_zipfian: {Z_zipfian.shape}")
    print(f"  labels: {labels.shape}")
    print(f"  cumArea: {cumArea.shape}")
    print(f"  CH: {CH.shape}")

    # Reconstruction (cleaner API)
    print(f"\n✓ Reconstruction:")
    original, reconstructed = extractor.reconstruct('uniform', val_loader, n_samples=100)
    print(f"  Original: {original.shape}")
    print(f"  Reconstructed: {reconstructed.shape}")

    # Bonus: Create AnalysisContext for pipeline use
    print(f"\n✓ Creating AnalysisContext for pipeline...")
    context = AnalysisContext(
        embeddings={
            'uniform': Z_uniform,
            'zipfian': Z_zipfian,
        },
        features={
            'labels': labels,
            'cum_area': cumArea,
            'convex_hull': CH,
        },
        architecture='iDBN_1500_500',
        distribution='uniform',
    )
    print(f"  Context: {context}")
    print(f"  Has uniform embedding: {context.has_embedding('uniform')}")
    print(f"  Has zipfian embedding: {context.has_embedding('zipfian')}")
    print(f"  Has labels: {context.has_feature('labels')}")


def advanced_example_single_model():
    """
    ADVANCED: Extract from single model only

    Use case: You only want embeddings from one model
    """
    print("\n" + "=" * 60)
    print("ADVANCED: Single Model Extraction")
    print("=" * 60)

    from pipeline_refactored.core import (
        DatasetManager,
        ModelManager,
        EmbeddingExtractor,
    )

    # Setup
    dataset_mgr = DatasetManager("path/to/data", "dataset.npz")
    model_mgr = ModelManager()

    # Load only uniform model
    model_mgr.load_model("path/to/uniform_model.pkl", label='uniform')

    # Get dataloader
    val_loader = dataset_mgr.get_dataloader('uniform', split='val')

    # Extract from single model
    extractor = EmbeddingExtractor(model_mgr)
    Z_uniform = extractor.extract('uniform', val_loader, verbose=True)

    print(f"✓ Extracted from single model: {Z_uniform.shape}")


def advanced_example_layerwise():
    """
    ADVANCED: Layer-wise extraction

    Use case: You want embeddings from all layers (for geometric analysis)
    """
    print("\n" + "=" * 60)
    print("ADVANCED: Layer-wise Extraction")
    print("=" * 60)

    from pipeline_refactored.core import (
        DatasetManager,
        ModelManager,
        EmbeddingExtractor,
    )

    # Setup
    dataset_mgr = DatasetManager("path/to/data", "dataset.npz")
    model_mgr = ModelManager()
    model_mgr.load_model("path/to/uniform_model.pkl", label='uniform')

    # Get dataloader
    val_loader = dataset_mgr.get_dataloader('uniform', split='val')

    # Extract all layers
    extractor = EmbeddingExtractor(model_mgr)
    layer_embeddings = extractor.extract_layerwise(
        'uniform', val_loader, layers=None, verbose=True
    )

    print(f"✓ Extracted {len(layer_embeddings)} layers:")
    for layer_idx, Z in layer_embeddings.items():
        print(f"  Layer {layer_idx}: {Z.shape}")


def comparison_summary():
    """
    Side-by-side comparison of OLD vs NEW
    """
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    comparison = """
    OLD WAY (Embedding_analysis):
    ────────────────────────────────────────────────────────
    ✗ Everything in constructor (side effects)
    ✗ Creates 4 dataloaders even if only need 1
    ✗ Loads 2 models always
    ✗ Hard to test components independently
    ✗ 292 lines in one class
    ✗ Can't easily extend (e.g., add third model)

    Code:
        analyzer = Embedding_analysis(path, data, model_u, model_z, arch)
        output = analyzer._get_encodings()
        Z_u = output['Z_uniform']
        Z_z = output['Z_zipfian']


    NEW WAY (3-Class System):
    ────────────────────────────────────────────────────────
    ✓ Lazy loading (only create what you need)
    ✓ Clear separation of concerns
    ✓ Each component testable independently
    ✓ ~250 lines across 3 focused classes
    ✓ Easy to extend (add models, distributions, layers)
    ✓ Flexible (single model, aligned pairs, layer-wise)
    ✓ Works for training AND analysis

    Code:
        # Setup
        dm = DatasetManager(path, data)
        mm = ModelManager()
        mm.load_model(model_u, 'uniform')
        mm.load_model(model_z, 'zipfian')

        # Extract
        val_loader = dm.get_dataloader('uniform', split='val')
        extractor = EmbeddingExtractor(mm)
        Z_u, Z_z = extractor.extract_aligned_pair('uniform', 'zipfian', val_loader)
        features = dm.get_features('uniform', split='val')


    MIGRATION PATH:
    ────────────────────────────────────────────────────────
    1. Keep old Embedding_analysis for backward compatibility
    2. Update analyze.py to use new system
    3. Update training code to use new system
    4. Eventually deprecate Embedding_analysis
    """

    print(comparison)


def create_backward_compatible_helper():
    """
    BONUS: Helper function to create old-style output_dict from new system

    Use case: You want to migrate gradually without changing downstream code
    """
    print("\n" + "=" * 60)
    print("BONUS: Backward Compatible Helper")
    print("=" * 60)

    from pipeline_refactored.core import create_backward_compatible_bundle

    # Suppose you already have the new system results
    Z_uniform = np.random.randn(100, 500)
    Z_zipfian = np.random.randn(100, 500)
    labels = np.arange(100)
    cumArea = np.random.rand(100)
    CH = np.random.rand(100)

    # Create old-style output_dict
    output_dict = create_backward_compatible_bundle(
        Z_uniform=Z_uniform,
        Z_zipfian=Z_zipfian,
        labels=labels,
        cumArea=cumArea,
        CH=CH,
    )

    print("✓ Created backward compatible output_dict:")
    print(f"  Keys: {list(output_dict.keys())}")
    print(f"  Z_uniform: {output_dict['Z_uniform'].shape}")
    print(f"  Z_zipfian: {output_dict['Z_zipfian'].shape}")
    print(f"  labels_uniform: {output_dict['labels_uniform'].shape}")
    print(f"  labels_zipfian: {output_dict['labels_zipfian'].shape}")
    print(f"  cumArea_uniform: {output_dict['cumArea_uniform'].shape}")
    print(f"  CH_uniform: {output_dict['CH_uniform'].shape}")
    print("\n  This matches the old Embedding_analysis._get_encodings() output!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MIGRATION GUIDE: Embedding_analysis → 3-Class System")
    print("=" * 60)
    print("\nThis file demonstrates how to replace the old Embedding_analysis")
    print("with the new DatasetManager + ModelManager + EmbeddingExtractor.\n")
    print("NOTE: The examples below use dummy paths. To actually run them,")
    print("      replace with real model paths and data paths from your project.\n")

    # Show comparison
    comparison_summary()

    # Show examples (commented out since we don't have real models here)
    print("\n" + "=" * 60)
    print("EXAMPLE CODE (see functions above for details)")
    print("=" * 60)
    print("""
To run the examples:
    1. Update paths to point to your actual models and data
    2. Uncomment the function calls below
    3. Run: python migrate_from_embedding_analysis.py

Examples:
    - old_way_example()              # See the old monolithic approach
    - new_way_example()              # See the new 3-class approach
    - advanced_example_single_model() # Extract from one model only
    - advanced_example_layerwise()   # Extract all layers
    - create_backward_compatible_helper() # Gradual migration helper
    """)

    # Uncomment to run with real models:
    # old_way_example()
    # new_way_example()
    # advanced_example_single_model()
    # advanced_example_layerwise()
    # create_backward_compatible_helper()
