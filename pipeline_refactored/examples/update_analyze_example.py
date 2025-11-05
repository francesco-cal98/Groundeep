#!/usr/bin/env python3
"""
Example: How to update analyze.py to use the new 3-class system

This shows a concrete before/after comparison of how the _prepare_model_context()
function in analyze.py should be updated to use the new architecture.
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# OLD VERSION (from analyze.py)
# ============================================================================

def _prepare_model_context_OLD(cfg: Dict[str, Any]) -> Any:
    """
    OLD: How analyze.py currently creates Embedding_analysis

    From src/main_scripts/analyze.py line ~1800
    """
    from src.analyses.embedding_analysis import Embedding_analysis

    # Extract config
    path2data = cfg['data']['path']
    data_name = cfg['data']['name']
    model_uniform = cfg['model']['path_uniform']
    model_zipfian = cfg['model']['path_zipfian']
    arch_name = cfg['model']['architecture']
    val_size = cfg.get('val_size', 0.05)

    # Everything happens here (side effects!)
    # - Creates 4 dataloaders
    # - Loads 2 models
    # - No control over what gets loaded
    analyzer = Embedding_analysis(
        path2data=path2data,
        data_name=data_name,
        model_uniform=model_uniform,
        model_zipfian=model_zipfian,
        arch_name=arch_name,
        val_size=val_size,
    )

    return analyzer


def _extract_embeddings_OLD(analyzer) -> Dict[str, np.ndarray]:
    """
    OLD: How analyze.py extracts embeddings

    From src/main_scripts/analyze.py
    """
    # Hidden in private method
    output_dict = analyzer._get_encodings()

    # Access with string keys (error-prone)
    return output_dict


# ============================================================================
# NEW VERSION (using 3-class system)
# ============================================================================

def _prepare_model_context_NEW(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    NEW: How analyze.py should create the 3-class system

    Benefits:
    - Explicit about what gets loaded
    - Lazy loading (only load what you need)
    - Returns structured objects instead of monolithic analyzer
    """
    from pipeline_refactored.core import (
        DatasetManager,
        ModelManager,
        EmbeddingExtractor,
    )

    # Extract config
    path2data = cfg['data']['path']
    data_name = cfg['data']['name']
    model_uniform = cfg['model']['path_uniform']
    model_zipfian = cfg['model']['path_zipfian']
    val_size = cfg.get('val_size', 0.05)

    # Create managers (no side effects, no loading yet)
    dataset_mgr = DatasetManager(
        path2data,
        data_name,
        default_val_size=val_size,
    )

    model_mgr = ModelManager()

    # Load models (explicit, lazy)
    model_mgr.load_model(model_uniform, label='uniform')
    model_mgr.load_model(model_zipfian, label='zipfian')

    # Create extractor
    extractor = EmbeddingExtractor(model_mgr)

    # Return structured context
    return {
        'dataset_mgr': dataset_mgr,
        'model_mgr': model_mgr,
        'extractor': extractor,
    }


def _extract_embeddings_NEW(context: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    NEW: How analyze.py should extract embeddings

    Benefits:
    - Explicit about what dataloader to use
    - Type-safe (no string key errors)
    - Flexible (can extract from different splits, layers, etc.)
    """
    from pipeline_refactored.core import AnalysisContext

    # Get components
    dataset_mgr = context['dataset_mgr']
    extractor = context['extractor']

    # Get dataloader (lazy, only creates what you need)
    val_loader = dataset_mgr.get_dataloader('uniform', split='val')

    # Extract aligned embeddings (same inputs → both models)
    Z_uniform, Z_zipfian = extractor.extract_aligned_pair(
        'uniform', 'zipfian', val_loader
    )

    # Extract features
    features = dataset_mgr.get_features('uniform', split='val')

    # Create AnalysisContext (structured, type-safe)
    analysis_ctx = AnalysisContext(
        embeddings={
            'uniform': Z_uniform,
            'zipfian': Z_zipfian,
        },
        features={
            'labels': features['labels'],
            'cum_area': features['cum_area'],
            'convex_hull': features['convex_hull'],
        },
        architecture=context.get('architecture', 'unknown'),
        distribution='uniform',
    )

    return analysis_ctx


# ============================================================================
# COMPLETE EXAMPLE: OLD vs NEW
# ============================================================================

def analyze_one_model_OLD(cfg: Dict[str, Any]):
    """
    OLD: How analyze.py currently works (simplified)
    """
    print("=" * 60)
    print("OLD VERSION")
    print("=" * 60)

    # Step 1: Create analyzer (everything loaded at once)
    print("\n[1/3] Creating analyzer...")
    analyzer = _prepare_model_context_OLD(cfg)
    print(f"  ✓ Created: {type(analyzer).__name__}")
    print(f"  (Loaded 2 models + created 4 dataloaders)")

    # Step 2: Extract embeddings
    print("\n[2/3] Extracting embeddings...")
    output_dict = _extract_embeddings_OLD(analyzer)
    print(f"  ✓ Extracted:")
    print(f"    Z_uniform: {output_dict['Z_uniform'].shape}")
    print(f"    Z_zipfian: {output_dict['Z_zipfian'].shape}")

    # Step 3: Run analyses (using string keys)
    print("\n[3/3] Running analyses...")
    Z_uniform = output_dict['Z_uniform']
    Z_zipfian = output_dict['Z_zipfian']
    labels = output_dict['labels_uniform']
    print(f"  ✓ Ready for analysis")

    return output_dict


def analyze_one_model_NEW(cfg: Dict[str, Any]):
    """
    NEW: How analyze.py should work (simplified)
    """
    print("\n" + "=" * 60)
    print("NEW VERSION")
    print("=" * 60)

    # Step 1: Create context (no loading yet)
    print("\n[1/3] Creating context...")
    context = _prepare_model_context_NEW(cfg)
    print(f"  ✓ Created DatasetManager")
    print(f"  ✓ Created ModelManager (loaded 2 models)")
    print(f"  ✓ Created EmbeddingExtractor")

    # Step 2: Extract embeddings (lazy, only what we need)
    print("\n[2/3] Extracting embeddings...")
    analysis_ctx = _extract_embeddings_NEW(context)
    print(f"  ✓ Extracted:")
    print(f"    uniform: {analysis_ctx.get_embedding('uniform').shape}")
    print(f"    zipfian: {analysis_ctx.get_embedding('zipfian').shape}")
    print(f"  ✓ Features:")
    print(f"    labels: {analysis_ctx.get_feature('labels').shape}")
    print(f"    cum_area: {analysis_ctx.get_feature('cum_area').shape}")

    # Step 3: Run analyses (type-safe, clear API)
    print("\n[3/3] Running analyses...")
    Z_uniform = analysis_ctx.get_embedding('uniform')
    Z_zipfian = analysis_ctx.get_embedding('zipfian')
    labels = analysis_ctx.get_feature('labels')
    print(f"  ✓ Ready for analysis")

    return analysis_ctx


# ============================================================================
# MIGRATION STRATEGY
# ============================================================================

def show_migration_strategy():
    """
    Step-by-step migration strategy for analyze.py
    """
    print("\n" + "=" * 60)
    print("MIGRATION STRATEGY")
    print("=" * 60)

    strategy = """
    PHASE 1: Add new system alongside old (NO breaking changes)
    ────────────────────────────────────────────────────────────
    1. Keep old Embedding_analysis imports
    2. Add new imports:
       from pipeline_refactored.core import (
           DatasetManager, ModelManager, EmbeddingExtractor, AnalysisContext
       )
    3. Add new _prepare_model_context_v2() function
    4. Add flag in config: use_new_system: true/false
    5. Test both systems side-by-side


    PHASE 2: Update analysis functions to use AnalysisContext
    ────────────────────────────────────────────────────────────
    1. Update _run_linear_probes() to accept AnalysisContext
    2. Update _run_layerwise_analysis() to accept AnalysisContext
    3. Update _run_cka_analysis() to accept AnalysisContext
    4. Update _run_behavioral_suite() to accept AnalysisContext
    5. Keep backward compatibility with old output_dict


    PHASE 3: Switch default to new system
    ────────────────────────────────────────────────────────────
    1. Set default: use_new_system: true
    2. Mark Embedding_analysis as deprecated
    3. Update documentation


    PHASE 4: Remove old system (after validation)
    ────────────────────────────────────────────────────────────
    1. Remove Embedding_analysis usage
    2. Remove backward compatibility helpers
    3. Clean up imports


    CONCRETE CHANGES TO analyze.py:
    ────────────────────────────────────────────────────────────

    BEFORE (analyze.py line ~1800):
    ───────────────────────────────
    from src.analyses.embedding_analysis import Embedding_analysis

    def _prepare_model_context(cfg):
        analyzer = Embedding_analysis(...)
        return analyzer

    def analyze_one_model(cfg):
        analyzer = _prepare_model_context(cfg)
        output_dict = analyzer._get_encodings()
        Z_uniform = output_dict['Z_uniform']
        Z_zipfian = output_dict['Z_zipfian']
        labels = output_dict['labels_uniform']
        # ... run analyses ...


    AFTER (using new system):
    ──────────────────────────
    from pipeline_refactored.core import (
        DatasetManager, ModelManager, EmbeddingExtractor, AnalysisContext
    )

    def _prepare_model_context(cfg):
        # Create managers
        dataset_mgr = DatasetManager(cfg['data']['path'], cfg['data']['name'])
        model_mgr = ModelManager()
        model_mgr.load_model(cfg['model']['path_uniform'], 'uniform')
        model_mgr.load_model(cfg['model']['path_zipfian'], 'zipfian')
        extractor = EmbeddingExtractor(model_mgr)

        return {
            'dataset_mgr': dataset_mgr,
            'model_mgr': model_mgr,
            'extractor': extractor,
        }

    def analyze_one_model(cfg):
        context = _prepare_model_context(cfg)

        # Extract embeddings
        val_loader = context['dataset_mgr'].get_dataloader('uniform', split='val')
        Z_u, Z_z = context['extractor'].extract_aligned_pair('uniform', 'zipfian', val_loader)
        features = context['dataset_mgr'].get_features('uniform', split='val')

        # Create AnalysisContext
        analysis_ctx = AnalysisContext(
            embeddings={'uniform': Z_u, 'zipfian': Z_z},
            features=features,
            architecture=cfg['model']['architecture'],
            distribution='uniform',
        )

        # Run analyses (cleaner API)
        Z_uniform = analysis_ctx.get_embedding('uniform')
        Z_zipfian = analysis_ctx.get_embedding('zipfian')
        labels = analysis_ctx.get_feature('labels')
        # ... run analyses ...


    KEY BENEFITS:
    ────────────────────────────────────────────────────────────
    ✓ Lazy loading (only load what you need)
    ✓ Type-safe (no string key errors)
    ✓ Testable (each component independent)
    ✓ Flexible (can extract from different splits, layers)
    ✓ Cleaner code (separation of concerns)
    ✓ Works for training AND analysis
    """

    print(strategy)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("UPDATING analyze.py: OLD → NEW")
    print("=" * 60)

    # Show migration strategy
    show_migration_strategy()

    # Show side-by-side example (with dummy data)
    print("\n" + "=" * 60)
    print("SIDE-BY-SIDE COMPARISON (with dummy config)")
    print("=" * 60)

    # Dummy config
    dummy_cfg = {
        'data': {
            'path': 'data/behavioral_datasets',
            'name': 'dummy_dataset.npz',
        },
        'model': {
            'path_uniform': 'models/uniform_model.pkl',
            'path_zipfian': 'models/zipfian_model.pkl',
            'architecture': 'iDBN_1500_500',
        },
        'val_size': 0.05,
    }

    print("\nNOTE: This uses dummy paths and won't actually run.")
    print("      See the code above for concrete examples.\n")

    # Uncomment to see the comparison (requires real models):
    # analyze_one_model_OLD(dummy_cfg)
    # analyze_one_model_NEW(dummy_cfg)
