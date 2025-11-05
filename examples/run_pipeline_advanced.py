#!/usr/bin/env python3
"""
Advanced example: Customized pipeline with specific analyses.

Shows how to:
- Create custom analysis configurations
- Run specific analyses programmatically
- Access and process results
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from groundeep.pipeline import AnalysisPipeline
from groundeep.core.config_loader import PipelineConfig, AnalysisConfig
from groundeep.utils.io import save_results
import json


def main():
    config_path = PROJECT_ROOT / "src" / "configs" / "analysis.yaml"

    print("=" * 80)
    print("GROUNDEEP Analysis Pipeline - Advanced Example")
    print("=" * 80)

    # Load config
    pipeline = AnalysisPipeline.from_config(config_path)

    # Optionally modify analysis configs programmatically
    # For example, change probe parameters
    pipeline.config.analyses["probing"].params["n_bins"] = 7
    pipeline.config.analyses["probing"].params["max_steps"] = 2000

    print("\nRunning full analysis suite...")

    # Run all enabled analyses
    results = pipeline.run()

    # Process results
    print("\n" + "=" * 80)
    print("PROCESSING RESULTS")
    print("=" * 80)

    # Extract specific metrics
    for model_key, analyses in results.get("single_model", {}).items():
        print(f"\nModel: {model_key}")

        if "geometry" in analyses:
            geom_metrics = analyses["geometry"].get("metrics", {})
            for key, val in geom_metrics.items():
                if "monotonicity/spearman_rho" in key:
                    print(f"  Monotonicity ρ: {val:.4f}")

        if "scaling" in analyses:
            scaling_metrics = analyses["scaling"].get("metrics", {})
            for key, val in scaling_metrics.items():
                if "/beta" in key:
                    print(f"  Scaling exponent β: {val:.4f}")

        if "probes" in analyses:
            probe_metrics = analyses["probes"].get("metrics", {})
            label_acc = probe_metrics.get("top/labels/accuracy", None)
            if label_acc is not None:
                print(f"  Probe accuracy (labels): {label_acc:.4f}")

    # Save results in multiple formats
    output_dir = PROJECT_ROOT / "results" / "advanced_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    pipeline.save_results(results, output_dir=output_dir)

    # Custom extraction: Save only metrics as CSV-friendly format
    metrics_flat = {}
    for model_key, analyses in results.get("single_model", {}).items():
        for analysis_name, analysis_data in analyses.items():
            for metric_key, value in analysis_data.get("metrics", {}).items():
                full_key = f"{model_key}/{analysis_name}/{metric_key}"
                metrics_flat[full_key] = value

    metrics_path = output_dir / "metrics_flat.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_flat, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"Flat metrics saved to: {metrics_path}")

    # Generate report
    report = pipeline.generate_report(results)
    report_path = output_dir / "report.txt"
    with open(report_path, 'w') as f:
        f.write(report)

    pipeline.close()


if __name__ == "__main__":
    main()
