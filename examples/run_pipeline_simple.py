#!/usr/bin/env python3
"""
Simple example: Run the GROUNDEEP analysis pipeline.

This script demonstrates basic usage of the refactored pipeline.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from groundeep.pipeline import AnalysisPipeline


def main():
    # Path to your config file
    config_path = PROJECT_ROOT / "src" / "configs" / "analysis.yaml"

    print("=" * 80)
    print("GROUNDEEP Analysis Pipeline - Simple Example")
    print("=" * 80)
    print(f"Config: {config_path}")
    print()

    # Create pipeline from config
    pipeline = AnalysisPipeline.from_config(config_path)

    # Run selected analyses
    # Options: "probes", "geometry", "scaling", "cka"
    results = pipeline.run(
        analyses=["geometry", "scaling"],  # Run only geometry and scaling
        # models=["uniform"]  # Optionally filter by model distribution
    )

    # Save results
    output_dir = PROJECT_ROOT / "results" / "pipeline_output"
    pipeline.save_results(results, output_dir=output_dir)

    # Generate and print report
    report = pipeline.generate_report(results)

    # Save report
    report_path = output_dir / "report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\nResults saved to: {output_dir}")
    print(f"Report saved to: {report_path}")

    # Cleanup
    pipeline.close()


if __name__ == "__main__":
    main()
