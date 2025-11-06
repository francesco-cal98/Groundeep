"""PCA diagnostics stage (LEGACY - needs refactoring)."""

from pathlib import Path
from typing import Dict, Any


class PCADiagnosticsStage:
    """Stage for PCA geometry diagnostics.

    NOTE: This is a legacy stub. The actual PCA diagnostics functionality
    is already implemented in DimensionalityStage (pca_geometry and pca_report).

    This stage is disabled by default and should not be used. If you need
    PCA diagnostics, use DimensionalityStage instead:

    pca_geometry:
      enabled: true
      n_components: 5

    pca_report:
      enabled: true
    """

    name = "pca_geometry"

    def is_enabled(self, settings: Dict[str, Any]) -> bool:
        return settings.get('enabled', False)

    def run(self, ctx: Any, settings: Dict[str, Any], output_dir: Path) -> None:
        """Run PCA diagnostics.

        This is a legacy stub and is not implemented in the modular pipeline.
        Use DimensionalityStage instead.
        """
        print("⚠️  PCADiagnosticsStage is a legacy stub and is not implemented.")
        print("   Use DimensionalityStage with pca_geometry and pca_report instead.")
        return
