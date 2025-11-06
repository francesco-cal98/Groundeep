"""Base stage protocol for analysis pipeline."""

from typing import Protocol, Dict, Any, Optional
from pathlib import Path


class BaseStage(Protocol):
    """Protocol for analysis stages.

    Each stage:
    - Has a name
    - Takes context (ModelAnalysisContext) and settings
    - Runs analysis
    - Produces outputs (plots, metrics, etc.)
    """

    name: str

    def run(self, ctx: Any, settings: Dict[str, Any], output_dir: Path) -> None:
        """Run the analysis stage.

        Args:
            ctx: ModelAnalysisContext with embeddings, models, etc.
            settings: Config dict for this stage
            output_dir: Where to save outputs
        """
        ...

    def is_enabled(self, settings: Dict[str, Any]) -> bool:
        """Check if this stage should run based on config."""
        ...


class StageRegistry:
    """Registry for managing analysis stages."""

    def __init__(self):
        self._stages: Dict[str, BaseStage] = {}

    def register(self, stage: BaseStage) -> None:
        """Register a stage."""
        self._stages[stage.name] = stage

    def get(self, name: str) -> Optional[BaseStage]:
        """Get stage by name."""
        return self._stages.get(name)

    def run_all(self, ctx: Any, settings: Any, output_dir: Path) -> None:
        """Run all enabled stages."""
        for stage in self._stages.values():
            stage_settings = getattr(settings, stage.name, {})
            if stage.is_enabled(stage_settings):
                print(f"\n{'='*60}")
                print(f"Running: {stage.name}")
                print(f"{'='*60}")
                stage.run(ctx, stage_settings, output_dir)
