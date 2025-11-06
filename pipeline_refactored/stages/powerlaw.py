"""Power-law scaling stage."""

from pathlib import Path
from typing import Dict, Any
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analyses.powerfit_pairs import (
    build_pairwise_xy,
    fit_power_loglog_pairs,
    plot_pairs_fit,
    plot_pairs_fit_loglog,
    save_pairs_fit,
)


class PowerLawStage:
    """Stage for power-law scaling analysis."""

    name = "powerlaw"

    def is_enabled(self, settings: Dict[str, Any]) -> bool:
        return settings.get('enabled', True)

    def run(self, ctx: Any, settings: Dict[str, Any], output_dir: Path) -> None:
        """Fit power-law relationship between centroid distances and numerosity."""
        pf_dir = output_dir / "powerfit_pairs"
        pf_dir.mkdir(parents=True, exist_ok=True)

        Z = ctx.bundle.embeddings
        labels = ctx.bundle.labels
        arch_name = ctx.spec.arch_name
        dist_name = ctx.spec.distribution

        # Build pairwise distances
        x_pairs, y_pairs, pairs_df = build_pairwise_xy(Z, labels, metric="euclidean")
        pairs_df.to_csv(pf_dir / f"pairs_table_{arch_name}_{dist_name}.csv", index=False)

        if x_pairs.size == 0:
            print(f"[PowerFit] {arch_name}/{dist_name}: no valid centroid pairs")
            return

        # Fit power-law
        fit = fit_power_loglog_pairs(x_pairs, y_pairs)
        save_pairs_fit(fit, pf_dir / f"params_{arch_name}_{dist_name}.csv")

        # Plot
        plot_pairs_fit(
            x_pairs, y_pairs, fit,
            pf_dir / f"fit_linear_{arch_name}_{dist_name}.png",
            f"{arch_name} ({dist_name})",
        )
        plot_pairs_fit_loglog(
            x_pairs, y_pairs, fit,
            pf_dir / f"fit_loglog_{arch_name}_{dist_name}.png",
            f"{arch_name} ({dist_name})",
        )

        print(f"[PowerFit] {arch_name}/{dist_name}: b={fit['b']:.3f}, R²={fit['r2']:.3f}")

        # WandB logging (optional)
        if ctx.wandb_run is not None:
            try:
                import wandb
                ctx.wandb_run.log({
                    "powerfit_pairs/fit_linear": wandb.Image(str(pf_dir / f"fit_linear_{arch_name}_{dist_name}.png")),
                    "powerfit_pairs/fit_loglog": wandb.Image(str(pf_dir / f"fit_loglog_{arch_name}_{dist_name}.png")),
                    "powerfit_pairs/b": fit["b"],
                    "powerfit_pairs/r2": fit["r2"],
                })
            except Exception:
                pass
