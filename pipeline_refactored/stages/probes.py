"""Linear probes analysis stage."""

from pathlib import Path
from typing import Dict, Any
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.probe_utils import log_linear_probe
# Import from clean pipeline_refactored (zero dependency on analyze.py)
from pipeline_refactored.core.probe_model import ProbeReadyModel


class LinearProbesStage:
    """Stage for running linear probes (labels, cumArea, CH)."""

    name = "probing"

    def is_enabled(self, settings: Dict[str, Any]) -> bool:
        return settings.get('enabled', False)

    def run(self, ctx: Any, settings: Dict[str, Any], output_dir: Path) -> None:
        """Execute linear probes on requested layers."""
        # Extract parameters
        probe_settings = settings.get('settings', {})
        features = settings.get('features', {})
        arch_name = settings.get('arch_name', 'unknown')
        dist_name = settings.get('dist_name', 'unknown')
        seed = settings.get('seed', 42)
        wandb_run = settings.get('wandb_run', None)

        model_for_probe = (
            ctx.get_model("uniform")
            if dist_name == "uniform"
            else ctx.get_model("zipfian")
        )

        layers = probe_settings.get("layers", ["top"])

        for layer in layers:
            upto = None if str(layer).lower() == "top" else int(layer)
            layer_tag = "top" if upto is None else f"layer{upto}"
            probe_dir = output_dir / "probes" / layer_tag

            prm = ProbeReadyModel(
                raw_model=model_for_probe,
                val_loader=ctx.uniform_val_loader,
                features_dict=features,
                out_dir=probe_dir,
                wandb_run=wandb_run,
            )

            summary_rows = log_linear_probe(
                model=prm,
                epoch=0,
                n_bins=int(probe_settings.get("n_bins", 5)),
                test_size=float(probe_settings.get("test_size", 0.2)),
                steps=int(probe_settings.get("steps", 1000)),
                lr=float(probe_settings.get("lr", 1e-2)),
                rng_seed=int(seed),
                patience=int(probe_settings.get("patience", 20)),
                min_delta=0.0,
                save_csv=True,
                upto_layer=upto,
                layer_tag=layer_tag,
            )

            if not summary_rows:
                continue

            # Save summary
            probe_dir.mkdir(parents=True, exist_ok=True)
            records = [
                {"metric": str(row.get("metric")), "accuracy": float(row.get("accuracy", 0.0))}
                for row in summary_rows
            ]
            df_probe = pd.DataFrame(records)
            df_probe.to_csv(probe_dir / "probe_summary.csv", index=False)

            # Plot summary
            fig, ax = plt.subplots(figsize=(max(6, len(records) * 1.2), 4))
            sns.barplot(
                data=df_probe, x="metric", y="accuracy",
                color="steelblue", errorbar=None, ax=ax,
                order=df_probe["metric"].tolist(),
            )
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Accuracy")
            ax.set_xlabel("")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_title(f"Linear probe accuracy — {arch_name} ({dist_name}) [{layer_tag}]")

            labels_fmt = [f"{val:.2f}" for val in df_probe["accuracy"]]
            bar_containers = getattr(ax, "containers", None)
            if bar_containers:
                try:
                    ax.bar_label(bar_containers[0], labels=labels_fmt, padding=3, fontsize=10)
                except Exception:
                    bar_containers = None

            if not bar_containers:
                for patch, label in zip(ax.patches, labels_fmt):
                    ax.annotate(
                        label,
                        (patch.get_x() + patch.get_width() / 2,
                         min(patch.get_height() + 0.02, ax.get_ylim()[1] - 0.01)),
                        ha="center", va="bottom", fontsize=10,
                    )

            fig.tight_layout()
            fig.savefig(probe_dir / "probe_summary.png", dpi=300)
            plt.close(fig)

            # Confusion matrices
            for row in summary_rows:
                conf_df = row.get("confusion")
                if not isinstance(conf_df, pd.DataFrame):
                    continue

                cm_counts = conf_df.values.astype(float)
                row_sums = cm_counts.sum(axis=1, keepdims=True)
                with np.errstate(invalid="ignore", divide="ignore"):
                    cm_norm = np.divide(cm_counts, row_sums, where=row_sums > 0)

                cm_plot = pd.DataFrame(cm_norm, index=conf_df.index, columns=conf_df.columns)
                fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                sns.heatmap(
                    cm_plot, annot=True, fmt=".2f", cmap="viridis",
                    cbar=True, vmin=0.0, vmax=1.0, ax=ax_cm,
                )
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("True")
                ax_cm.set_title(f"Probe confusion ({arch_name} – {dist_name} – {row['metric']})")
                ax_cm.set_xticklabels(ax_cm.get_xticklabels(), rotation=45, ha="right")
                ax_cm.set_yticklabels(ax_cm.get_yticklabels(), rotation=0)
                fig_cm.tight_layout()

                safe_metric = str(row["metric"]).replace("/", "_").replace(" ", "_")
                fig_cm.savefig(probe_dir / f"{safe_metric}_confusion.png", dpi=300)
                plt.close(fig_cm)
