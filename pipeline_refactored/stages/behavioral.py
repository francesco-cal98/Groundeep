"""Behavioral tasks stage."""

from pathlib import Path
from typing import Dict, Any, Optional
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analyses.behavioral_analysis import (
    load_behavioral_inputs,
    run_behavioral_analysis,
)
from src.analyses.task_comparison import run_task_comparison
from src.analyses.task_fixed_reference import (
    load_fixed_reference_inputs,
    run_task_fixed_reference,
)
from src.analyses.task_numerosity_estimation import (
    load_estimation_dataset,
    run_task_numerosity_estimation,
)


class BehavioralStage:
    """Stage for behavioral task suite (comparison, fixed reference, estimation)."""

    name = "behavioral"

    def is_enabled(self, settings: Dict[str, Any]) -> bool:
        return settings.get('enabled', False)

    def _get_model_device(self, model_obj) -> torch.device:
        """Infer device from model."""
        try:
            import torch.nn as nn
            if isinstance(model_obj, nn.Module):
                try:
                    return next(model_obj.parameters()).device
                except StopIteration:
                    pass
        except Exception:
            pass

        try:
            layers = getattr(model_obj, "layers", [])
            if layers:
                first_rbm = layers[0]
                for attr_name in ("W", "c", "b", "weights"):
                    attr_val = getattr(first_rbm, attr_name, None)
                    if isinstance(attr_val, torch.Tensor):
                        return attr_val.device
        except Exception:
            pass
        return torch.device("cpu")

    def run(self, ctx: Any, settings: Dict[str, Any], output_dir: Path) -> None:
        """Execute behavioral tasks."""
        tasks_cfg = settings.get("tasks", {})
        model_for_behavior = (
            ctx.get_model("uniform")
            if ctx.spec.distribution == "uniform"
            else ctx.get_model("zipfian")
        )
        behaviors_dir = output_dir / "behavioral"
        behavior_label = f"{ctx.spec.arch_name}_{ctx.spec.distribution}"

        # Load base behavioral inputs
        train_path = settings.get("train_pickle")
        test_path = settings.get("test_pickle")
        mat_path = settings.get("mat_file")

        if not (train_path and test_path and mat_path):
            print("[Behavioral] Missing paths: train_pickle/test_pickle/mat_file")
            return

        try:
            device_behavior = self._get_model_device(model_for_behavior)
            behavioral_inputs = load_behavioral_inputs(
                Path(train_path), Path(test_path), Path(mat_path), device_behavior
            )
        except Exception as exc:
            print(f"[Behavioral] Failed to load datasets ({exc})")
            return

        # Run base behavioral analysis
        guess_rate = float(settings.get("guess_rate", 0.01))
        results_beh = run_behavioral_analysis(
            model_for_behavior, behavioral_inputs,
            behaviors_dir, behavior_label, guess_rate=guess_rate
        )

        if ctx.wandb_run:
            try:
                import wandb
                ctx.wandb_run.log({
                    "behavioral/accuracy_test": results_beh.get("accuracy_test"),
                    "behavioral/accuracy_train": results_beh.get("accuracy_train"),
                    "behavioral/beta_number": results_beh.get("beta_number"),
                    "behavioral/beta_size": results_beh.get("beta_size"),
                    "behavioral/beta_spacing": results_beh.get("beta_spacing"),
                    "behavioral/weber_fraction": results_beh.get("weber_fraction"),
                })
            except Exception:
                pass

        # Comparison task
        comparison_cfg = tasks_cfg.get("comparison", {})
        if comparison_cfg.get("enabled", False):
            self._run_comparison_task(
                ctx, model_for_behavior, behavioral_inputs,
                comparison_cfg, settings, behaviors_dir, behavior_label, device_behavior
            )

        # Fixed reference task
        fixed_cfg = tasks_cfg.get("fixed_reference", {})
        if fixed_cfg.get("enabled", False):
            self._run_fixed_reference_task(
                ctx, model_for_behavior, fixed_cfg, settings,
                behaviors_dir, behavior_label, device_behavior
            )

        # Estimation task
        estimation_cfg = tasks_cfg.get("estimation", {})
        if estimation_cfg.get("enabled", False):
            self._run_estimation_task(
                ctx, estimation_cfg, behaviors_dir, behavior_label
            )

    def _run_comparison_task(self, ctx, model, behavioral_inputs, comparison_cfg,
                            settings, behaviors_dir, behavior_label, device_behavior):
        """Run comparison task."""
        out_cmp = behaviors_dir / "comparison"
        guess_rate_cmp = float(comparison_cfg.get("guess_rate", settings.get("guess_rate", 0.01)))

        try:
            comparison_inputs = behavioral_inputs
            cmp_train = comparison_cfg.get("train_pickle")
            cmp_test = comparison_cfg.get("test_pickle")
            if cmp_train and cmp_test:
                cmp_mat = comparison_cfg.get("mat_file") or settings.get("mat_file")
                comparison_inputs = load_behavioral_inputs(
                    Path(cmp_train), Path(cmp_test),
                    Path(cmp_mat), device_behavior or self._get_model_device(model)
                )

            results_cmp = run_task_comparison(
                model, comparison_inputs, out_cmp, behavior_label, guess_rate=guess_rate_cmp
            )

            if ctx.wandb_run:
                ctx.wandb_run.log({
                    "behavioral/comparison/accuracy_test": results_cmp.get("accuracy_test"),
                    "behavioral/comparison/accuracy_train": results_cmp.get("accuracy_train"),
                    "behavioral/comparison/weber_fraction": results_cmp.get("weber_fraction"),
                    "behavioral/comparison/beta_number": results_cmp.get("beta_number"),
                })
        except Exception as exc:
            print(f"[Behavioral] Comparison task failed: {exc}")

    def _run_fixed_reference_task(self, ctx, model, fixed_cfg, settings,
                                  behaviors_dir, behavior_label, device_behavior):
        """Run fixed reference task."""
        refs = fixed_cfg.get("references", [])
        train_template = fixed_cfg.get("train_template")
        test_template = fixed_cfg.get("test_template")
        mat_path = fixed_cfg.get("mat_file") or settings.get("mat_file")
        guess_rate = float(fixed_cfg.get("guess_rate", settings.get("guess_rate", 0.01)))

        if not (train_template and test_template and mat_path):
            print("[Behavioral] Fixed reference: missing paths")
            return

        for ref in refs:
            try:
                train_p = Path(str(train_template).format(ref=ref))
                test_p = Path(str(test_template).format(ref=ref))
                fixed_inputs = load_fixed_reference_inputs(
                    train_p, test_p, Path(mat_path),
                    device_behavior or self._get_model_device(model)
                )

                out_dir = behaviors_dir / "fixed_reference" / f"ref{ref}"
                results_fr = run_task_fixed_reference(
                    model, fixed_inputs, out_dir,
                    f"{behavior_label}_ref{ref}", guess_rate=guess_rate
                )

                if ctx.wandb_run:
                    ctx.wandb_run.log({
                        f"behavioral/fixed_ref{ref}/accuracy": results_fr.get("accuracy"),
                        f"behavioral/fixed_ref{ref}/weber_fraction": results_fr.get("weber_fraction"),
                    })
            except Exception as exc:
                print(f"[Behavioral] Fixed reference ref={ref} failed: {exc}")

    def _run_estimation_task(self, ctx, estimation_cfg, behaviors_dir, behavior_label):
        """Run estimation task."""
        datasets_cfg = estimation_cfg.get("datasets", {})
        uniform_cfg = datasets_cfg.get("uniform", {})
        zipfian_cfg = datasets_cfg.get("zipfian", {})

        dist_name = ctx.spec.distribution
        target_cfg = uniform_cfg if dist_name == "uniform" else zipfian_cfg

        if not target_cfg:
            print(f"[Behavioral] Estimation: no config for {dist_name}")
            return

        try:
            train_pkl = target_cfg.get("train_pickle")
            test_pkl = target_cfg.get("test_pickle")
            if not (train_pkl and test_pkl):
                return

            model = (
                ctx.get_model("uniform")
                if dist_name == "uniform"
                else ctx.get_model("zipfian")
            )
            device = self._get_model_device(model)

            est_inputs = load_estimation_dataset(
                Path(train_pkl), Path(test_pkl), device
            )

            out_dir = behaviors_dir / "estimation" / dist_name
            classifiers = estimation_cfg.get("classifiers", ["SGD_regression"])
            label_mode = estimation_cfg.get("label_mode", "int")
            scale_targets = estimation_cfg.get("scale_targets", False)
            max_display = estimation_cfg.get("max_display_classes", 32)

            results_est = run_task_numerosity_estimation(
                model, est_inputs, out_dir,
                f"{behavior_label}_estimation",
                classifiers=classifiers,
                label_mode=label_mode,
                scale_targets=scale_targets,
                max_display_classes=max_display,
            )

            if ctx.wandb_run:
                for clf_name, metrics in results_est.items():
                    ctx.wandb_run.log({
                        f"behavioral/estimation/{clf_name}/r2": metrics.get("r2"),
                        f"behavioral/estimation/{clf_name}/mae": metrics.get("mae"),
                    })
        except Exception as exc:
            print(f"[Behavioral] Estimation task failed: {exc}")
