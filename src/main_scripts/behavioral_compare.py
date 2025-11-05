"""
Compare behavioural readouts between uniform and Zipfian DBNs for the same architecture.

Example
-------
python src/main_scripts/behavioral_compare.py \
    --arch-name iDBN_1500_500 \
    --model-uniform networks/uniform/dataset_10_10/uniform_i_/dbn_trained_uniform_1500_500.pkl \
    --model-zipfian networks/zipfian/dataset_10_10/zipfian_i_/dbn_trained_zipfian_1500_500.pkl \
    --train-pickle behavioral_datasets/binary_de_wind_train.pkl \
    --test-pickle behavioral_datasets/binary_de_wind_test.pkl \
    --mat-file circle_dataset_100x100/NumStim_7to28_100x100_TE.mat \
    --outdir results/behavioral_comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analyses.behavioral_analysis import (
    load_behavioral_inputs,
    run_behavioral_analysis,
)
from src.analyses.embedding_analysis import Embedding_analysis
from src.analyses.task_comparison import run_task_comparison
from src.main_scripts.analyze import _get_model_device


def _load_dbn(path: Path):
    """Reuse the DBN loader from Embedding_analysis without materialising the class."""
    return Embedding_analysis._load_model(str(path))


def _prepare_inputs(
    train_pickle: Path,
    test_pickle: Path,
    mat_file: Path,
    device: torch.device,
):
    """Load behavioural tensors for a specific device."""
    return load_behavioral_inputs(train_pickle, test_pickle, mat_file, device)


def _ensure_parent(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _set_device_for_ccnl(device: torch.device) -> None:
    import CCNL_readout_DBN

    CCNL_readout_DBN.DEVICE = device


def _summarise_results(
    standard: Dict[str, float],
    comparison: Dict[str, float],
    regime: str,
) -> Dict[str, float]:
    summary = {
        "regime": regime,
        "std_accuracy_train": standard.get("accuracy_train"),
        "std_accuracy_test": standard.get("accuracy_test"),
        "std_beta_number": standard.get("beta_number"),
        "std_beta_size": standard.get("beta_size"),
        "std_beta_spacing": standard.get("beta_spacing"),
        "std_weber_fraction": standard.get("weber_fraction"),
    }
    summary.update(
        {
            "cmp_accuracy_train": comparison.get("accuracy_train"),
            "cmp_accuracy_test": comparison.get("accuracy_test"),
            "cmp_beta_number": comparison.get("beta_number"),
            "cmp_beta_size": comparison.get("beta_size"),
            "cmp_beta_spacing": comparison.get("beta_spacing"),
            "cmp_weber_fraction": comparison.get("weber_fraction"),
        }
    )
    return summary


def compute_deltas(rows: List[Dict[str, float]]) -> Dict[str, float]:
    frame = pd.DataFrame(rows).set_index("regime")
    if not {"uniform", "zipfian"}.issubset(frame.index):
        return {}
    diff = frame.loc["zipfian"] - frame.loc["uniform"]
    diff.name = "zipfian_minus_uniform"
    return {"regime": diff.name, **diff.to_dict()}


def run_pipeline(args) -> Path:
    outdir = Path(args.outdir).resolve()
    _ensure_parent(outdir)

    train_pickle = Path(args.train_pickle).resolve()
    test_pickle = Path(args.test_pickle).resolve()
    mat_file = Path(args.mat_file).resolve()

    regimes = {
        "uniform": Path(args.model_uniform).resolve(),
        "zipfian": Path(args.model_zipfian).resolve(),
    }

    guess_rate = float(args.guess_rate)
    summaries: List[Dict[str, float]] = []

    for regime, model_path in regimes.items():
        model = _load_dbn(model_path)
        device = _get_model_device(model)
        _set_device_for_ccnl(device)

        inputs = _prepare_inputs(train_pickle, test_pickle, mat_file, device)

        regime_dir = outdir / regime
        standard_dir = regime_dir / "standard"
        comparison_dir = regime_dir / "comparison_standard"
        _ensure_parent(standard_dir)
        _ensure_parent(comparison_dir)

        label = f"{args.arch_name}_{regime}"
        standard = run_behavioral_analysis(
            model,
            inputs,
            standard_dir,
            label,
            guess_rate=guess_rate,
        )

        comparison = run_task_comparison(
            model,
            inputs,
            comparison_dir,
            label,
            guess_rate=guess_rate,
        )

        summaries.append(_summarise_results(standard, comparison, regime))
        del inputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    delta_row = compute_deltas(summaries)
    if delta_row:
        summaries.append(delta_row)

    df = pd.DataFrame(summaries)
    csv_path = outdir / f"behavioral_summary_{args.arch_name}.csv"
    df.to_csv(csv_path, index=False)

    json_path = outdir / f"behavioral_summary_{args.arch_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print(df.to_string(index=False))
    print(f"\nSaved summary to {csv_path}")
    return csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Uniform vs Zipfian behavioural comparison.")
    parser.add_argument("--arch-name", required=True, help="Architecture label used in outputs.")
    parser.add_argument("--model-uniform", required=True, help="Path to the uniform-trained DBN (.pkl).")
    parser.add_argument("--model-zipfian", required=True, help="Path to the Zipfian-trained DBN (.pkl).")
    parser.add_argument("--train-pickle", required=True, help="Behavioural training pickle (pairs).")
    parser.add_argument("--test-pickle", required=True, help="Behavioural test pickle (pairs).")
    parser.add_argument("--mat-file", required=True, help="MAT archive supplying feature metadata (e.g., NumStim_7to28_100x100_TE.mat).")
    parser.add_argument("--guess-rate", type=float, default=0.01, help="Guess rate used in probit fits (default: 0.01).")
    parser.add_argument("--outdir", default="results/behavioral_comparison", help="Directory where reports are written.")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_pipeline(args)


if __name__ == "__main__":
    main()
