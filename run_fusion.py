"""
run_fusion.py — Unified runner for NF-MLP and GTF fusion baselines
===================================================================
Iterates over all run_N/First_level_data/ folders for a given ML
algorithm, trains the chosen fusion model on each run, and writes a
CSV with per-run metrics plus summary rows (Mean, Std).

Usage
-----
  # NF-MLP on DecisionTree predictions
  python run_fusion.py --algorithm DecisionTree --fusion nfmlp

  # GTF on RandomForest predictions, custom output path
  python run_fusion.py --algorithm RandomForest --fusion gtf \\
      --output results/rf_gtf.csv

  # Override data root and hyperparameters
  python run_fusion.py --algorithm SVC --fusion nfmlp \\
      --data_dir prediction_data --lr 5e-4 --max_epochs 500

  # Pass BN baseline stats to trigger Welch t-test
  python run_fusion.py --algorithm XGB --fusion gtf \\
      --bn_acc 87.61 --bn_acc_std 4.69 \\
      --bn_sen 85.12 --bn_sen_std 7.96 \\
      --bn_spe 90.63 --bn_spe_std 14.18 \\
      --bn_f1  88.28 --bn_f1_std  4.09
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# Constants
# ============================================================

VALID_ALGORITHMS = ("DecisionTree", "RandomForest", "SVC", "XGB")
VALID_FUSIONS    = ("nfmlp", "gtf")

FIRST_LEVEL_SUBDIR = "First_level_data"


# ============================================================
# Folder discovery
# ============================================================

def discover_run_folders(data_dir: Path, algorithm: str) -> list[Path]:
    """
    Return the First_level_data sub-paths for all run_N folders under
    data_dir/algorithm/, sorted numerically by run index.
    """
    algo_dir = data_dir / algorithm
    if not algo_dir.exists():
        raise FileNotFoundError(f"Algorithm directory not found: {algo_dir}")

    pattern = re.compile(r"^run_(\d+)$")
    runs: list[tuple[int, Path]] = []

    for entry in algo_dir.iterdir():
        match = pattern.match(entry.name)
        if match and entry.is_dir():
            first_level = entry / FIRST_LEVEL_SUBDIR
            if first_level.exists():
                runs.append((int(match.group(1)), first_level))
            else:
                print(f"  [WARN] No {FIRST_LEVEL_SUBDIR}/ in {entry.name} — skipped")

    if not runs:
        raise FileNotFoundError(
            f"No valid run_N/{FIRST_LEVEL_SUBDIR} directories found in {algo_dir}"
        )

    runs.sort(key=lambda t: t[0])
    return [path for _, path in runs]


# ============================================================
# CSV output
# ============================================================

def write_results_csv(rows: list[dict], output_path: Path) -> None:
    """
    Write per-run metrics to CSV.

    Columns : Run | Accuracy | Sensitivity | Specificity | F1-Score
    Appended: Mean and Std summary rows.
    """
    df = pd.DataFrame(rows, columns=["Run", "Accuracy", "Sensitivity",
                                     "Specificity", "F1-Score"])

    numeric_cols = ["Accuracy", "Sensitivity", "Specificity", "F1-Score"]
    mean_vals = df[numeric_cols].mean()
    std_vals  = df[numeric_cols].std(ddof=1)

    summary = pd.DataFrame([
        {"Run": "Mean", **mean_vals.to_dict()},
        {"Run": "Std",  **std_vals.to_dict()},
    ])

    output = pd.concat([df, summary], ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, float_format="%.5f")
    print(f"\nResults written to: {output_path}")


# ============================================================
# Import fusion module
# ============================================================

def load_fusion_module(fusion: str):
    """Dynamically import the requested fusion module from dl_fusion/."""
    dl_fusion_dir = Path(__file__).parent / "dl_fusion"
    if str(dl_fusion_dir) not in sys.path:
        sys.path.insert(0, str(dl_fusion_dir))

    if fusion == "nfmlp":
        import nf_mlp_fusion as mod
    else:
        import gtf_fusion as mod

    return mod


# ============================================================
# Run-loop dispatchers
# ============================================================

def run_nfmlp(mod, folders: list[Path], args) -> list[dict]:
    rows: list[dict] = []

    for i, folder in enumerate(folders):
        run_name = folder.parent.name  # e.g. "run_1"
        try:
            X_tr, y_tr, X_te, y_te = mod.load_fold(
                folder,
                args.train_pred, args.train_proba,
                args.test_pred,  args.test_proba,
            )
        except FileNotFoundError as e:
            print(f"  [SKIP] {run_name}: {e}")
            continue

        metrics = mod.run_fold(
            X_tr, y_tr, X_te, y_te,
            lr         = args.lr,
            batch_size = args.batch_size,
            max_epochs = args.max_epochs,
            patience   = args.patience,
            val_frac   = args.val_frac,
        )

        rows.append({
            "Run":         run_name,
            "Accuracy":    metrics["acc"],
            "Sensitivity": metrics["sen"],
            "Specificity": metrics["spe"],
            "F1-Score":    metrics["f1"],
        })

        print(f"  [{i+1:02d}/{len(folders)}] {run_name:<10}  "
              f"Acc={metrics['acc']:.2f}%  Sen={metrics['sen']:.2f}%  "
              f"Spe={metrics['spe']:.2f}%  F1={metrics['f1']:.2f}%")

    return rows


def run_gtf(mod, folders: list[Path], args) -> list[dict]:
    rows: list[dict] = []

    for i, folder in enumerate(folders):
        run_name = folder.parent.name
        try:
            pred_tr, proba_tr, y_tr, pred_te, proba_te, y_te = mod.load_fold(
                folder,
                args.train_pred, args.train_proba,
                args.test_pred,  args.test_proba,
            )
        except FileNotFoundError as e:
            print(f"  [SKIP] {run_name}: {e}")
            continue

        metrics = mod.run_fold(
            pred_tr, proba_tr, y_tr,
            pred_te, proba_te, y_te,
            bn_scoring     = args.bn_scoring,
            bn_max_parents = args.bn_max_parents,
            lr             = args.lr,
            batch_size     = args.batch_size,
            max_epochs     = args.max_epochs,
            patience       = args.patience,
            val_frac       = args.val_frac,
        )

        rows.append({
            "Run":         run_name,
            "Accuracy":    metrics["acc"],
            "Sensitivity": metrics["sen"],
            "Specificity": metrics["spe"],
            "F1-Score":    metrics["f1"],
        })

        print(f"  [{i+1:02d}/{len(folders)}] {run_name:<10}  "
              f"Acc={metrics['acc']:.2f}%  Sen={metrics['sen']:.2f}%  "
              f"Spe={metrics['spe']:.2f}%  F1={metrics['f1']:.2f}%")

    return rows


# ============================================================
# Summary + Welch t-test
# ============================================================

def print_summary(rows: list[dict], args, mod) -> None:
    if not rows:
        print("\nNo completed runs — nothing to summarise.")
        return

    metrics_keys = ["Accuracy", "Sensitivity", "Specificity", "F1-Score"]
    arrays = {k: np.array([r[k] for r in rows]) for k in metrics_keys}
    n = len(rows)

    print("\n" + "=" * 60)
    print(f"RESULTS  (mean ± std,  n={n} runs)  "
          f"[{args.algorithm} / {args.fusion.upper()}]")
    print("=" * 60)
    for k in metrics_keys:
        arr = arrays[k]
        print(f"  {k:<15}: {np.mean(arr):.2f}  ({np.std(arr, ddof=1):.2f})")

    # Optional Welch t-test — only runs when the user supplied BN baselines
    bn_map = {
        "Accuracy":    (args.bn_acc, args.bn_acc_std),
        "Sensitivity": (args.bn_sen, args.bn_sen_std),
        "Specificity": (args.bn_spe, args.bn_spe_std),
        "F1-Score":    (args.bn_f1,  args.bn_f1_std),
    }
    # All four defaults are None — only print test if at least one was supplied
    if any(v[0] is not None for v in bn_map.values()):
        model_label = "NF-MLP" if args.fusion == "nfmlp" else "GTF"
        print("\n" + "=" * 60)
        print(f"WELCH T-TEST  {model_label} vs BN  "
              f"(one-sided, H1: {model_label} > BN)")
        print("=" * 60)
        for k in metrics_keys:
            mn, sd = bn_map[k]
            if mn is None or sd is None:
                continue
            p   = mod.welch_test(arrays[k], mn, sd, n_bn=n)
            sig = "* (p<0.05)" if p < 0.05 else "n.s."
            direction = f"{model_label} better" if np.mean(arrays[k]) > mn else "BN better"
            print(f"  {k:<15}: p={p:.4f}  {sig}  [{direction}]")

    print()


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fusion runner: NF-MLP or GTF over prediction_data run folders",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--algorithm", choices=VALID_ALGORITHMS, default="DecisionTree",
        help="ML algorithm folder to use as input"
    )
    parser.add_argument(
        "--fusion", choices=VALID_FUSIONS, default="nfmlp",
        help="Fusion model to run: nfmlp (MLP) or gtf (GCN)"
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Run all algorithm x fusion combinations and save results into dl_fusion/"
    )

    # Paths
    parser.add_argument(
        "--data_dir", default="prediction_data",
        help="Root directory containing algorithm sub-folders"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output CSV path (default: {algorithm}_{fusion}_results.csv)"
    )

    # File-name overrides (matches defaults in both fusion scripts)
    parser.add_argument("--train_pred",  default="Trainings_data.csv")
    parser.add_argument("--train_proba", default="Trainings_data_proba.csv")
    parser.add_argument("--test_pred",   default="Test_data.csv")
    parser.add_argument("--test_proba",  default="Test_data_probabilities.csv")

    # Training hyperparameters
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--max_epochs", type=int,   default=300)
    parser.add_argument("--patience",   type=int,   default=20)
    parser.add_argument("--val_frac",   type=float, default=0.10)

    # GTF-only adjacency settings (ignored for nfmlp)
    parser.add_argument("--bn_scoring",     default="k2", choices=["k2", "bic"],
                        help="GTF only: BN structure-learning scoring function")
    parser.add_argument("--bn_max_parents", type=int, default=3,
                        help="GTF only: max parents in BN Hill Climb")

    # BN baseline stats for optional Welch t-test
    # Default None so we can detect whether the user actually passed them
    parser.add_argument("--bn_acc",     type=float, default=None)
    parser.add_argument("--bn_acc_std", type=float, default=None)
    parser.add_argument("--bn_sen",     type=float, default=None)
    parser.add_argument("--bn_sen_std", type=float, default=None)
    parser.add_argument("--bn_spe",     type=float, default=None)
    parser.add_argument("--bn_spe_std", type=float, default=None)
    parser.add_argument("--bn_f1",      type=float, default=None)
    parser.add_argument("--bn_f1_std",  type=float, default=None)

    return parser


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    data_dir    = Path(args.data_dir)
    if args.output:
        output_path = Path(args.output)
    else:
        default_dir = Path(__file__).parent / "dl_fusion_results"
        output_path = default_dir / f"{args.algorithm}_{args.fusion}_results.csv"

    print(f"\nFusion Runner")
    print(f"  Algorithm : {args.algorithm}")
    print(f"  Fusion    : {args.fusion.upper()}")
    print(f"  Data dir  : {data_dir.resolve()}")
    print(f"  Output    : {output_path}")
    print("-" * 60)

    # Batch mode: iterate all algorithm x fusion combinations
    if args.batch:
        combos = []
        for alg in VALID_ALGORITHMS:
            for fus in VALID_FUSIONS:
                combos.append((alg, fus))

        for alg, fus in combos:
            print(f"\nRunning combination: {alg} / {fus.upper()}")
            try:
                folders = discover_run_folders(data_dir, alg)
            except FileNotFoundError as e:
                print(f"  [SKIP COMBO] {alg}/{fus}: {e}")
                continue

            mod = load_fusion_module(fus)
            if fus == "nfmlp":
                rows = run_nfmlp(mod, folders, args)
            else:
                rows = run_gtf(mod, folders, args)

            out_dir = Path(__file__).parent / "dl_fusion_results"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{alg}_{fus}_results.csv"
            if rows:
                write_results_csv(rows, out_path)
            print_summary(rows, args, mod)
        return

    # Non-batch single run
    folders = discover_run_folders(data_dir, args.algorithm)
    print(f"  Found {len(folders)} run folder(s)\n")

    mod = load_fusion_module(args.fusion)

    if args.fusion == "nfmlp":
        rows = run_nfmlp(mod, folders, args)
    else:
        rows = run_gtf(mod, folders, args)

    if rows:
        write_results_csv(rows, output_path)

    print_summary(rows, args, mod)


if __name__ == "__main__":
    main()
