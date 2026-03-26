"""
Neural Fusion MLP (NF-MLP) for Alzheimer's Disease Detection
=============================================================
Deep fusion baseline that concatenates 25 calibrated prediction scores
and 25 confidence scores (50 features total) and learns non-linear
cross-task interactions via a two-layer feedforward network.

Data layout (one folder per run, 30 folders total)
---------------------------------------------------
Each folder must contain exactly four files whose names are matched by
the patterns below (case-insensitive, configurable via CLI):

  Trainings_data.csv       — 121 x 26  predictions {-1,1}  + Label {-1,1}
  Trainings_data_proba.csv — 121 x 26  probabilities [0,1] + Label {-1,1}
  Test_data.csv            —  53 x 26  predictions {-1,1}  + Label {-1,1}
  Test_data_probabilities.csv —  53 x 26 probabilities [0,1] + Label {-1,1}

The split is therefore pre-determined by the folder structure; this
script never re-splits the data.

Usage
-----
  python nf_mlp_fusion.py \\
      --folders run_001 run_002 ... run_030 \\
      --bn_acc 87.61 --bn_acc_std 4.69 \\
      --bn_sen 85.12 --bn_sen_std 7.96 \\
      --bn_spe 90.63 --bn_spe_std 14.18 \\
      --bn_f1  88.28 --bn_f1_std  4.09

  # Alternatively supply a glob / parent directory:
  python nf_mlp_fusion.py --folder_glob "runs/run_*" ...
"""

import argparse
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (accuracy_score, f1_score,
                             recall_score, confusion_matrix)


# ============================================================
# File-name patterns (override via CLI if your names differ)
# ============================================================
DEFAULT_TRAIN_PRED  = "Trainings_data.csv"
DEFAULT_TRAIN_PROBA = "Trainings_data_proba.csv"
DEFAULT_TEST_PRED   = "Test_data.csv"
DEFAULT_TEST_PROBA  = "Test_data_probabilities.csv"
LABEL_COL           = "Label"
N_TASKS             = 25


# ============================================================
# Data loading
# ============================================================

def load_fold(folder: Path,
              train_pred_name:  str,
              train_proba_name: str,
              test_pred_name:   str,
              test_proba_name:  str):
    """
    Load one run-folder and return numpy arrays.

    Labels are remapped from {-1, 1} to {0, 1} for BCELoss.

    Returns
    -------
    X_tr : (n_train, 50)  float32  — [predictions | probabilities]
    y_tr : (n_train,)     int
    X_te : (n_test,  50)  float32
    y_te : (n_test,)      int
    """
    def _load(fname):
        path = folder / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        return pd.read_csv(path)

    tr_pred  = _load(train_pred_name)
    tr_proba = _load(train_proba_name)
    te_pred  = _load(test_pred_name)
    te_proba = _load(test_proba_name)

    task_cols = [c for c in tr_pred.columns if c != LABEL_COL]
    assert len(task_cols) == N_TASKS, (
        f"Expected {N_TASKS} task columns, got {len(task_cols)} in {folder}"
    )

    def _build(pred_df, proba_df):
        pred_arr  = pred_df[task_cols].values.astype(np.float32)   # {-1, 1}
        proba_arr = proba_df[task_cols].values.astype(np.float32)  # [0, 1]
        X = np.concatenate([pred_arr, proba_arr], axis=1)          # (n, 50)
        # remap labels: {-1, 1} -> {0, 1}
        y = ((pred_df[LABEL_COL].values + 1) // 2).astype(int)
        return X, y

    X_tr, y_tr = _build(tr_pred,  tr_proba)
    X_te, y_te = _build(te_pred,  te_proba)

    return X_tr, y_tr, X_te, y_te


# ============================================================
# Model
# ============================================================

class NFMLP(nn.Module):
    """
    Two-hidden-layer MLP for task-prediction fusion.

    Input (50) -> Linear(50,32) -> ReLU -> Dropout(0.3)
               -> Linear(32,16) -> ReLU -> Dropout(0.3)
               -> Linear(16, 1) -> Sigmoid
    """

    def __init__(self, input_dim: int = 50, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        # Xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ============================================================
# Training helpers
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for X_b, y_b in loader:
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_b)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def val_loss_fn(model, X, y, criterion):
    model.eval()
    return criterion(model(X), y).item()


# ============================================================
# Single-fold training + evaluation
# ============================================================

def run_fold(X_tr: np.ndarray,
             y_tr: np.ndarray,
             X_te: np.ndarray,
             y_te: np.ndarray,
             lr:         float = 1e-3,
             batch_size: int   = 32,
             max_epochs: int   = 300,
             patience:   int   = 20,
             val_frac:   float = 0.10) -> dict:
    """
    Train NF-MLP on (X_tr, y_tr), evaluate on (X_te, y_te).

    A small internal validation split (val_frac of training data) is used
    exclusively for early stopping; it is never used for testing.

    Returns
    -------
    dict with keys: acc, sen, spe, f1  (all in percent)
    """
    # ---- validation split for early stopping ----
    n_val   = max(2, int(len(X_tr) * val_frac))
    X_val_t = torch.tensor(X_tr[-n_val:],  dtype=torch.float32)
    y_val_t = torch.tensor(y_tr[-n_val:],  dtype=torch.float32)
    X_fit   = torch.tensor(X_tr[:-n_val],  dtype=torch.float32)
    y_fit   = torch.tensor(y_tr[:-n_val],  dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_fit, y_fit),
        batch_size=min(batch_size, len(X_fit)),
        shuffle=True,
    )

    model     = NFMLP(input_dim=X_tr.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val   = float("inf")
    best_state = None
    no_improve = 0

    for _ in range(max_epochs):
        train_one_epoch(model, loader, optimizer, criterion)
        vl = val_loss_fn(model, X_val_t, y_val_t, criterion)
        if vl < best_val - 1e-6:
            best_val   = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- test ----
    model.eval()
    X_te_t = torch.tensor(X_te, dtype=torch.float32)
    with torch.no_grad():
        probs = model(X_te_t).numpy()

    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_te, preds) * 100.0
    sen = recall_score(y_te, preds, pos_label=1, zero_division=0) * 100.0
    f1  = f1_score(y_te, preds, zero_division=0) * 100.0

    cm          = confusion_matrix(y_te, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spe = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0

    return {"acc": acc, "sen": sen, "spe": spe, "f1": f1}


# ============================================================
# Welch t-test
# ============================================================

def welch_test(sample: np.ndarray,
               bn_mean: float,
               bn_std:  float,
               n_bn:    int = 30) -> float:
    """One-sided Welch t-test. H1: NF-MLP mean > BN mean."""
    t_stat, p_two = stats.ttest_ind_from_stats(
        mean1=float(np.mean(sample)),
        std1 =float(np.std(sample, ddof=1)),
        nobs1=len(sample),
        mean2=bn_mean, std2=bn_std, nobs2=n_bn,
        equal_var=False,
    )
    return p_two / 2.0 if t_stat > 0 else 1.0 - p_two / 2.0


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="NF-MLP deep fusion baseline (pre-split 30-folder input)"
    )

    # ---- folder specification (one of two styles) ----
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--folders", nargs="+", metavar="DIR",
        help="Explicit list of run folders (e.g. run_001 run_002 ... run_030)"
    )
    grp.add_argument(
        "--folder_glob", metavar="PATTERN",
        help='Glob pattern for run folders (e.g. "runs/run_*")'
    )

    # ---- file-name overrides ----
    parser.add_argument("--train_pred",  default=DEFAULT_TRAIN_PRED)
    parser.add_argument("--train_proba", default=DEFAULT_TRAIN_PROBA)
    parser.add_argument("--test_pred",   default=DEFAULT_TEST_PRED)
    parser.add_argument("--test_proba",  default=DEFAULT_TEST_PROBA)

    # ---- training hyper-parameters ----
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--max_epochs", type=int,   default=300)
    parser.add_argument("--patience",   type=int,   default=20)
    parser.add_argument("--val_frac",   type=float, default=0.10,
                        help="Fraction of training data used for early stopping")

    # ---- BN baselines for statistical test ----
    parser.add_argument("--bn_acc",     type=float, default=87.61)
    parser.add_argument("--bn_acc_std", type=float, default=4.69)
    parser.add_argument("--bn_sen",     type=float, default=85.12)
    parser.add_argument("--bn_sen_std", type=float, default=7.96)
    parser.add_argument("--bn_spe",     type=float, default=90.63)
    parser.add_argument("--bn_spe_std", type=float, default=14.18)
    parser.add_argument("--bn_f1",      type=float, default=88.28)
    parser.add_argument("--bn_f1_std",  type=float, default=4.09)

    args = parser.parse_args()

    # ---- resolve folder list ----
    if args.folders:
        folders = [Path(f) for f in args.folders]
    else:
        folders = sorted(Path(p) for p in glob.glob(args.folder_glob))

    if not folders:
        parser.error("No folders found. Check --folders or --folder_glob.")

    print(f"\nNF-MLP Fusion  |  {len(folders)} run folder(s) found")
    print(f"Architecture : 50 -> 32 -> 16 -> 1  (ReLU, Dropout=0.3)")
    print(f"Optimizer    : Adam  lr={args.lr}  weight_decay=1e-5")
    print(f"Early stop   : patience={args.patience}  val_frac={args.val_frac}")
    print("-" * 60)

    # ---- iterate over folders ----
    results = {"acc": [], "sen": [], "spe": [], "f1": []}

    for i, folder in enumerate(folders):
        try:
            X_tr, y_tr, X_te, y_te = load_fold(
                folder,
                args.train_pred, args.train_proba,
                args.test_pred,  args.test_proba,
            )
        except FileNotFoundError as e:
            print(f"  [SKIP] {folder.name}: {e}")
            continue

        metrics = run_fold(
            X_tr, y_tr, X_te, y_te,
            lr         = args.lr,
            batch_size = args.batch_size,
            max_epochs = args.max_epochs,
            patience   = args.patience,
            val_frac   = args.val_frac,
        )

        for k in results:
            results[k].append(metrics[k])

        print(f"  Fold {i+1:02d}/{len(folders)}  [{folder.name}]  "
              f"Acc={metrics['acc']:.2f}%  "
              f"Sen={metrics['sen']:.2f}%  "
              f"Spe={metrics['spe']:.2f}%  "
              f"F1={metrics['f1']:.2f}%")

    if not results["acc"]:
        print("\nNo folds completed successfully. Exiting.")
        return

    results_np = {k: np.array(v) for k, v in results.items()}
    n_runs     = len(results_np["acc"])

    # ---- summary ----
    print("\n" + "=" * 60)
    print(f"NF-MLP RESULTS  (mean ± std,  n={n_runs} runs)")
    print("=" * 60)
    rows = []
    for metric, label in [("acc", "Accuracy"),
                           ("sen", "Sensitivity"),
                           ("spe", "Specificity"),
                           ("f1",  "F1 Score")]:
        arr = results_np[metric]
        m, s = np.mean(arr), np.std(arr, ddof=1)
        rows.append((label, m, s))
        print(f"  {label:<15}: {m:.2f} ({s:.2f})")

    # ---- Welch t-tests ----
    bn_stats = {
        "acc": (args.bn_acc, args.bn_acc_std),
        "sen": (args.bn_sen, args.bn_sen_std),
        "spe": (args.bn_spe, args.bn_spe_std),
        "f1":  (args.bn_f1,  args.bn_f1_std),
    }

    print("\n" + "=" * 60)
    print("WELCH T-TEST  NF-MLP vs BN  (one-sided, H1: NF-MLP > BN)")
    print("=" * 60)
    for metric, label in [("acc", "Accuracy"),
                           ("sen", "Sensitivity"),
                           ("spe", "Specificity"),
                           ("f1",  "F1 Score")]:
        arr     = results_np[metric]
        mn, sd  = bn_stats[metric]
        p       = welch_test(arr, mn, sd, n_bn=n_runs)
        sig     = "* (p<0.05)" if p < 0.05 else "n.s."
        direct  = "NF-MLP better" if np.mean(arr) > mn else "BN better"
        print(f"  {label:<15}: p={p:.4f}  {sig}  [{direct}]")

    print()


if __name__ == "__main__":
    main()
