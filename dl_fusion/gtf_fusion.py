"""
Graph-based Task Fusion (GTF) for Alzheimer's Disease Detection
===============================================================
Deep fusion baseline that represents the 25 tasks as nodes in a graph
whose adjacency is initialised from the BN structure learned on the
training fold.  A two-layer GCN aggregates neighbour information and
a global-mean-pool readout feeds a linear classifier.

Data layout (one folder per run, 30 folders total)
---------------------------------------------------
Each folder must contain exactly four files:

  Trainings_data.csv           — 121 x 26  predictions {-1,1}  + Label {-1,1}
  Trainings_data_proba.csv     — 121 x 26  probabilities [0,1] + Label {-1,1}
  Test_data.csv                —  53 x 26  predictions {-1,1}  + Label {-1,1}
  Test_data_probabilities.csv  —  53 x 26  probabilities [0,1] + Label {-1,1}

The train/test split is pre-determined by the folder structure; this
script never re-splits the data.

Key design choice
-----------------
The BN adjacency matrix is estimated on the TRAINING data of each fold
using pgmpy Hill Climbing (K2 score, max_parents=3) — exactly the same
settings as the paper.  The test set is never touched during adjacency
estimation.

Dependencies
------------
  pip install torch scikit-learn scipy pandas numpy pgmpy

Usage
-----
  python gtf_fusion.py \\
      --folders run_001 run_002 ... run_030 \\
      --bn_acc 87.61 --bn_acc_std 4.69 \\
      --bn_sen 85.12 --bn_sen_std 7.96 \\
      --bn_spe 90.63 --bn_spe_std 14.18 \\
      --bn_f1  88.28 --bn_f1_std  4.09

  # Or with a glob:
  python gtf_fusion.py --folder_glob "runs/run_*" ...
"""

import argparse
import glob
import warnings
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

from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.estimators import HillClimbSearch, K2 as K2Score, BIC as BicScore


# ============================================================
# Constants / defaults
# ============================================================
DEFAULT_TRAIN_PRED  = "Trainings_data.csv"
DEFAULT_TRAIN_PROBA = "Trainings_data_proba.csv"
DEFAULT_TEST_PRED   = "Test_data.csv"
DEFAULT_TEST_PROBA  = "Test_data_probabilities.csv"
LABEL_COL           = "Label"
N_TASKS             = 25
TASK_COLS           = [f"Task_{i}" for i in range(1, N_TASKS + 1)]


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

    Labels remapped from {-1, 1} -> {0, 1} for BCELoss.

    Returns
    -------
    pred_tr  : (n_train, 25) int predictions {-1,1}  (kept for BN learning)
    proba_tr : (n_train, 25) float probabilities [0,1]
    y_tr     : (n_train,)    int {0,1}
    pred_te  : (n_test,  25)
    proba_te : (n_test,  25)
    y_te     : (n_test,)     int {0,1}
    """
    def _load(fname):
        path = folder / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        return pd.read_csv(path)

    tr_pred  = _load(train_pred_name)
    tr_proba = _load(train_proba_name)
    te_pred  = _load(test_pred_name)
    te_proba = _load(test_proba_name)

    actual_task_cols = [c for c in tr_pred.columns if c != LABEL_COL]
    assert len(actual_task_cols) == N_TASKS, (
        f"Expected {N_TASKS} task columns, got {len(actual_task_cols)} in {folder}"
    )

    def _split(pred_df, proba_df):
        pred_arr  = pred_df[TASK_COLS].values.astype(np.float32)
        proba_arr = proba_df[TASK_COLS].values.astype(np.float32)
        y         = ((pred_df[LABEL_COL].values + 1) // 2).astype(int)
        return pred_arr, proba_arr, y

    pred_tr, proba_tr, y_tr = _split(tr_pred,  tr_proba)
    pred_te, proba_te, y_te = _split(te_pred,  te_proba)

    return pred_tr, proba_tr, y_tr, pred_te, proba_te, y_te


# ============================================================
# BN-based adjacency estimation
# ============================================================

def estimate_adjacency(pred_tr:  np.ndarray,
                       y_tr:     np.ndarray,
                       scoring:  str = "k2",
                       max_parents: int = 3) -> np.ndarray:
    """
    Learn BN structure on training predictions + labels using Hill Climbing.
    Returns a symmetric binary adjacency matrix (25 x 25) over task nodes.

    The label node is included during structure learning (as in the paper)
    but is excluded from the returned adjacency — only task-to-task edges
    are kept, because the GCN readout provides the classification signal.
    """
    # Discretise predictions to {0,1} for pgmpy
    pred_disc  = ((pred_tr + 1) // 2).astype(int)   # {-1,1} -> {0,1}
    label_disc = y_tr.astype(int)

    col_names = TASK_COLS + [LABEL_COL]
    df = pd.DataFrame(
        np.column_stack([pred_disc, label_disc]),
        columns=col_names
    )

    score_fn = K2Score(df) if scoring == "k2" else BicScore(df)
    hc       = HillClimbSearch(df)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best_model = hc.estimate(
                scoring_method=score_fn,
                max_indegree=max_parents,
                show_progress=False,
            )
    except Exception:
        best_model = BayesianNetwork()   # empty graph fallback

    # Symmetric adjacency over task nodes only
    A        = np.zeros((N_TASKS, N_TASKS), dtype=np.float32)
    task_set = set(TASK_COLS)

    for u, v in best_model.edges():
        if u in task_set and v in task_set:
            ui = int(u.split("_")[1]) - 1   # "Task_3" -> 2
            vi = int(v.split("_")[1]) - 1
            A[ui, vi] = 1.0
            A[vi, ui] = 1.0                 # symmetric

    return A


def normalise_adjacency(A: np.ndarray) -> torch.Tensor:
    """
    Compute D̃^{-1/2} Ã D̃^{-1/2}  where  Ã = A + I.
    Returns float32 tensor (25, 25).
    """
    n    = A.shape[0]
    A_tl = A + np.eye(n, dtype=np.float32)
    deg  = A_tl.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    norm = d_inv_sqrt[:, None] * A_tl * d_inv_sqrt[None, :]
    return torch.tensor(norm, dtype=torch.float32)


# ============================================================
# GCN layer
# ============================================================

class GCNLayer(nn.Module):
    """
    H' = ReLU( D̃^{-1/2} Ã D̃^{-1/2} H W )

    Parameters
    ----------
    in_features, out_features : layer dimensions.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, H: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        """
        H        : (batch, n_nodes, in_features)
        norm_adj : (n_nodes, n_nodes)
        Returns  : (batch, n_nodes, out_features)
        """
        agg = torch.einsum("bni,mn->bmi", H, norm_adj)
        return torch.relu(self.W(agg))


# ============================================================
# GTF model
# ============================================================

class GTFModel(nn.Module):
    """
    Two-layer GCN + global mean pool + linear classifier.

    Node feature per subject-task pair: [prediction_t, probability_t]  dim=2
    Layer widths: 2 -> 16 -> 8 -> 1
    """

    def __init__(self, node_feat_dim: int = 2,
                 hidden_dim: int = 16,
                 out_dim: int = 8):
        super().__init__()
        self.gcn1       = GCNLayer(node_feat_dim, hidden_dim)
        self.gcn2       = GCNLayer(hidden_dim,    out_dim)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 1),
            nn.Sigmoid(),
        )
        nn.init.xavier_uniform_(self.classifier[0].weight)
        nn.init.zeros_(self.classifier[0].bias)

    def forward(self, H: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        """
        H        : (batch, 25, 2)
        norm_adj : (25, 25)
        Returns  : (batch,)
        """
        H1 = self.gcn1(H, norm_adj)           # (batch, 25, 16)
        H2 = self.gcn2(H1, norm_adj)          # (batch, 25, 8)
        z  = H2.mean(dim=1)                   # (batch, 8)
        return self.classifier(z).squeeze(-1) # (batch,)


# ============================================================
# Build node-feature tensor
# ============================================================

def build_node_features(pred: np.ndarray,
                        proba: np.ndarray) -> torch.Tensor:
    """
    pred  : (n, 25)  values in {-1.0, 1.0}
    proba : (n, 25)  values in [0, 1]
    Returns (n, 25, 2) float32 tensor.
    """
    H = np.stack([pred, proba], axis=-1).astype(np.float32)
    return torch.tensor(H)


# ============================================================
# Training helpers
# ============================================================

def train_one_epoch(model, loader, norm_adj, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for H_b, y_b in loader:
        optimizer.zero_grad()
        loss = criterion(model(H_b, norm_adj), y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_b)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def val_loss_fn(model, H, y, norm_adj, criterion):
    model.eval()
    return criterion(model(H, norm_adj), y).item()


# ============================================================
# Single-fold training + evaluation
# ============================================================

def run_fold(pred_tr:  np.ndarray,
             proba_tr: np.ndarray,
             y_tr:     np.ndarray,
             pred_te:  np.ndarray,
             proba_te: np.ndarray,
             y_te:     np.ndarray,
             bn_scoring:     str = "k2",
             bn_max_parents: int = 3,
             lr:         float = 1e-3,
             batch_size: int   = 32,
             max_epochs: int   = 300,
             patience:   int   = 20,
             val_frac:   float = 0.10) -> dict:
    """
    1. Estimate BN adjacency on training data only.
    2. Build and normalise the graph.
    3. Train GTF with early stopping.
    4. Evaluate on held-out test set.

    Returns
    -------
    dict with keys: acc, sen, spe, f1  (all in percent)
    """
    # ---- BN adjacency (training fold only) ----
    A        = estimate_adjacency(pred_tr, y_tr, bn_scoring, bn_max_parents)
    norm_adj = normalise_adjacency(A)       # (25, 25)

    # ---- node feature tensors ----
    H_full = build_node_features(pred_tr, proba_tr)   # (n_tr, 25, 2)
    H_te   = build_node_features(pred_te, proba_te)   # (n_te, 25, 2)

    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    y_te_t = torch.tensor(y_te, dtype=torch.float32)

    # ---- internal validation split for early stopping ----
    n_val    = max(2, int(len(H_full) * val_frac))
    H_val    = H_full[-n_val:]
    y_val_t  = y_tr_t[-n_val:]
    H_fit    = H_full[:-n_val]
    y_fit_t  = y_tr_t[:-n_val]

    loader = DataLoader(
        TensorDataset(H_fit, y_fit_t),
        batch_size=min(batch_size, len(H_fit)),
        shuffle=True,
    )

    model     = GTFModel(node_feat_dim=2, hidden_dim=16, out_dim=8)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val   = float("inf")
    best_state = None
    no_improve = 0

    for _ in range(max_epochs):
        train_one_epoch(model, loader, norm_adj, optimizer, criterion)
        vl = val_loss_fn(model, H_val, y_val_t, norm_adj, criterion)
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
    with torch.no_grad():
        probs = model(H_te, norm_adj).numpy()

    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_te, preds) * 100.0
    sen = recall_score(y_te, preds, pos_label=1, zero_division=0) * 100.0
    f1  = f1_score(y_te, preds, zero_division=0) * 100.0

    cm             = confusion_matrix(y_te, preds, labels=[0, 1])
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
    """One-sided Welch t-test. H1: GTF mean > BN mean."""
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
        description="GTF deep fusion baseline (GCN, pre-split 30-folder input)"
    )

    # ---- folder specification ----
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--folders", nargs="+", metavar="DIR",
        help="Explicit list of run folders"
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

    # ---- BN adjacency settings ----
    parser.add_argument("--bn_scoring",     default="k2", choices=["k2", "bic"])
    parser.add_argument("--bn_max_parents", type=int, default=3)

    # ---- GCN training hyper-parameters ----
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--max_epochs", type=int,   default=300)
    parser.add_argument("--patience",   type=int,   default=20)
    parser.add_argument("--val_frac",   type=float, default=0.10)

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

    # ---- resolve folders ----
    if args.folders:
        folders = [Path(f) for f in args.folders]
    else:
        folders = sorted(Path(p) for p in glob.glob(args.folder_glob))

    if not folders:
        parser.error("No folders found. Check --folders or --folder_glob.")

    print(f"\nGTF Fusion (GCN)  |  {len(folders)} run folder(s) found")
    print(f"GCN architecture : 2 -> 16 -> 8 -> 1  (global mean pool)")
    print(f"BN adjacency     : HillClimb  scoring={args.bn_scoring}  "
          f"max_parents={args.bn_max_parents}")
    print(f"Optimizer        : Adam  lr={args.lr}  weight_decay=1e-5")
    print(f"Early stop       : patience={args.patience}  val_frac={args.val_frac}")
    print("-" * 65)

    results = {"acc": [], "sen": [], "spe": [], "f1": []}

    for i, folder in enumerate(folders):
        try:
            pred_tr, proba_tr, y_tr, pred_te, proba_te, y_te = load_fold(
                folder,
                args.train_pred, args.train_proba,
                args.test_pred,  args.test_proba,
            )
        except FileNotFoundError as e:
            print(f"  [SKIP] {folder.name}: {e}")
            continue

        metrics = run_fold(
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

        for k in results:
            results[k].append(metrics[k])

        print(f"  Fold {i+1:02d}/{len(folders)}  [{folder.name}]  "
              f"Acc={metrics['acc']:.2f}%  "
              f"Sen={metrics['sen']:.2f}%  "
              f"Spe={metrics['spe']:.2f}%  "
              f"F1={metrics['f1']:.2f}%")

    if not results["acc"]:
        print("\nNo folds completed. Exiting.")
        return

    results_np = {k: np.array(v) for k, v in results.items()}
    n_runs     = len(results_np["acc"])

    # ---- summary ----
    print("\n" + "=" * 65)
    print(f"GTF RESULTS  (mean ± std,  n={n_runs} runs)")
    print("=" * 65)
    for metric, label in [("acc", "Accuracy"),
                           ("sen", "Sensitivity"),
                           ("spe", "Specificity"),
                           ("f1",  "F1 Score")]:
        arr = results_np[metric]
        print(f"  {label:<15}: {np.mean(arr):.2f} ({np.std(arr, ddof=1):.2f})")

    # ---- Welch t-tests ----
    bn_stats = {
        "acc": (args.bn_acc, args.bn_acc_std),
        "sen": (args.bn_sen, args.bn_sen_std),
        "spe": (args.bn_spe, args.bn_spe_std),
        "f1":  (args.bn_f1,  args.bn_f1_std),
    }

    print("\n" + "=" * 65)
    print("WELCH T-TEST  GTF vs BN  (one-sided, H1: GTF > BN)")
    print("=" * 65)
    for metric, label in [("acc", "Accuracy"),
                           ("sen", "Sensitivity"),
                           ("spe", "Specificity"),
                           ("f1",  "F1 Score")]:
        arr    = results_np[metric]
        mn, sd = bn_stats[metric]
        p      = welch_test(arr, mn, sd, n_bn=n_runs)
        sig    = "* (p<0.05)" if p < 0.05 else "n.s."
        direct = "GTF better" if np.mean(arr) > mn else "BN better"
        print(f"  {label:<15}: p={p:.4f}  {sig}  [{direct}]")

    print()


if __name__ == "__main__":
    main()
