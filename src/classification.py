import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import hyperparameters as hp


def majority_vote(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform majority vote on the predictions dataframe.
    :param df: pd.DataFrame
    :return: pd.DataFrame
    """
    return df.mode(axis=1)[0].astype(int)


def weighted_majority_vote(df: pd.DataFrame, df_proba: pd.DataFrame) -> pd.DataFrame:
    """
    Perform weighted majority vote on the predictions dataframe.
    :param df: pd.DataFrame
    :param df_proba: pd.DataFrame
    :return: pd.DataFrame
    """
    weight_df = df * df_proba

    # Create a column WMV with the sum of columns in the dataframe
    weight_df['WMV'] = weight_df.sum(axis=1)

    # create a column Pred that is 1 if WMV is greater than 0, else -1
    weight_df['Pred'] = np.where(weight_df['WMV'] > 0, 1, -1)

    return weight_df['Pred'].astype(int)


def stacking_classification(cfg: dict, df: pd.DataFrame, target: str = "Label", seed: int = 42, verbose: bool = False):
    """
    Perform stacking on the predictions dataframe.
    :param cfg: dict
    :param df:
    :param target:
    :param seed:
    :param verbose:
    :return:
    """
    X, y = df.drop(target, axis=1), df[target]

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=cfg.experiment.folds, shuffle=True, random_state=seed)

    # Initialize lists to store results
    fold_accuracies = []
    best_models = []
    best_hyperparameters = []

    # Perform cross-validation
    for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
        logging.info(f"--------- Fold {fold}--------------") if verbose else None

        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Perform hyperparameter tuning
        best_params, best_model, best_score = hp.hyperparameter_tuning(
            model_name=cfg.experiment.stacking_model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            n_trials=cfg.optuna.n_trials,
            scoring='accuracy',
            verbose=verbose
        )

        # Store results
        best_models.append(best_model)
        best_hyperparameters.append(best_params)
        fold_accuracies.append(best_score)

    # Select the best hyperparameters across all folds
    best_fold = np.argmax(fold_accuracies)
    overall_best_hyperparameters = best_hyperparameters[best_fold]

    final_model = None

    if cfg.experiment.stacking_model == "MLP":
        final_model = MLPClassifier(**overall_best_hyperparameters, random_state=seed)
    elif cfg.experiment.stacking_model == "LogisticRegression":
        final_model = LogisticRegression(**overall_best_hyperparameters, random_state=seed)
    else:
        raise ValueError(f"Invalid stacking model: {cfg.experiment.stacking_model}")

    final_model.fit(X, y)

    return final_model
