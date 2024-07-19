import logging
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
import preprocessing as prep
import hyperparameters as hp
import utils


def process_tasks(file_list, cfg, seed, verbose, data_type, train_predictions,
                  train_probabilities, test_predictions, test_probabilities):
    """
    Process multiple tasks for machine learning predictions.

    This function reads data from files, splits it into train and test sets,
    performs cross-validation, hyperparameter tuning, and generates predictions
    for both train and test data.

    Args:
        file_list (list): List of file paths containing task data.
        cfg (object): Configuration object containing various settings.
        seed (int): Random seed for reproducibility.
        verbose (bool): If True, print detailed logs.
        data_type (str): Type of data processing ('ML' for machine learning).
        train_predictions (pd.DataFrame): DataFrame to store train set predictions.
        train_probabilities (pd.DataFrame): DataFrame to store train set probabilities.
        test_predictions (pd.DataFrame): DataFrame to store test set predictions.
        test_probabilities (pd.DataFrame): DataFrame to store test set probabilities.

    Returns:
        tuple: Contains four DataFrames - train_predictions, train_probabilities,
               test_predictions, and test_probabilities.
    """
    for file_idx, file in enumerate(file_list):
        logging.info(f"-------------------Task {file_idx + 1}-------------------") if verbose else None
        task_df = pd.read_csv(file, sep=cfg.data.separator)

        train_df, test_df = prep.data_split(task_df, cfg.data.target, cfg.experiment.test_size, seed, verbose)

        # sort by Id column
        train_df.sort_values(by=cfg.data.id, inplace=True)
        test_df.sort_values(by=cfg.data.id, inplace=True)

        # set Id column as index and call it Id_index but keep it as a column
        train_df.set_index(cfg.data.id, inplace=True, drop=False)
        train_df.index.name = cfg.data.id_index

        test_df.set_index(cfg.data.id, inplace=True, drop=False)
        test_df.index.name = cfg.data.id_index

        logging.info(f"Task {file_idx + 1} - Train set: {train_df.shape[0]} samples, ") if verbose else None
        logging.info(f"Task {file_idx + 1} - Test set: {test_df.shape[0]} samples") if verbose else None

        # Check correct split
        test_ids = test_df[cfg.data.id].values
        train_ids = train_df[cfg.data.id].values
        common_ids = np.intersect1d(test_ids, train_ids)
        if common_ids.size > 0:
            raise ValueError(f"Test set contains Ids also present in the train set: {common_ids}")

        X_train = train_df.drop(columns=cfg.data.target)
        y_train = train_df[cfg.data.target]

        X_test = test_df.drop(columns=cfg.data.target)
        y_test = test_df[cfg.data.target]

        # Preprocess the data
        if data_type == "ML":
            X_train, scaler_cv = prep.data_scaling(X_train, cfg.data.id, cfg.scaling.type, verbose)
            X_test = prep.apply_scaling(X_test, scaler_cv, cfg.data.id, verbose)

        val_scores = []
        models = []
        all_val_indices = set()

        # Stratified K-Fold Cross Validation
        skf = StratifiedKFold(n_splits=cfg.experiment.folds, shuffle=True, random_state=seed)

        for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train), 1):
            logging.info(f"--------- Fold {fold}--------------") if verbose else None

            X_train_cv, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_cv, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
            all_val_indices.update(val_index)

            # Hyperparameter tuning using optuna
            best_hyperparameters, best_model_cv, _ = hp.hyperparameter_tuning(cfg.model.name, X_train_cv,
                                                                              y_train_cv,
                                                                              X_val, y_val,
                                                                              n_trials=cfg.optuna.n_trials,
                                                                              verbose=verbose)

            val_pred = best_model_cv.predict(X_val)
            val_proba = best_model_cv.predict_proba(X_val)[:, 1]
            train_predictions.loc[X_train.index[val_index], f'Task_{file_idx + 1}'] = val_pred
            train_probabilities.loc[X_train.index[val_index], f'Task_{file_idx + 1}'] = val_proba

            # Accuracy score but can be changed to other metrics
            val_score = utils.compute_metrics(y_val, val_pred)['accuracy']
            val_scores.append(val_score)
            models.append(best_model_cv)

            if verbose:
                logging.info(f"Best hyperparameters: {best_hyperparameters}")
                logging.info(f"Validation score: {val_score}")
        assert len(all_val_indices) == len(X_train), "Folds do not contain all different samples"

        avg_val_score = np.mean(val_scores)
        if verbose:
            logging.info("------------------------------------------------")
            logging.info(f"Average validation score: {avg_val_score}")

        # Final model training
        best_hyperparameters, best_model, _ = hp.hyperparameter_tuning(
            cfg.model.name, X_train, y_train, None, None,
            n_trials=cfg.optuna.n_trials, verbose=verbose
        )
        best_model.fit(X_train, y_train)

        if cfg.experiment.calibration:
            logging.info("Calibrating model...") if verbose else None
            best_model = CalibratedClassifierCV(
                estimator=best_model,
                method=cfg.experiment.calibration_method,
                cv=cfg.experiment.calibration_cv
            )
            best_model.fit(X_train, y_train)

        y_pred_test = best_model.predict(X_test)
        y_pred_test_proba = best_model.predict_proba(X_test)[:, 1]

        test_predictions[cfg.data.id] = test_df.index.to_list()
        test_predictions[f'Task_{file_idx + 1}'] = y_pred_test
        test_probabilities[cfg.data.id] = test_df.index.to_list()
        test_probabilities[f'Task_{file_idx + 1}'] = y_pred_test_proba

    # Clean and prepare final predictions
    train_predictions = utils.clean_predictions(train_predictions, cfg.data.id).copy()
    train_predictions.loc[:, 'Label'] = y_train.to_numpy()

    train_probabilities = utils.clean_predictions(train_probabilities, cfg.data.id).copy()
    train_probabilities.loc[:, 'Label'] = y_train.to_numpy()

    test_predictions = test_predictions.copy()
    test_predictions.loc[:, 'Label'] = y_test.to_numpy()
    test_predictions.sort_values(by=cfg.data.id, inplace=True)

    test_probabilities = test_probabilities.copy()
    test_probabilities.loc[:, 'Label'] = y_test.to_numpy()
    test_probabilities.sort_values(by=cfg.data.id, inplace=True)

    logging.info(f"Train predictions: \n {train_predictions}") if verbose else None
    logging.info(f"Train probabilities: \n {train_probabilities}") if verbose else None
    logging.info(f"Test predictions: \n {test_predictions}") if verbose else None
    logging.info(f"Test probabilities: \n {test_probabilities}") if verbose else None

    return train_predictions, train_probabilities, test_predictions, test_probabilities


def combined_analysis(file_list, file_list_two, cfg, seed, verbose, train_predictions, train_probabilities,
                      test_predictions, test_probabilities):
    """
    Processes two sets of tasks (e.g., ML and DL) and merges their results.

    Args:
    file_list (list): List of files to process
    file_list_two (list): List of files to process for the second set
    cfg (object): Configuration object
    seed (int): Random seed
    verbose (bool): Verbosity flag
    train_predictions (pd.DataFrame): Initial train predictions dataframe
    train_probabilities (pd.DataFrame): Initial train probabilities dataframe
    test_predictions (pd.DataFrame): Initial test predictions dataframe
    test_probabilities (pd.DataFrame): Initial test probabilities dataframe

    Returns:
    tuple: Merged train_preds, train_probs, test_preds, test_probs
    """
    # Process first set of tasks
    (first_train_preds, first_train_probs,
     first_test_preds, first_test_probs) = process_tasks(file_list, cfg, seed, verbose, cfg.data.type,
                                                         train_predictions, train_probabilities,
                                                         test_predictions, test_probabilities)

    # Rename columns for first set
    first_train_preds = utils.rename_task_columns(first_train_preds, cfg.data.type)
    first_train_probs = utils.rename_task_columns(first_train_probs, cfg.data.type)
    first_test_preds = utils.rename_task_columns(first_test_preds, cfg.data.type)
    first_test_probs = utils.rename_task_columns(first_test_probs, cfg.data.type)

    if verbose:
        logging.info(f"\n{first_train_preds}")

    # Process second set of tasks if type_2 is specified
    if cfg.data.type_2 != "None":
        (second_train_preds, second_train_probs,
         second_test_preds, second_test_probs) = process_tasks(file_list_two, cfg, seed,
                                                               verbose, cfg.data.type_2,
                                                               train_predictions, train_probabilities,
                                                               test_predictions, test_probabilities)

        # Rename columns for second set
        second_train_preds = utils.rename_task_columns(second_train_preds, cfg.data.type_2)
        second_train_probs = utils.rename_task_columns(second_train_probs, cfg.data.type_2)
        second_test_preds = utils.rename_task_columns(second_test_preds, cfg.data.type_2)
        second_test_probs = utils.rename_task_columns(second_test_probs, cfg.data.type_2)

        # Drop the target column from the second set of predictions
        second_train_preds.drop(cfg.data.target, axis=1, inplace=True)
        second_train_probs.drop(cfg.data.target, axis=1, inplace=True)
        second_test_preds.drop(cfg.data.target, axis=1, inplace=True)
        second_test_probs.drop(cfg.data.target, axis=1, inplace=True)

        # Merge the datasets
        train_preds = utils.merge_task_dataframes(first_train_preds, second_train_preds,
                                                  cfg.data.id, cfg.data.target)
        train_probs = utils.merge_task_dataframes(first_train_probs, second_train_probs,
                                                  cfg.data.id, cfg.data.target)
        test_preds = utils.merge_task_dataframes(first_test_preds, second_test_preds,
                                                 cfg.data.id, cfg.data.target)
        test_probs = utils.merge_task_dataframes(first_test_probs, second_test_probs,
                                                 cfg.data.id, cfg.data.target)
    else:
        # If there's no second type, just use the first set
        train_preds = first_train_preds
        train_probs = first_train_probs
        test_preds = first_test_preds
        test_probs = first_test_probs

    return train_preds, train_probs, test_preds, test_probs
