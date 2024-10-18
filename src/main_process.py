import logging
import os
from pathlib import Path
from typing import Tuple, Dict

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV
import preprocessing as prep
import hyperparameters as hp
import utils
import classification as clf


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
    train_indices = None
    test_indices = None
    train_ids = None
    test_ids = None

    for file_idx, file in enumerate(file_list):
        logging.info(f"-------------------Task {file_idx + 1}-------------------") if verbose else None
        task_df = pd.read_csv(file, sep=cfg.data.separator)

        # Perform initial checks
        if cfg.data.target not in task_df.columns:
            raise ValueError(f"Target column {cfg.data.target} not found in the task data")
        if cfg.data.id not in task_df.columns:
            raise ValueError(f"Id column {cfg.data.id} not found in the task data")
        if not task_df[cfg.data.id].is_unique:
            raise ValueError(f"Id column {cfg.data.id} is not unique")
        if task_df.shape[0] != 174:
            raise ValueError(f"Task data does not contain 174 samples")

        # For the first file, determine the train/test split
        if file_idx == 0:
            X = task_df.drop(columns=[cfg.data.target])
            y = task_df[cfg.data.target]

            train_indices, test_indices = train_test_split(
                np.arange(len(task_df)),test_size=cfg.experiment.test_size, random_state=seed, stratify=y)

            # Store the IDs as sets for validation
            train_ids = set(task_df.iloc[train_indices][cfg.data.id].values)
            test_ids = set(task_df.iloc[test_indices][cfg.data.id].values)

            logging.info(
                f"Train/Test split determined. Train size: {len(train_indices)}, Test size: {len(test_indices)}") \
                if verbose else None

        # Apply the split to the current file
        train_df = task_df.iloc[train_indices].copy()
        test_df = task_df.iloc[test_indices].copy()

        # Verify that the split is consistent with the first file
        current_train_ids = set(train_df[cfg.data.id].values)
        current_test_ids = set(test_df[cfg.data.id].values)

        if current_train_ids != train_ids or current_test_ids != test_ids:
            raise ValueError(f"Inconsistent train/test split in file {file_idx + 1}")

        # Sort DataFrames by ID
        train_df.sort_values(by=cfg.data.id, inplace=True)
        test_df.sort_values(by=cfg.data.id, inplace=True)

        # Set ID as index
        train_df.set_index(cfg.data.id, inplace=True, drop=False)
        train_df.index.name = cfg.data.id_index
        test_df.set_index(cfg.data.id, inplace=True, drop=False)
        test_df.index.name = cfg.data.id_index

        # Log info about the split
        logging.info(f"Task {file_idx + 1} - Train set: {train_df.shape[0]} samples") if verbose else None
        logging.info(f"Task {file_idx + 1} - Test set: {test_df.shape[0]} samples") if verbose else None

        # Prepare data for modeling
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

            # Check if val + train_cv has the same number of samples as X_train
            if len(X_train_cv) + len(X_val) != len(X_train):
                raise ValueError(f"Data integrity error in fold {fold}: "
                                 f"train_cv ({len(X_train_cv)}) + val ({len(X_val)}) != "
                                 f"total train ({len(X_train)})")

            all_val_indices.update(val_index)

            # Create and train model
            model = clf.create_model(cfg.model.name, seed=seed)
            model.fit(X_train_cv, y_train_cv)

            # Make predictions on validation set
            val_pred = model.predict(X_val)
            val_proba = model.predict_proba(X_val)[:, 1]  # Probability of positive class

            # Store predictions and probabilities
            train_predictions.loc[X_train.index[val_index], f'Task_{file_idx + 1}'] = val_pred
            train_probabilities.loc[X_train.index[val_index], f'Task_{file_idx + 1}'] = val_proba

            # Calculate and store validation score
            val_score = utils.compute_metrics(y_val, val_pred)['Accuracy']
            val_scores.append(val_score)
            models.append(model)

            if verbose:
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

        assert len(y_pred_test) == len(y_test), "Predictions do not match the test set"
        assert len(y_pred_test_proba) == len(y_test), "Probabilities do not match the test set"

        test_predictions[cfg.data.id] = test_df.index.to_list()
        test_predictions[f'Task_{file_idx + 1}'] = y_pred_test
        test_probabilities[cfg.data.id] = test_df.index.to_list()
        test_probabilities[f'Task_{file_idx + 1}'] = y_pred_test_proba

    # Clean and prepare final predictions
    train_predictions = utils.clean_predictions(train_predictions, cfg.data.id).copy()

    logging.info(f"Train predictions: \n {train_predictions.to_string()}") if verbose else None
    train_predictions.loc[:, 'Label'] = y_train.to_numpy()

    train_probabilities = utils.clean_predictions(train_probabilities, cfg.data.id).copy()
    train_probabilities.loc[:, 'Label'] = y_train.to_numpy()

    test_predictions = test_predictions.copy()
    test_predictions.loc[:, 'Label'] = y_test.to_numpy()
    test_predictions.sort_values(by=cfg.data.id, inplace=True)

    test_probabilities = test_probabilities.copy()
    test_probabilities.loc[:, 'Label'] = y_test.to_numpy()
    test_probabilities.sort_values(by=cfg.data.id, inplace=True)

    # Check if all the dataframes have the Label column
    if 'Label' not in train_predictions.columns:
        raise ValueError("Label column not found in train predictions")
    if 'Label' not in train_probabilities.columns:
        raise ValueError("Label column not found in train probabilities")
    if 'Label' not in test_predictions.columns:
        raise ValueError("Label column not found in test predictions")
    if 'Label' not in test_probabilities.columns:
        raise ValueError("Label column not found in test probabilities")

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


def combine_datasets(output_folders: Dict[str, Path], run_number: int,
                     verbose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Combine datasets from multiple folders into single dataframes for training and testing.

    This function reads CSV files for each dataset, combines them, and ensures the Label column
    is added from the last appended dataframe.

    Args:
        output_folders (Dict[str, Path]): A dictionary mapping dataset names to their folder paths.
        run_number (int): The current run number.
        verbose (bool, optional): If True, print verbose logging information. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing:
            - Combined training predictions dataframe
            - Combined training probabilities dataframe
            - Combined test predictions dataframe
            - Combined test probabilities dataframe

    Raises:
        ValueError: If no datasets are found to combine.
        Exception: If there's an error reading CSV files for a dataset.
    """
    combined_data = {
        'train_preds': [],
        'train_probs': [],
        'test_preds': [],
        'test_probs': []
    }

    datasets = [dataset for dataset in output_folders.keys() if dataset != "combined"]
    if not datasets:
        raise ValueError("No datasets found to combine.")

    for dataset in datasets:
        folder = output_folders[dataset]
        data_folder = folder / f"run_{run_number}" / "First_level_data"
        try:
            train_preds = pd.read_csv(data_folder / "Trainings_data.csv")
            train_probs = pd.read_csv(data_folder / "Trainings_data_proba.csv")
            test_preds = pd.read_csv(data_folder / "Test_data.csv")
            test_probs = pd.read_csv(data_folder / "Test_data_probabilities.csv")

            # Add dataset name as prefix to column names (except 'Id' and 'Label')
            for df in [train_preds, train_probs, test_preds, test_probs]:
                df.columns = [f"{dataset}_{col}" if col not in ['Id', 'Label'] else col for col in df.columns]

            combined_data['train_preds'].append(train_preds)
            combined_data['train_probs'].append(train_probs)
            combined_data['test_preds'].append(test_preds)
            combined_data['test_probs'].append(test_probs)
        except Exception as e:
            logging.error(f"Error reading CSV files for dataset {dataset}: {str(e)}")
            raise

    # Concatenate DataFrames
    for key in combined_data:
        combined_data[key] = pd.concat(combined_data[key], axis=1)
        # Remove duplicate 'Id' columns
        combined_data[key] = combined_data[key].loc[:, ~combined_data[key].columns.duplicated()]

    # Ensure 'Label' column is present in the final dataframes
    label_column = combined_data['train_preds']['Label']
    for key in combined_data:
        if 'Label' not in combined_data[key].columns:
            combined_data[key]['Label'] = label_column

    # put label column at the end
    for key in combined_data:
        label = combined_data[key].pop('Label')
        combined_data[key]['Label'] = label

    if verbose:
        for key, df in combined_data.items():
            logging.info(f"Combined {key} shape: {df.shape}")

    return (combined_data['train_preds'], combined_data['train_probs'],
            combined_data['test_preds'], combined_data['test_probs'])
