import os
import random
import time
import hydra
import numpy as np
from omegaconf import DictConfig
import logging
from pathlib import Path
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import utils
import hyperparameters as hp
import preprocessing as prep
import bayesian_network as bn
import classification as clf


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # region Folder Paths
    project_root = Path(__file__).parent.parent
    data_parent_path = project_root / Path(cfg.paths.source)
    output_path = project_root / Path(cfg.paths.output)
    data_type = cfg.data.type
    data_folder = data_parent_path / data_type
    # endregion

    # region Experiment Settings
    num_runs = cfg.settings.runs
    global_seed = cfg.settings.seed
    verbose = cfg.settings.verbose

    # endregion

    # region Folder Check
    if not data_folder.exists():
        logging.error(f"Data folder {data_folder} does not exist.")
        return

    if not output_path.exists():
        os.makedirs(output_path)
    # endregion

    # Load the CSV files
    file_list = utils.load_csv_file(data_folder, cfg.data.extension)

    sample_df = pd.read_csv(file_list[0], sep=cfg.data.separator)
    num_subjects = sample_df.shape[0]
    num_tasks = len(file_list)

    # Stacking placeholders
    first_level_train_preds = np.zeros((num_subjects, num_tasks))
    first_level_test_preds = np.zeros((int(num_subjects * cfg.experiment.test_size) + 1, num_tasks))

    first_level_train_preds = pd.DataFrame(index=sample_df[cfg.data.id],
                                           columns=[f'Task_{i + 1}' for i in range(num_tasks)])
    first_level_train_preds_proba = pd.DataFrame(index=sample_df[cfg.data.id],
                                                 columns=[f'Task_{i + 1}' for i in range(num_tasks)])
    first_level_test_preds = pd.DataFrame(index=None,
                                          columns=[cfg.data.id] + [f'Task_{i + 1}' for i in range(num_tasks)])
    first_level_test_preds_proba = pd.DataFrame(index=None,
                                                columns=[cfg.data.id] + [f'Task_{i + 1}' for i in range(num_tasks)])

    logging.info(f"Bayesian Network Analysis Experiment starting...") if verbose else None
    logging.info(f"Number of runs: {num_runs}") if verbose else None
    logging.info(f"Tasks: {num_tasks}") if verbose else None
    logging.info(f"Data Type: {data_type} ") if verbose else None
    logging.info(f"Number of Folds: {cfg.experiment.folds}") if verbose else None

    root_output_folder = utils.get_output_folder(output_path, cfg)

    # Start time
    start_time = time.time()

    for run in range(num_runs):
        logging.info(f"------------------------------------------------") if verbose else None
        run_folder = root_output_folder / f"run_{run + 1}"
        os.makedirs(run_folder) if not run_folder.exists() else None
        seed = global_seed + run
        logging.info(f"Run {run + 1}") if verbose else None
        logging.info(f"Seed: {seed}") if verbose else None

        for file_idx, file in enumerate(file_list):
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

            best_val_score = 0
            best_model = None
            scaler_opt = None

            val_scores = []
            models = []
            all_val_indices = set()

            # Stratified K-Fold Cross Validation to avoid class imbalance into folds
            skf = StratifiedKFold(n_splits=cfg.experiment.folds, shuffle=True, random_state=seed)

            for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train), 1):
                logging.info(f"--------- Fold {fold}--------------") if verbose else None

                X_train_cv, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
                y_train_cv, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

                all_val_indices.update(val_index)

                # Preprocess the data
                if data_type == "ML":
                    X_train_cv, scaler_cv = prep.data_scaling(X_train_cv, cfg.data.id, cfg.scaling.type, verbose)
                    X_val = prep.apply_scaling(X_val, scaler_cv, cfg.data.id, verbose)

                # Hyperparameter tuning using optuna
                best_hyperparameters, best_model_cv, _ = hp.hyperparameter_tuning(cfg.model.name, X_train_cv,
                                                                                  y_train_cv,
                                                                                  X_val, y_val,
                                                                                  n_trials=cfg.optuna.n_trials,
                                                                                  verbose=verbose)

                val_pred = best_model_cv.predict(X_val)
                val_proba = best_model_cv.predict_proba(X_val)[:, 1]
                first_level_train_preds.loc[X_train.index[val_index], f'Task_{file_idx + 1}'] = val_pred
                first_level_train_preds_proba.loc[X_train.index[val_index], f'Task_{file_idx + 1}'] = val_proba

                # Accuracy score
                val_score = utils.compute_metrics(y_val, val_pred)['accuracy']
                val_scores.append(val_score)
                models.append(best_model_cv)

                if verbose:
                    logging.info(f"Best hyperparameters: {best_hyperparameters}")
                    logging.info(f"Validation score: {val_score}")
            assert len(all_val_indices) == len(X_train), "Folds do not contain all different samples"

            # Average validation score
            avg_val_score = np.mean(val_scores)
            logging.info("------------------------------------------------") if verbose else None
            logging.info(f"Average validation score: {avg_val_score}") if verbose else None

            if data_type == "ML":
                X_train, scaler_opt = prep.data_scaling(X_train, cfg.data.id, cfg.scaling.type, verbose)
                X_test = prep.apply_scaling(X_test, scaler_opt, cfg.data.id, verbose)

            # Perform hyperparameter search on the entire training dataset
            best_hyperparameters, best_model, _ = hp.hyperparameter_tuning(cfg.model.name, X_train, y_train,
                                                                           None, None,
                                                                           n_trials=cfg.optuna.n_trials,
                                                                           verbose=verbose)

            best_model.fit(X_train, y_train)

            if cfg.experiment.calibration:
                logging.info("Calibrating model...") if verbose else None
                best_model = CalibratedClassifierCV(estimator=best_model,
                                                    method=cfg.experiment.calibration_method,
                                                    cv=cfg.experiment.calibration_cv)
                best_model.fit(X_train, y_train)

            y_pred_test = best_model.predict(X_test)
            first_level_test_preds[cfg.data.id] = test_df.index.to_list()
            first_level_test_preds[f'Task_{file_idx + 1}'] = y_pred_test

            y_pred_test_proba = best_model.predict_proba(X_test)[:, 1]
            first_level_test_preds_proba[cfg.data.id] = test_df.index.to_list()
            first_level_test_preds_proba[f'Task_{file_idx + 1}'] = y_pred_test_proba

            break
        # Clean predictions
        first_level_train_preds = utils.clean_predictions(first_level_train_preds, cfg.data.id)
        first_level_train_preds['Label'] = y_train.to_numpy()
        first_level_train_preds_proba = utils.clean_predictions(first_level_train_preds_proba, cfg.data.id)
        first_level_train_preds_proba['Label'] = y_train.to_numpy()

        first_level_test_preds['Label'] = y_test.to_numpy()
        first_level_test_preds.sort_values(by=cfg.data.id, inplace=True)

        first_level_test_preds_proba['Label'] = y_test.to_numpy()
        first_level_test_preds_proba.sort_values(by=cfg.data.id, inplace=True)

        # Prepare stacking data
        stacking_trainings_data = first_level_train_preds.drop(cfg.data.id, axis=1)
        stacking_trainings_data_proba = first_level_train_preds_proba.drop(cfg.data.id, axis=1)
        stacking_test_data = first_level_test_preds.drop(cfg.data.id, axis=1)
        stacking_test_data_probabilities = first_level_test_preds_proba.drop(cfg.data.id, axis=1)

        # For Debugging, eliminate later
        stacking_trainings_data = utils.mod_predictions(stacking_trainings_data, cfg.data.target)
        stacking_trainings_data_proba = utils.mod_proba_predictions(stacking_trainings_data_proba, cfg.data.target)
        stacking_test_data = utils.mod_predictions(stacking_test_data, cfg.data.target)
        stacking_test_data_probabilities = utils.mod_proba_predictions(stacking_test_data_probabilities,
                                                                       cfg.data.target)

        # Map predictions to -1 and 1
        stacking_trainings_data.replace({0: -1}, inplace=True)
        stacking_trainings_data_proba.replace({0: -1}, inplace=True)
        stacking_test_data.replace({0: -1}, inplace=True)
        stacking_test_data_probabilities.replace({0: -1}, inplace=True)

        logging.info(f"Stacking trainings data: \n {stacking_trainings_data}") if verbose else None
        logging.info(f"Stacking trainings data proba: \n {stacking_trainings_data_proba}") if verbose else None
        logging.info(f"Stacking test data predictions: \n {stacking_test_data}") if verbose else None
        logging.info(f"Stacking test data probabilities: \n {stacking_test_data_probabilities}") if verbose else None

        if cfg.experiment.stacking_method == 'Bayesian':
            selected_columns, predictions = bn.bayesian_network(cfg, run, run_folder, stacking_trainings_data,
                                                                stacking_trainings_data_proba)

            logging.info(f"Selected tasks: {selected_columns}") if verbose else None

            # Filter the predictions based on the selected tasks from the Bayesian Network
            stacking_test_data = stacking_test_data[selected_columns]
            stacking_test_data_probabilities = stacking_test_data_probabilities[selected_columns]
            selected_columns.remove(cfg.data.target)

            y_true_stck = stacking_test_data[cfg.data.target]
            logging.info(f"Stacking test data: \n {stacking_test_data}") if verbose else None

            # Perform Majority Vote
            mv_pred = clf.majority_vote(stacking_test_data)
            logging.info(f"Majority Vote predictions: \n {mv_pred}") if verbose else None
            mv_metrics = utils.compute_metrics(y_true_stck, mv_pred)
            filename_mv = run_folder / f"majority_vote_metrics_{run + 1}.txt"
            utils.save_metrics_to_file(mv_metrics, filename_mv)

            logging.info(f"Majority Vote metrics: \n {mv_metrics}") if verbose else None

            # Perform Weighted Majority Vote
            wmv_pred = clf.weighted_majority_vote(stacking_test_data, stacking_test_data_probabilities)
            logging.info(f"Weighted Majority Vote predictions: \n {wmv_pred}") if verbose else None
            wmv_metrics = utils.compute_metrics(y_true_stck, wmv_pred)
            filename_wmv = run_folder / f"weighted_majority_vote_metrics_{run + 1}.txt"
            utils.save_metrics_to_file(wmv_metrics, filename_wmv)
            logging.info(f"Weighted Majority Vote metrics: \n {wmv_metrics}") if verbose else None
        elif cfg.experiment.stacking_method == 'Classifier':
            stacking_model = clf.stacking_classification(cfg, stacking_trainings_data, cfg.data.target, seed)
            y_true_stck = stacking_test_data[cfg.data.target]
            X_test_stck = stacking_test_data.drop(cfg.data.target, axis=1)
            y_pred_stck = stacking_model.predict(X_test_stck)
            y_pred_proba_stck = stacking_model.predict_proba(X_test_stck)

            stck_metrics = utils.compute_metrics(y_true_stck, y_pred_stck)

            filename_stck = run_folder / f"stacking_metrics_{run + 1}.txt"
            utils.save_metrics_to_file(stck_metrics, filename_stck)
            logging.info(f"Stacking metrics: \n {stck_metrics}") if verbose else None
        seed += 1
        break

    # End time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Format elapsed time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    seconds, milliseconds = divmod(seconds, 1)
    milliseconds = round(milliseconds * 1000)
    formatted_time = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds + milliseconds / 1000)

    logging.info(f"Elapsed time: {formatted_time} seconds") if verbose else None


if __name__ == "__main__":
    main()
