import os
import random
import time
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
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
import main_process as mp


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # region Folder Path
    project_root = Path(__file__).parent.parent
    data_parent_path = project_root / Path(cfg.paths.source)
    output_path = project_root / Path(cfg.paths.output)
    if not output_path.exists():
        os.makedirs(output_path)
    # output_data_cleaned = project_root / Path(cfg.paths.source) / "ML" / cfg.data.dataset

    # region Experiment Settings
    num_runs = cfg.settings.runs
    global_seed = cfg.settings.seed
    verbose = cfg.settings.verbose
    debug = cfg.settings.debug
    # endregion

    data_paths = utils.build_data_paths(cfg, data_parent_path)

    # Validate paths and determine the analysis type
    valid_paths = [path for path in data_paths if path.exists()]

    if len(valid_paths) == 0:
        logging.error("No valid data folders found.")
        return
    elif len(valid_paths) == 1:
        if cfg.data.type not in ["ML", "DL"]:
            logging.error(f"Invalid data type {cfg.data.type} for single path.")
            return
        analysis_type = cfg.data.type
    elif len(valid_paths) == 2:
        analysis_type = "Combined"
    else:
        logging.error("Unexpected number of valid data paths.")
        return

    # Load the CSV files
    file_list = utils.load_csv_file(valid_paths[0], cfg.data.extension)
    file_list_comb = utils.load_csv_file(valid_paths[1], cfg.data.extension) if len(valid_paths) > 1 else None

    sample_df = pd.read_csv(file_list[0], sep=cfg.data.separator)
    num_subjects = sample_df.shape[0]
    num_tasks = len(file_list)

    logging.info(f"Bayesian Network Analysis Experiment starting...")
    logging.info(f"Number of runs: {num_runs}")
    logging.info(f"Tasks: {num_tasks}")
    logging.info(f"Analysis Type: {analysis_type}")
    logging.info(f"Number of Folds: {cfg.experiment.folds}")

    root_output_folder = utils.get_output_folder(output_path, cfg)

    # Start time
    start_time = time.time()

    for run in range(num_runs):
        run_folder = root_output_folder / f"run_{run + 1}"
        os.makedirs(run_folder) if not run_folder.exists() else None
        seed = global_seed + run
        logging.info(f"-------------------Run {run + 1} | Seed: {seed}-------------------")

        train_predictions = pd.DataFrame(index=sample_df[cfg.data.id],
                                         columns=[f'Task_{i + 1}' for i in range(num_tasks)])

        train_probabilities = pd.DataFrame(index=sample_df[cfg.data.id],
                                           columns=[f'Task_{i + 1}' for i in range(num_tasks)])

        test_predictions = pd.DataFrame(index=None,
                                        columns=[cfg.data.id] + [f'Task_{i + 1}' for i in range(num_tasks)])

        test_probabilities = pd.DataFrame(index=None,
                                          columns=[cfg.data.id] + [f'Task_{i + 1}' for i in range(num_tasks)])

        if analysis_type == "ML":
            logging.info(f"Machine Learning Analysis") if verbose else None
            train_preds, train_probs, test_preds, test_probs = mp.process_tasks(file_list, cfg, seed, verbose,
                                                                                cfg.data.type,
                                                                                train_predictions, train_probabilities,
                                                                                test_predictions, test_probabilities)
        elif analysis_type == "DL":
            logging.info(f"Deep Learning Analysis") if verbose else None
            train_preds, train_probs, test_preds, test_probs = mp.process_tasks(file_list, cfg, seed, verbose,
                                                                                cfg.data.type,
                                                                                train_predictions, train_probabilities,
                                                                                test_predictions, test_probabilities)
        elif analysis_type == "Combined":
            logging.info(f"Combined Analysis") if verbose else None
            train_preds, train_probs, test_preds, test_probs = mp.combined_analysis(file_list, file_list_comb,
                                                                                    cfg, seed, verbose,
                                                                                    train_predictions,
                                                                                    train_probabilities,
                                                                                    test_predictions,
                                                                                    test_probabilities)
        else:
            raise ValueError(f"Analysis type {analysis_type} not recognized.")

        # Prepare stacking data
        stacking_trainings_data = train_preds.drop(cfg.data.id, axis=1)
        stacking_trainings_data_proba = train_probs.drop(cfg.data.id, axis=1)
        stacking_test_data = test_preds.drop(cfg.data.id, axis=1)
        stacking_test_data_probabilities = test_probs.drop(cfg.data.id, axis=1)

        # Map predictions to -1 and 1
        stacking_trainings_data.replace({0: -1}, inplace=True)
        stacking_trainings_data_proba.replace({0: -1}, inplace=True)
        stacking_test_data.replace({0: -1}, inplace=True)
        stacking_test_data_probabilities.replace({0: -1}, inplace=True)

        first_level_data_folder = run_folder / "First_level_data"
        os.makedirs(first_level_data_folder) if not first_level_data_folder.exists() else None

        # Save the stacking data and probabilities to csv files
        stacking_trainings_data.to_csv(first_level_data_folder / f"Trainings_data.csv", index=False)
        stacking_trainings_data_proba.to_csv(first_level_data_folder / f"Trainings_data_proba.csv", index=False)
        stacking_test_data.to_csv(first_level_data_folder / f"Test_data.csv", index=False)
        stacking_test_data_probabilities.to_csv(first_level_data_folder / f"Test_data_probabilities.csv", index=False)

        logging.info(f"Stacking trainings data: \n {stacking_trainings_data}") if debug else None
        logging.info(f"Stacking trainings data proba: \n {stacking_trainings_data_proba}") if debug else None
        logging.info(f"Stacking test data predictions: \n {stacking_test_data}") if debug else None
        logging.info(f"Stacking test data probabilities: \n {stacking_test_data_probabilities}") if debug else None

        logging.info(f"-------------------Second Level Classification for Run {run + 1} -------------------")

        if cfg.experiment.stacking_method == 'Bayesian':
            (selected_columns, predictions_train, predictions_test, bic_score,
             log_likelihood_train, log_likelihood_test) = bn.bayesian_network(cfg, run, run_folder,
                                                                              stacking_trainings_data,
                                                                              stacking_trainings_data_proba,
                                                                              stacking_test_data)

            # Evaluate results on training data
            y_true_train = stacking_trainings_data[cfg.data.target]
            y_pred_train = predictions_train[cfg.data.target]
            train_metrics = utils.compute_metrics(y_true_train, y_pred_train)
            train_metrics['bic_score'] = bic_score
            train_metrics['log_likelihood'] = log_likelihood_train
            filename_train = run_folder / f"bayesian_network_train_metrics_{run + 1}.txt"
            utils.save_metrics_bn_to_file(train_metrics, filename_train, verbose)
            logging.info(f"Bayesian Network training metrics: \n {train_metrics}") if verbose else None

            # Evaluate results on test data
            y_true_test = stacking_test_data[cfg.data.target]
            y_pred_test = predictions_test[cfg.data.target]
            test_metrics = utils.compute_metrics(y_true_test, y_pred_test)
            test_metrics['bic_score'] = bic_score
            test_metrics['log_likelihood'] = log_likelihood_test
            filename_test = run_folder / f"bayesian_network_test_metrics_{run + 1}.txt"
            utils.save_metrics_bn_to_file(test_metrics, filename_test, verbose)
            logging.info(f"Bayesian Network test metrics: \n {test_metrics}") if verbose else None

            # Filter the predictions based on the selected tasks from the Bayesian Network
            stacking_test_data = stacking_test_data[selected_columns]
            stacking_test_data_probabilities = stacking_test_data_probabilities[selected_columns]
            selected_columns.remove(cfg.data.target)

            y_true_stck = stacking_test_data[cfg.data.target]
            logging.info(f"Stacking test data: \n {stacking_test_data}") if verbose else None

            # Save the columns selected by the Bayesian Network
            selected_columns_file = run_folder / f"Markov_Blanket_{run + 1}.txt"
            with open(selected_columns_file, 'w') as f:
                f.write("Task in the Markov Blanket:\n")
                for column in selected_columns:
                    f.write(f"{column}\n")

            # Perform Majority Vote
            mv_pred = clf.majority_vote(stacking_test_data)
            logging.info(f"Majority Vote predictions: \n {mv_pred}") if verbose else None
            mv_metrics = utils.compute_metrics(y_true_stck, mv_pred)
            filename_mv_bn = run_folder / f"majority_vote_bn_metrics_{run + 1}.txt"
            utils.save_metrics_to_file(mv_metrics, filename_mv_bn)
            logging.info(f"Majority Vote using BN, metrics: \n {mv_metrics}") if verbose else None

            # Perform Weighted Majority Vote
            wmv_pred = clf.weighted_majority_vote(stacking_test_data, stacking_test_data_probabilities)
            logging.info(f"Weighted Majority Vote predictions: \n {wmv_pred}") if verbose else None
            wmv_metrics = utils.compute_metrics(y_true_stck, wmv_pred)
            filename_wmv_bn = run_folder / f"weighted_majority_bn_vote_metrics_{run + 1}.txt"
            utils.save_metrics_to_file(wmv_metrics, filename_wmv_bn)
            logging.info(f"Weighted Majority Vote using BN, metrics: \n {wmv_metrics}") if verbose else None
        elif cfg.experiment.stacking_method == 'Classification':
            stacking_model = clf.stacking_classification(cfg, stacking_trainings_data, cfg.data.target, seed, verbose)
            y_true_stck = stacking_test_data[cfg.data.target]
            X_test_stck = stacking_test_data.drop(cfg.data.target, axis=1)
            y_pred_stck = stacking_model.predict(X_test_stck)
            y_pred_proba_stck = stacking_model.predict_proba(X_test_stck)

            stck_metrics = utils.compute_metrics(y_true_stck, y_pred_stck)
            filename_stck = run_folder / f"stacking_metrics_{run + 1}.txt"
            utils.save_metrics_to_file(stck_metrics, filename_stck)
            logging.info(f"Stacking metrics: \n {stck_metrics}") if verbose else None
        elif cfg.experiment.stacking_method == 'MajorityVote':
            y_true_test = stacking_test_data[cfg.data.target]
            stacking_test_data.drop(cfg.data.target, axis=1, inplace=True)
            stacking_test_data_probabilities.drop(cfg.data.target, axis=1, inplace=True)
            mv_pred = clf.majority_vote(stacking_test_data)
            logging.info(f"Majority Vote predictions: \n {mv_pred}") if verbose else None
            mv_metrics = utils.compute_metrics(y_true_test, mv_pred)
            filename_mv = run_folder / f"weighted_majority_vote_metrics_{run + 1}.txt"
            utils.save_metrics_to_file(mv_metrics, filename_mv)
            logging.info(f"Majority Vote, using first level predictions,metrics: \n {mv_metrics}") if verbose else None
        elif cfg.experiment.stacking_method == 'WeightedMajorityVote':
            y_true_test = stacking_test_data[cfg.data.target]
            stacking_test_data.drop(cfg.data.target, axis=1, inplace=True)
            stacking_test_data_probabilities.drop(cfg.data.target, axis=1, inplace=True)
            wmv_pred = clf.weighted_majority_vote(stacking_test_data, stacking_test_data_probabilities)
            logging.info(f"Weighted Majority Vote predictions: \n {wmv_pred}") if verbose else None
            wmv_metrics = utils.compute_metrics(y_true_test, wmv_pred)
            filename_wmv = run_folder / f"weighted_majority_vote_metrics_{run + 1}.txt"
            utils.save_metrics_to_file(wmv_metrics, filename_wmv)
            logging.info(f"Weighted Majority Vote, using first level predictions, "
                         f"metrics: \n {wmv_metrics}") if verbose else None
        seed += 1

    # Calculate average metrics across all runs
    average_metrics = utils.calculate_average_metrics(root_output_folder)
    if average_metrics:
        logging.info("Average metrics across all runs:") if debug else None
        for approach, metrics in average_metrics.items():
            logging.info(f"\n{approach.capitalize()} Approach:") if debug else None
            for metric, value in metrics.items():
                logging.info(f"  {metric}: {value:.5f}") if debug else None

        # Save average metrics to a file
        average_metrics_file = root_output_folder / "average_metrics.txt"
        with open(average_metrics_file, 'w') as f:
            f.write("Average Metrics:\n")
            for approach, metrics in average_metrics.items():
                f.write(f"\n{approach.capitalize()} Approach:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.5f}\n")
        logging.info(f"Average metrics saved to {average_metrics_file}") if verbose else None
    else:
        logging.warning("No metrics files found or processed.")

    # End time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Format elapsed time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    seconds, milliseconds = divmod(seconds, 1)
    milliseconds = round(milliseconds * 1000)
    formatted_time = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds + milliseconds / 1000)

    logging.info(f"Elapsed time: {formatted_time} seconds")

    # save the time taken to run the experiment
    time_file = root_output_folder / "Execution_time.txt"
    with open(time_file, 'w') as f:
        f.write(f"Elapsed time: {formatted_time} seconds")

    # Save the configuration file
    config_file = root_output_folder / "config.yaml"
    with open(config_file, 'w') as f:
        OmegaConf.save(cfg, f)


if __name__ == "__main__":
    main()
