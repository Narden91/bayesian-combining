import csv
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
import results_analysis as ra
import task_analysis as ta


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # region Folder Path
    project_root = Path(__file__).parent.parent
    data_parent_path = project_root / Path(cfg.paths.source)
    output_path = project_root / Path(cfg.paths.output)
    if not output_path.exists():
        os.makedirs(output_path)
    # endregion

    # region Experiment Settings
    num_runs = cfg.settings.runs
    global_seed = cfg.settings.seed
    verbose = cfg.settings.verbose
    debug = cfg.settings.debug
    # endregion

    if cfg.settings.results_analysis:
        logging.info("Results analysis is enabled.")
        output_results = output_path / Path(cfg.settings.type)
        # output_results = output_path / Path("Tasks_analysis") / "XGB_base_clf_Task_16_Task_21" #XGB_base_clf_Task_16_Task_21
        # ra.result_analysis(output_results)
        # ra.result_analysis(output_path, cfg.data.type)  # Organizzare le metriche
        # ra.result_analysis_tasks(output_path, cfg.settings.type)  # Performance Tasks base
        ra.traverse_and_count_tasks_separately(output_results)  # occorrenze task Markov Blanket

        task_counts, average_elements = ra.count_tasks_in_mb(output_results)
        output_results_name = output_results.parts[-1]
        csv_file_path = f"{output_results_name}_task_counts.csv"

        # Write the task counts to a CSV file
        with open(csv_file_path, mode='w', newline='') as csvfile:
            # Create a CSV writer object
            writer = csv.writer(csvfile)
            writer.writerow(task_counts.keys())
            writer.writerow(task_counts.values())

        logging.info(f"Task counts: {task_counts}") if verbose else None
        logging.info(f"Average elements: {average_elements}") if verbose else None
        return
    else:
        logging.info("Results analysis is disabled.")

    try:
        data_paths, output_paths, analysis_type = utils.build_data_paths(cfg, data_parent_path, output_path)
    except ValueError as e:
        logging.error(f"Error in build_data_paths: {e}")
    except Exception as e:
        logging.error(f"Unexpected error in build_data_paths: {e}")

    logging.info(f"Data paths: {data_paths}")

    # Load the CSV files
    # file_list = utils.load_csv_file(data_paths[0], cfg.data.extension)
    # file_list_comb = utils.load_csv_file(data_paths[1], cfg.data.extension) if len(data_paths) > 1 else None

    file_lists = [utils.load_csv_file(folder_path, cfg.data.extension) for folder_path in data_paths]

    sample_df = pd.read_csv(file_lists[0][0], sep=cfg.data.separator)
    num_tasks = 25

    logging.info(f"Bayesian Network Analysis Experiment starting...")
    logging.info(f"Number of runs: {num_runs}")
    logging.info(f"Tasks: {num_tasks}")
    logging.info(f"Analysis Type: {analysis_type}")

    if not cfg.settings.tasks:
        try:
            output_folders = utils.get_output_folder(output_paths, analysis_type, cfg)
        except ValueError as e:
            print(f"Error: {e}")
    else:
        if not isinstance(output_paths, list):
            clf_name = "XGB_base_clf"
            root_output_folder = output_path / "Tasks_analysis" / clf_name
            os.makedirs(root_output_folder) if not root_output_folder.exists() else None
        else:
            raise ValueError("Tasks analysis is not supported for combined analysis yet.")

    logging.info(f"Output folders: {output_folders}") if verbose else None

    if len(output_folders.keys()) == 1:
        root_output_folder = list(output_folders.values())[0]
        print(f"Output folder find: {root_output_folder}")
    elif len(output_folders.keys()) > 1:
        if "combined" in output_folders:
            root_output_folder = output_folders["combined"]
            print(f"Combined output folder: {root_output_folder}")
            for dataset, folder in output_folders.items():
                if dataset != "combined":
                    print(f"Output folder for {dataset}: {folder}")
        else:
            raise ValueError("Multiple output folders found but no combined folder.")

    data_flag = utils.check_existing_data(output_folders)
    read_data = False

    # Start time
    start_time = time.time()

    for run in range(num_runs):
        run_folder = root_output_folder / f"run_{run + 1}"
        os.makedirs(run_folder) if not run_folder.exists() else None
        seed = global_seed + run
        logging.info(f"-------------------Run {run + 1} | Seed: {seed}-------------------")
        task_columns = [f'Task_{i + 1}' for i in range(num_tasks)]

        train_predictions = pd.DataFrame(index=sample_df[cfg.data.id], columns=task_columns)
        train_probabilities = pd.DataFrame(index=sample_df[cfg.data.id], columns=task_columns)
        test_predictions = pd.DataFrame(index=None, columns=[cfg.data.id] + task_columns)
        test_probabilities = pd.DataFrame(index=None, columns=[cfg.data.id] + task_columns)

        logging.info(f"Run folder: {run_folder}") if verbose else None

        if not cfg.settings.tasks:
            logging.info("Tasks analysis disabled.") if verbose else None
            if analysis_type in ["ML", "DL"]:
                logging.info(f"{analysis_type} Analysis") if verbose else None

                if data_flag:
                    logging.info("Existing data found.") if verbose else None
                    train_preds, train_probs, test_preds, test_probs = utils.read_existing_data(run_folder)

                    logging.info(f"Train predictions: \n {train_preds}") if verbose else None
                    logging.info(f"Train probabilities: \n {train_probs}") if verbose else None
                    logging.info(f"Test predictions: \n {test_preds}") if verbose else None
                    logging.info(f"Test probabilities: \n {test_probs}") if verbose else None
                    read_data = True
                else:
                    train_preds, train_probs, test_preds, test_probs = mp.process_tasks(file_lists[0], cfg, seed,
                                                                                        verbose,
                                                                                        analysis_type,
                                                                                        train_predictions,
                                                                                        train_probabilities,
                                                                                        test_predictions,
                                                                                        test_probabilities)
            elif analysis_type == "Combined":
                logging.info(f"Combined Analysis") if verbose else None

                # Check if all required datasets have existing data
                all_data_exists = all(
                    (folder / f"run_{run + 1}" / "First_level_data" / "Trainings_data.csv").exists() and
                    (folder / f"run_{run + 1}" / "First_level_data" / "Trainings_data_proba.csv").exists() and
                    (folder / f"run_{run + 1}" / "First_level_data" / "Test_data.csv").exists() and
                    (folder / f"run_{run + 1}" / "First_level_data" / "Test_data_probabilities.csv").exists()
                    for folder in output_folders.values() if folder != output_folders.get("combined")
                )

                if all_data_exists:
                    logging.info("Existing data found for all datasets. Combining directly.") if verbose else None

                    try:
                        train_preds, train_probs, test_preds, test_probs = mp.combine_datasets(output_folders, run + 1,
                                                                                               verbose)
                        if verbose:
                            logging.info(f"Combined train predictions: {train_preds}")
                            logging.info(f"Combined train probabilities: {train_probs}")
                            logging.info(f"Combined test predictions: {test_preds}")
                            logging.info(f"Combined test probabilities: {test_probs}")
                        read_data = True
                    except Exception as e:
                        logging.error(f"Error combining datasets: {str(e)}")
                        raise e
                else:
                    logging.info("Processing datasets as not all have existing data.") if verbose else None
                    train_preds, train_probs, test_preds, test_probs = mp.combined_analysis(file_lists, cfg, seed,
                                                                                            verbose,
                                                                                            train_predictions,
                                                                                            train_probabilities,
                                                                                            test_predictions,
                                                                                            test_probabilities)
            else:
                raise ValueError(f"Analysis type {analysis_type} not recognized.")

            if not read_data:
                # Prepare stacking data
                stacking_trainings_data = train_preds.drop(cfg.data.id, axis=1)
                stacking_trainings_data_proba = train_probs.drop(cfg.data.id, axis=1)
                stacking_test_data = test_preds.drop(cfg.data.id, axis=1)
                stacking_test_data_probabilities = test_probs.drop(cfg.data.id, axis=1)

                # Map predictions to -1 and 1 for non-probability dataframes
                stacking_trainings_data.replace({0: -1}, inplace=True)
                stacking_test_data.replace({0: -1}, inplace=True)

                # Map only the 'Label' column for probability dataframes
                stacking_trainings_data_proba[cfg.data.target] = stacking_trainings_data_proba[cfg.data.target].replace(
                    {0: -1})
                stacking_test_data_probabilities[cfg.data.target] = stacking_test_data_probabilities[
                    cfg.data.target].replace({0: -1})

                first_level_data_folder = run_folder / "First_level_data"
                os.makedirs(first_level_data_folder) if not first_level_data_folder.exists() else None

                # Save the stacking data and probabilities to csv files
                stacking_trainings_data.to_csv(first_level_data_folder / f"Trainings_data.csv", index=False)
                stacking_trainings_data_proba.to_csv(first_level_data_folder / f"Trainings_data_proba.csv", index=False)
                stacking_test_data.to_csv(first_level_data_folder / f"Test_data.csv", index=False)
                stacking_test_data_probabilities.to_csv(first_level_data_folder / f"Test_data_probabilities.csv",
                                                        index=False)

                logging.info(f"Stacking trainings data: \n {stacking_trainings_data}") if debug else None
                logging.info(f"Stacking trainings data proba: \n {stacking_trainings_data_proba}") if debug else None
                logging.info(f"Stacking test data predictions: \n {stacking_test_data}") if debug else None
                logging.info(
                    f"Stacking test data probabilities: \n {stacking_test_data_probabilities}") if debug else None
            else:
                logging.info("Data read from existing files, no further manipulation.") if verbose else None
                stacking_trainings_data = train_preds
                stacking_trainings_data_proba = train_probs
                stacking_test_data = test_preds
                stacking_test_data_probabilities = test_probs

        else:
            task_list = list(cfg.settings.task_list)
            logging.info(f"Tasks analysis enabled on tasks: {task_list} | len: {len(task_list)}")

            (stacking_test_data, stacking_test_data_probabilities,
             stacking_trainings_data, stacking_trainings_data_proba) = ta.get_predictions_df(output_path, run + 1,
                                                                                             task_list, clf_name)

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
            train_metrics_df = utils.compute_metrics(y_true_train, y_pred_train)

            # Save metrics to CSV
            filename_train = run_folder / f"bayesian_network_train_metrics_{run + 1}.csv"
            utils.save_metrics_to_csv(train_metrics_df, filename_train, run_number=run + 1, verbose=verbose)

            if verbose:
                logging.info(f"Bayesian Network training metrics: \n {train_metrics_df}")

            # Evaluate results on test data
            y_true_test = stacking_test_data[cfg.data.target]
            y_pred_test = predictions_test[cfg.data.target]
            test_metrics_df = utils.compute_metrics(y_true_test, y_pred_test)

            # Save metrics to CSV
            filename_test = run_folder / f"bayesian_network_test_metrics_{run + 1}.csv"
            utils.save_metrics_to_csv(test_metrics_df, filename_test, run_number=run + 1, verbose=verbose)

            if verbose:
                logging.info(f"Bayesian Network test metrics: \n {test_metrics_df}")

            # Filter the predictions based on the selected tasks from the Bayesian Network
            stacking_test_data = stacking_test_data[selected_columns]
            stacking_test_data_probabilities = stacking_test_data_probabilities[selected_columns]
            selected_columns.remove(cfg.data.target)

            y_true_stck = stacking_test_data[cfg.data.target]
            if verbose:
                logging.info(f"Stacking test data: \n {stacking_test_data}")

            # Save the columns selected by the Bayesian Network
            selected_columns_df = pd.DataFrame({'Task in the Markov Blanket': selected_columns})
            selected_columns_file = run_folder / f"Markov_Blanket_{run + 1}.csv"
            utils.save_metrics_to_csv(selected_columns_df, selected_columns_file, run_number=run + 1,
                                      verbose=verbose, index=False)

            # Perform Majority Vote
            mv_pred = clf.majority_vote(stacking_test_data)
            if verbose:
                logging.info(f"Majority Vote predictions: \n {mv_pred}")

            mv_metrics_df = utils.compute_metrics(y_true_stck, mv_pred)
            filename_mv_bn = run_folder / f"majority_vote_bn_metrics_{run + 1}.csv"
            utils.save_metrics_to_csv(mv_metrics_df, filename_mv_bn, run_number=run + 1, verbose=verbose)

            if verbose:
                logging.info(f"Majority Vote using BN, metrics: \n {mv_metrics_df}")

            # Perform Weighted Majority Vote
            wmv_pred = clf.weighted_majority_vote(stacking_test_data, stacking_test_data_probabilities)

            wmv_metrics_df = utils.compute_metrics(y_true_stck, wmv_pred)
            filename_wmv_bn = run_folder / f"weighted_majority_bn_vote_metrics_{run + 1}.csv"
            utils.save_metrics_to_csv(wmv_metrics_df, filename_wmv_bn, run_number=run + 1, verbose=verbose)

            if verbose:
                logging.info(f"Weighted Majority Vote using BN, metrics: \n {wmv_metrics_df}")

        elif cfg.experiment.stacking_method == 'Classification':
            stacking_model = clf.stacking_classification(cfg, stacking_trainings_data, cfg.data.target, seed, verbose)
            y_true_stck = stacking_test_data[cfg.data.target]
            X_test_stck = stacking_test_data.drop(cfg.data.target, axis=1)
            y_pred_stck = stacking_model.predict(X_test_stck)
            y_pred_proba_stck = stacking_model.predict_proba(X_test_stck)

            stck_metrics = utils.compute_metrics(y_true_stck, y_pred_stck)
            filename_stck = run_folder / f"stacking_{cfg.experiment.stacking_model}_metrics_{run + 1}.csv"
            utils.save_metrics_to_csv(stck_metrics, filename_stck, run_number=run + 1, verbose=verbose)
            logging.info(f"Stacking metrics: \n {stck_metrics}") if verbose else None
        elif cfg.experiment.stacking_method == 'MajorityVote':
            y_true_test = stacking_test_data[cfg.data.target]
            stacking_test_data.drop(cfg.data.target, axis=1, inplace=True)
            stacking_test_data_probabilities.drop(cfg.data.target, axis=1, inplace=True)
            mv_pred = clf.majority_vote(stacking_test_data)

            logging.info(f"Majority Vote predictions: \n {mv_pred}") if verbose else None

            mv_metrics = utils.compute_metrics(y_true_test, mv_pred)
            filename_mv = run_folder / f"weighted_majority_vote_metrics_{run + 1}.csv"
            utils.save_metrics_to_csv(mv_metrics, filename_mv, run_number=run + 1, verbose=verbose)
            logging.info(f"Majority Vote, using first level predictions,metrics: \n {mv_metrics}") if verbose else None
        elif cfg.experiment.stacking_method == 'WeightedMajorityVote':
            y_true_test = stacking_test_data[cfg.data.target]
            stacking_test_data.drop(cfg.data.target, axis=1, inplace=True)
            stacking_test_data_probabilities.drop(cfg.data.target, axis=1, inplace=True)
            wmv_pred = clf.weighted_majority_vote(stacking_test_data, stacking_test_data_probabilities)

            logging.info(f"Weighted Majority Vote predictions: \n {wmv_pred}") if verbose else None

            wmv_metrics = utils.compute_metrics(y_true_test, wmv_pred)
            filename_wmv = run_folder / f"weighted_majority_vote_metrics_{run + 1}.csv"
            utils.save_metrics_to_csv(wmv_metrics, filename_wmv, run_number=run + 1, verbose=verbose)
            logging.info(f"Weighted Majority Vote, using first level predictions, "
                         f"metrics: \n {wmv_metrics}") if verbose else None
        seed += 1

    # Calculate average metrics across all runs
    average_metrics = utils.calculate_average_metrics(root_output_folder)

    if cfg.experiment.stacking_method == 'Bayesian':
        df_occurrences = utils.calculate_markov_blanket_occurrences(root_output_folder, verbose)
        if df_occurrences is not None:
            utils.save_markov_blanket_occurrences_to_csv(df_occurrences, root_output_folder)

    if average_metrics:
        logging.info("Average metrics across all runs:") if debug else None
        for approach, metrics in average_metrics.items():
            logging.info(f"\n{approach.capitalize()} Approach:") if debug else None
            for metric, value in metrics.items():
                logging.info(f"  {metric}: {value:.5f}") if debug else None

        # Save average metrics to a CSV file
        utils.save_average_metrics_to_csv(average_metrics, root_output_folder)
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


if __name__ == "__main__":
    main()
