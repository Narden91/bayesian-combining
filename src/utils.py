import os
import random
import re
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union, Tuple, List

import pandas as pd
import logging

from omegaconf import DictConfig, ListConfig
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef


def get_output_folder(output_paths: List[Path], analysis_type: str, cfg: DictConfig) -> Dict[str, Path]:
    """
    Get the output folders based on the configuration parameters and analysis type.

    :param output_paths: List of output paths from build_data_paths function
    :param analysis_type: Analysis type determined by build_data_paths function
    :param cfg: Hydra configuration object
    :return: Dictionary of Path objects representing the output folders
    """
    output_folders = {}

    # Function to add model and stacking method to a path
    def add_model_and_stacking(path: Path) -> Path:
        path = path / f"{cfg.model.name}"
        if cfg.experiment.stacking_method == 'Classification':
            path = path / f"{cfg.experiment.stacking_model}_stacking"
        elif cfg.experiment.stacking_method in ['Bayesian', 'MajorityVote', 'WeightedMajorityVote']:
            path = path / f"{cfg.experiment.stacking_method}"
        else:
            raise ValueError(f"Invalid stacking method: {cfg.experiment.stacking_method}")
        return path

    if analysis_type == "Combined":
        # Create a folder name that combines all dataset names
        datasets = "_".join(path.name for path in output_paths)
        combined_folder = output_paths[0].parent.parent / "Combined" / datasets
        output_folders["combined"] = add_model_and_stacking(combined_folder)

        # Also create individual folders for each dataset
        for path in output_paths:
            dataset_name = path.name
            output_folders[dataset_name] = add_model_and_stacking(path)
    else:
        # Use the single output path
        dataset_name = output_paths[0].name
        output_folders[dataset_name] = add_model_and_stacking(output_paths[0])

    return output_folders


def check_existing_data(output_folders: Dict[str, Path], run: int = 30) -> bool:
    """
    Check if all 30 run folders have required CSV files in the First_level_data folder for each dataset.

    :param output_folders: Dictionary of output folders from get_output_folder function
    :param run: Number of runs to check (default is 30)
    :return: Boolean flag indicating if all datasets have existing CSV files for all 30 runs
    """
    all_datasets_have_csv = True
    required_csv_files = [
        "Trainings_data.csv",
        "Trainings_data_proba.csv",
        "Test_data.csv",
        "Test_data_probabilities.csv"
    ]

    for dataset, folder in output_folders.items():
        if dataset == "combined":
            continue

        for run in range(1, run + 1):
            run_folder = folder / f"run_{run}" / "First_level_data"

            # Check if all required CSV files exist for this run
            all_files_exist = all((run_folder / csv_file).exists() for csv_file in required_csv_files)

            if not all_files_exist:
                logging.info(f"Missing CSV files in {run_folder}")
                all_datasets_have_csv = False
                break

        if not all_datasets_have_csv:
            break

    return all_datasets_have_csv


def build_data_paths(cfg: DictConfig, data_parent_path: Union[str, Path],
                     output_path: Union[str, Path]) -> Tuple[List[Path], List[Path], str]:
    """
    Build the data paths based on the Hydra configuration parameters.

    :param cfg: Hydra configuration object
    :param data_parent_path: Parent path for data
    :param output_path: Path for output
    :return: Tuple of (list of data paths, list of output paths, analysis type)
    :raises ValueError: If configuration is invalid or paths don't exist
    """
    data_paths = []
    output_paths = []

    # Ensure paths are Path objects
    data_parent_path = Path(data_parent_path)
    output_path = Path(output_path)

    # Validate input parameters
    if not isinstance(cfg, DictConfig) or 'data' not in cfg or 'dataset' not in cfg.data:
        raise ValueError("Invalid Hydra configuration object")

    # Handle both single string and list of strings for dataset
    datasets = cfg.data.dataset if isinstance(cfg.data.dataset, (list, ListConfig)) else [cfg.data.dataset]

    if not datasets:
        raise ValueError("No datasets specified in the configuration.")

    ml_datasets = {"All", "InAir", "OnPaper", "InAirOnPaper"}
    dl_datasets = {"ConvNeXtSmall", "EfficientNetV2S", "InceptionResNetV2", "InceptionV3"}

    for dataset in datasets:
        if dataset in ml_datasets:
            data_type = "ML"
        elif dataset in dl_datasets:
            data_type = "DL"
        else:
            raise ValueError(f"Unknown dataset type: {dataset}")

        data_folder = data_parent_path / data_type / dataset
        output_folder = output_path / data_type / dataset

        logging.info(f"Data folder: {data_folder}")

        if not data_folder.exists():
            raise ValueError(f"Data folder {data_folder} does not exist.")

        data_paths.append(data_folder)
        output_paths.append(output_folder)

    if len(data_paths) == 1:
        analysis_type = data_type
    elif len(data_paths) >= 2:
        analysis_type = "Combined"
    else:
        analysis_type = "Error"

    return data_paths, output_paths, analysis_type


def load_csv_file(folder_path: Path, extension=",") -> list[Path]:
    """
    Load the CSV file from the folder path.
    :param folder_path:
    :param extension:
    :return: list[Path]
    """
    csv_files = list(folder_path.glob(f"*.{extension}"))
    if not csv_files:
        logging.error(f"No CSV files found in {folder_path}.")
        raise FileNotFoundError(f"No CSV files found in {folder_path}.")
    return csv_files


def clean_predictions(df: pd.DataFrame, id_column: str = "Id") -> pd.DataFrame:
    """
    Clean the predictions dataframe by removing rows with NaN values and converting the columns to integer type.
    :param df: pd.DataFrame
    :param id_column: str
    :return: pd.DataFrame
    """
    df.dropna(axis=0, how='all', inplace=True)

    # Create Id column from the index
    df[id_column] = df.index

    # move the 'Id' column to the first position
    columns = [id_column] + [col for col in df.columns if col != id_column]
    cleaned_df = df[columns]

    # Drop index column
    cleaned_df.reset_index(drop=True, inplace=True)

    return cleaned_df


def rename_task_columns(df: pd.DataFrame, suffix: str = 'ML') -> pd.DataFrame:
    """
    Renames 'Task_' columns in the given dataframe by adding a suffix.

    Args:
    df (pd.DataFrame): Input dataframe
    suffix (str): Suffix to add to 'Task_' columns. Default is '_ML'

    Returns:
    pd.DataFrame: Dataframe with renamed columns
    """
    # Create a dictionary for renaming
    rename_dict = {col: f"{col}_{suffix}" for col in df.columns if col.startswith('Task_')}

    # Rename the columns
    df_renamed = df.rename(columns=rename_dict)

    return df_renamed


def merge_task_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, id_column: str = 'Id',
                          target_column: str = 'Label') -> pd.DataFrame:
    """
    Merges two dataframes by adding the 'Task_' columns from the second dataframe to the first,
    and moves the 'Label' column to the end of the resulting dataframe.

    Args:
    df1 (pd.DataFrame): First input dataframe (e.g., with ML tasks)
    df2 (pd.DataFrame): Second input dataframe (e.g., with DL tasks)

    Returns:
    pd.DataFrame: Merged dataframe with all 'Task_' columns and 'Label' at the end
    """
    # Identify 'Task_' columns in the second dataframe
    task_columns_df2 = [col for col in df2.columns if col.startswith('Task_')]

    # Merge the dataframes on 'Id' column
    merged_df = pd.merge(df1, df2[[id_column] + task_columns_df2], on=id_column, how='left')

    # Move 'Label' column to the end
    if target_column in merged_df.columns:
        label_column = merged_df.pop(target_column)
        merged_df[target_column] = label_column

    return merged_df


def read_existing_data(run_folders: Dict[str, Path], run_number: int) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read existing data from the specified run folders.

    Args:
        run_folders: Dictionary mapping dataset names to their run folder paths
        run_number: Current run number

    Returns:
        Tuple containing DataFrames for training predictions, training probabilities,
        test predictions, and test probabilities.
    """
    try:
        all_train_preds = []
        all_train_probs = []
        all_test_preds = []
        all_test_probs = []

        for dataset, folder_path in run_folders.items():
            if dataset == "combined":
                continue

            run_folder = folder_path / f"run_{run_number}"
            logging.info(f"Reading data from {run_folder}")

            # Read data for each dataset
            train_preds_path = run_folder / "First_level_data" / "Trainings_data.csv"
            train_probs_path = run_folder / "First_level_data" / "Trainings_data_proba.csv"
            test_preds_path = run_folder / "First_level_data" / "Test_data.csv"
            test_probs_path = run_folder / "First_level_data" / "Test_data_probabilities.csv"

            # Read and add dataset prefix to column names
            train_preds = pd.read_csv(train_preds_path)
            train_probs = pd.read_csv(train_probs_path)
            test_preds = pd.read_csv(test_preds_path)
            test_probs = pd.read_csv(test_probs_path)

            # Add dataset prefix to task columns
            for df in [train_preds, train_probs, test_preds, test_probs]:
                df.columns = [f"{dataset}_{col}" if col not in ['Id', 'Label'] else col
                              for col in df.columns]

            all_train_preds.append(train_preds)
            all_train_probs.append(train_probs)
            all_test_preds.append(test_preds)
            all_test_probs.append(test_probs)

        # Combine all datasets
        combined_train_preds = pd.concat(all_train_preds, axis=1)
        combined_train_probs = pd.concat(all_train_probs, axis=1)
        combined_test_preds = pd.concat(all_test_preds, axis=1)
        combined_test_probs = pd.concat(all_test_probs, axis=1)

        # Remove duplicate columns
        for df in [combined_train_preds, combined_train_probs,
                   combined_test_preds, combined_test_probs]:
            df = df.loc[:, ~df.columns.duplicated()]

        return (combined_train_preds, combined_train_probs,
                combined_test_preds, combined_test_probs)

    except FileNotFoundError as e:
        logging.error(f"Error reading existing data: {e}")
        return None, None, None, None


def compute_metrics(y_true, y_pred, bic_score=None, k2_score=None, log_likelihood=None):
    """
    Compute the confusion matrix and other metrics.
    Optionally include bic_score, k2_score, and log_likelihood if provided.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_score(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_true, y_pred)

    metrics_dict = {
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Sensitivity': [sensitivity],
        'Specificity': [specificity],
        'F1_Score': [f1],
        'MCC': [matthews_corrcoef(y_true, y_pred)],
        'TN': [tn],
        'FP': [fp],
        'FN': [fn],
        'TP': [tp]
    }

    if bic_score is not None:
        metrics_dict['BIC_Score'] = [bic_score]
    if k2_score is not None:
        metrics_dict['K2_Score'] = [k2_score]
    if log_likelihood is not None:
        metrics_dict['Log_Likelihood'] = [log_likelihood]

    return pd.DataFrame(metrics_dict)


def save_metrics_to_csv(metrics: Union[pd.DataFrame, dict],
                        filename: Union[str, Path],
                        run_number: int,
                        verbose: bool = False,
                        index: bool = False) -> None:
    """
    Save the metrics to a CSV file with error handling, including the run number.

    Args:
    metrics (Union[pd.DataFrame, dict]): Metrics to save, either as a DataFrame or a dictionary
    filename (Union[str, Path]): Path or string for the output file
    run_number (int): The current run number
    verbose (bool, optional): If True, log the save operation. Defaults to False.
    index (bool, optional): If True, write row names (index). Defaults to False.

    Raises:
    ValueError: If the metrics argument is neither a DataFrame nor a dictionary
    IOError: If there's an error writing to the file

    Returns:
    None
    """
    try:
        file_path = Path(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(metrics, dict):
            metrics_df = pd.DataFrame(metrics, index=[0])
        elif isinstance(metrics, pd.DataFrame):
            metrics_df = metrics
        else:
            raise ValueError("Metrics must be either a pandas DataFrame or a dictionary")

        # Add 'Run' column at the beginning
        metrics_df.insert(0, 'Run', run_number)

        metrics_df.to_csv(file_path, index=index)

        if verbose:
            logging.info(f"Metrics for run {run_number} successfully saved to {file_path}")

    except ValueError as ve:
        logging.error(f"Error with metrics data: {ve}")
        raise
    except IOError as io_err:
        logging.error(f"IOError occurred while saving metrics to {file_path}: {io_err}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error occurred while saving metrics: {e}")
        raise


def calculate_average_metrics(root_output_folder, method="Bayesian"):
    """
    Calculate the average metrics for each CSV file across all runs, excluding Markov_Blanket files.
    :param root_output_folder: Path object representing the root output folder.
    :param method: The method used as stacking method for the experiment.
    :return: A dictionary with average metrics for each approach, including the number of runs.
    """
    # Define the metrics
    default_metrics = ["Accuracy", "Precision", "Sensitivity", "Specificity", "F1_Score", "MCC"]
    bayesian_test_metrics = default_metrics + ["BIC_Score", "K2_Score", "Log_Likelihood"]

    # Define dictionaries to store the cumulative sums of metrics and counts of runs per approach
    approach_metrics = defaultdict(lambda: defaultdict(float))
    approach_counts = defaultdict(int)

    # Iterate over all run folders
    for run_folder in root_output_folder.glob('run_*'):
        for csv_file in run_folder.glob('*.csv'):
            if "Markov_Blanket" in csv_file.name:
                continue

            df = pd.read_csv(csv_file)

            # Extract the approach name
            approach = csv_file.stem.split('_metrics')[0]

            # Determine which metrics to use
            if method == "Bayesian" and "test" in csv_file.name.lower():
                required_metrics = bayesian_test_metrics
            else:
                required_metrics = default_metrics

            for metric in required_metrics:
                if metric in df.columns:
                    approach_metrics[approach][metric] += df[metric].mean()
            approach_counts[approach] += 1

    average_metrics = {}
    for approach, metrics in approach_metrics.items():
        count = approach_counts[approach]
        average_metrics[approach] = {metric: value / count for metric, value in metrics.items()}
        average_metrics[approach]["Run_Executed"] = count

    return average_metrics


def save_average_metrics(average_metrics, path_output_current_experiment, debug=False):
    """
    Logs and saves the average metrics to a CSV file.

    :param average_metrics: Dictionary containing average metrics for each approach.
    :param path_output_current_experiment: Path where the CSV file will be saved.
    :param debug: Boolean flag to enable/disable logging.
    """
    if average_metrics:
        logging.info("Average metrics across all runs:") if debug else None

        # Prepare data for DataFrame
        metrics_data = []
        for approach, metrics in average_metrics.items():
            logging.info(f"\n{approach.capitalize()} Approach:") if debug else None
            row = {'Approach': approach.capitalize()}
            for metric, value in metrics.items():
                logging.info(f"  {metric}: {value:.5f}") if debug else None
                row[metric] = value
            metrics_data.append(row)

        # Create DataFrame
        df_metrics = pd.DataFrame(metrics_data)

        # Save DataFrame to CSV
        csv_path = path_output_current_experiment / "average_metrics.csv"
        df_metrics.to_csv(csv_path, index=False)
        logging.info(f"Average metrics saved to {csv_path}") if debug else None
    else:
        logging.warning("No metrics files found or processed.")


def save_average_metrics_to_csv(average_metrics, root_output_folder):
    """
    Save the average metrics into a CSV file, including the number of runs per approach.
    :param average_metrics: Dictionary containing the averaged metrics, including the number of runs.
    :param root_output_folder: Path object representing the root output folder.
    """
    df = pd.DataFrame.from_dict(average_metrics, orient='index')

    average_metrics_csv = root_output_folder / "average_metrics.csv"

    df.to_csv(average_metrics_csv, index=True)
    logging.info(f"Average metrics saved to {average_metrics_csv}")


def calculate_markov_blanket_occurrences(root_output_folder, verbose=False):
    """
    Calculate the occurrences of tasks in the Markov Blanket across all runs.
    :param root_output_folder: Path object representing the root output folder.
    :param verbose: Flag to enable verbose logging.
    :return: A DataFrame with Task, Absolute Occurrences, and Mean Occurrences across runs.
    """
    # Dictionary to store the counts of each task
    task_occurrences = defaultdict(int)
    run_count = 0

    for run_folder in root_output_folder.glob('run_*'):
        # Extract the run number from the folder name
        run_number = run_folder.name.split('_')[1]
        markov_blanket_file = run_folder / f"Markov_Blanket_{run_number}.csv"

        if markov_blanket_file.exists():
            df = pd.read_csv(markov_blanket_file)

            for task in df['Task in the Markov Blanket']:
                task_occurrences[task] += 1

            run_count += 1
        else:
            logging.warning(f"Markov Blanket file not found for {run_folder}") if verbose else None

    if run_count == 0:
        logging.warning("No Markov_Blanket files found.")
        return None

    occurrences_data = {
        "Task": [],
        "Absolute Occurrences": [],
        "Mean Occurrences Across Runs": []
    }

    for task, count in task_occurrences.items():
        occurrences_data["Task"].append(task)
        occurrences_data["Absolute Occurrences"].append(count)
        occurrences_data["Mean Occurrences Across Runs"].append(count / run_count)

    df_occurrences = pd.DataFrame(occurrences_data)

    return df_occurrences


def save_markov_blanket_occurrences_to_csv(df_occurrences, root_output_folder):
    """
    Save the Markov Blanket occurrences data to a CSV file.
    :param df_occurrences: DataFrame containing task occurrences.
    :param root_output_folder: Path object representing the root output folder.
    """
    markov_blanket_csv = root_output_folder / "Markov_blanket_occurrences.csv"

    df_occurrences.sort_values(by='Absolute Occurrences', ascending=False, inplace=True)

    df_occurrences.to_csv(markov_blanket_csv, index=False)
    logging.info(f"Markov Blanket occurrences saved to {markov_blanket_csv}")
