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


# def get_output_folder(output_path, analysis_type, cfg: dict) -> Path:
#     """
#     Get the output folder based on the configuration parameters.
#     :param output_path: Path
#     :param analysis_type: str
#     :param cfg: dict
#     :return:
#     """
#     if analysis_type == "Combined":
#         root_output_folder = output_path / "Combined" / f"{cfg.data.dataset}_and_{cfg.data.dataset2}"
#     else:
#         root_output_folder = output_path / cfg.data.type / cfg.data.dataset
#
#     root_output_folder = root_output_folder / f"{cfg.model.name}"
#
#     if cfg.experiment.stacking_method == 'Classification':
#         root_output_folder = root_output_folder / f"{cfg.experiment.stacking_model}_stacking"
#     elif cfg.experiment.stacking_method in ['Bayesian', 'MajorityVote', 'WeightedMajorityVote']:
#         root_output_folder = root_output_folder / f"{cfg.experiment.stacking_method}"
#     else:
#         raise ValueError(f"Invalid stacking method: {cfg.experiment.stacking_method}")
#
#     return root_output_folder


# def get_output_folder(output_paths: List[Path], analysis_type: str, cfg: DictConfig) -> Path:
#     """
#     Get the output folder based on the configuration parameters and analysis type.
#
#     :param output_paths: List of output paths from build_data_paths function
#     :param analysis_type: Analysis type determined by build_data_paths function
#     :param cfg: Hydra configuration object
#     :return: Path object representing the output folder
#     """
#     if analysis_type == "Combined":
#         # Create a folder name that combines all dataset names
#         datasets = "_".join(path.name for path in output_paths)
#         root_output_folder = output_paths[0].parent / "Combined" / datasets
#     else:
#         # Use the single output path
#         root_output_folder = output_paths[0]
#
#     # Add model name to the path
#     root_output_folder = root_output_folder / f"{cfg.model.name}"
#
#     # Add stacking method to the path
#     if cfg.experiment.stacking_method == 'Classification':
#         root_output_folder = root_output_folder / f"{cfg.experiment.stacking_model}_stacking"
#     elif cfg.experiment.stacking_method in ['Bayesian', 'MajorityVote', 'WeightedMajorityVote']:
#         root_output_folder = root_output_folder / f"{cfg.experiment.stacking_method}"
#     else:
#         raise ValueError(f"Invalid stacking method: {cfg.experiment.stacking_method}")
#
#     return root_output_folder


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
        combined_folder = output_paths[0].parent / "Combined" / datasets
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


def check_existing_data(output_folders: Dict[str, Path]) -> bool:
    """
    Check if all 30 run folders have required CSV files in the First_level_data folder for each dataset.

    :param output_folders: Dictionary of output folders from get_output_folder function
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
            continue  # Skip the combined folder for this check

        for run in range(1, 31):  # Check all 30 runs
            run_folder = folder / f"run_{run}" / "First_level_data"

            # Check if all required CSV files exist for this run
            all_files_exist = all((run_folder / csv_file).exists() for csv_file in required_csv_files)

            if not all_files_exist:
                logging.info(f"Missing CSV files in {run_folder}")
                all_datasets_have_csv = False
                break  # No need to check further runs for this dataset

        if not all_datasets_have_csv:
            break  # No need to check further datasets

    return all_datasets_have_csv

# def get_output_folder(output_path, cfg: dict) -> Path:
#     """
#     Get the output folder based on the configuration parameters.
#     :param output_path:
#     :param cfg: dict
#     :return:
#     """
#     root_output_folder = output_path / cfg.data.type / cfg.data.dataset
#
#     root_output_folder = root_output_folder / f"{cfg.model.name}_base_clf"
#
#     if cfg.experiment.stacking_method == 'Classification':
#         root_output_folder = root_output_folder / f"{cfg.experiment.stacking_model}_stacking_clf"
#     elif cfg.experiment.stacking_method == 'Bayesian':
#         root_output_folder = root_output_folder / f"{cfg.experiment.stacking_method}_stacking_clf"
#         # f"{cfg.bayesian_net.algorithm}_{cfg.bayesian_net.prior_type}")
#
#         # if cfg.bayesian_net.use_parents:
#         #     root_output_folder = root_output_folder / f"max_parents_{cfg.experiment.max_parents}"
#         # else:
#         #     root_output_folder = root_output_folder / "no_max_parents"
#     elif cfg.experiment.stacking_method == 'MajorityVote':
#         root_output_folder = root_output_folder / f"Pure_{cfg.experiment.stacking_method}"
#     elif cfg.experiment.stacking_method == 'WeightedMajorityVote':
#         root_output_folder = root_output_folder / f"Pure_{cfg.experiment.stacking_method}"
#     else:
#         raise ValueError(f"Invalid stacking method: {cfg.experiment.stacking_method}")
#
#     return root_output_folder


# def build_data_paths(cfg, data_parent_path):
#     """
#     Build the data paths based on the configuration parameters.
#     :param cfg:
#     :param data_parent_path:
#     :return:
#     """
#     paths = []
#
#     # Build the first path
#     data_type = cfg.data.type
#     data_folder = data_parent_path / data_type
#     data_folder = data_folder / cfg.data.dataset
#
#     # Check if data_folder exists
#     if not data_folder.exists():
#         logging.error(f"Data folder {data_folder} does not exist.")
#         return paths
#     paths.append(data_folder)
#
#     # Check if type_2 is not None
#     if cfg.data.type_2 != "None":
#         # Build the second path
#         data_type_2 = cfg.data.type_2
#         data_folder_2 = data_parent_path / data_type_2
#         data_folder_2 = data_folder_2 / cfg.data.dataset2
#         paths.append(data_folder_2)
#
#     return paths


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


def mod_predictions(df: pd.DataFrame, target: str = "Label") -> pd.DataFrame:
    """
    Modify the predictions dataframe by filling NaN values with random binary values and converting the columns to
    integer type.
    :param df: pd.DataFrame
    :param target: str
    :return: pd.DataFrame
    """

    def random_binary_fill(x):
        return x if pd.notnull(x) else random.choice([0, 1])

    # Apply random filling to all columns except 'Label'
    for col in df.columns:
        if col != target:
            df[col] = df[col].apply(random_binary_fill)

    # Ensure all columns are of type int
    for col in df.columns:
        df[col] = df[col].astype(int)

    return df


def mod_proba_predictions(df: pd.DataFrame, target: str = "Label") -> pd.DataFrame:
    """
    Modify the predictions probability dataframe by filling NaN values with random values and converting the columns to
    float type.
    :param df: pd.DataFrame
    :param target: str
    :return: pd.DataFrame
    """

    def random_fill(x):
        return x if pd.notnull(x) else random.uniform(0, 1)

    # Apply random filling to all columns except 'Label'
    for col in df.columns:
        if col != target:
            df[col] = df[col].apply(random_fill)

    # Ensure all columns are of type float
    for col in df.columns:
        if col != target:
            df[col] = df[col].astype(float)
        else:
            df[col] = df[col].astype(int)

    return df


def compute_metrics(y_true, y_pred) -> dict:
    """
    Compute the confusion matrix, accuracy, precision, sensitivity, specificity, F1 score, and MCC.
    :param y_true:
    :param y_pred:
    :return:
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    accuracy = round(accuracy, 5)
    precision = round(precision, 5)
    sensitivity = round(sensitivity, 5)
    f1 = round(f1, 5)
    mcc = round(mcc, 5)

    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Compile metrics into a dictionary
    metrics = {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1_score': f1,
        'mcc': mcc
    }

    return metrics


def format_confusion_matrix(cm, class_labels):
    formatted_cm = (
        f"Confusion Matrix:\n"
        f"                                            Predicted\n"
        f"                      |   {class_labels[0]}   |   {class_labels[1]}   |\n"
        f"Actual {class_labels[0]} |   {cm[0, 0]}          |   {cm[0, 1]}          |\n"
        f"       {class_labels[1]} |   {cm[1, 0]}          |   {cm[1, 1]}          |\n"
    )
    return formatted_cm


def atomic_write_with_retry(file_path: str, content: str, max_retries: int = 5, base_delay: float = 0.1) -> None:
    """
    Write content to a file atomically using a temporary file and rename operation,
    with a retry mechanism for concurrent access.

    :param file_path: Path to the target file
    :param content: Content to write to the file
    :param max_retries: Maximum number of retry attempts
    :param base_delay: Base delay between retries (will be multiplied by attempt number)
    """
    for attempt in range(max_retries):
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=os.path.dirname(file_path)) as temp_file:
                temp_file.write(content)
                temp_file.flush()
                os.fsync(temp_file.fileno())

            # Use shutil.move instead of os.rename for cross-device support
            shutil.move(temp_file.name, file_path)
            return
        except OSError as e:
            if attempt == max_retries - 1:
                raise e
            delay = base_delay * (attempt + 1) * (1 + random.random())
            time.sleep(delay)


def save_metrics_to_file(metrics: Dict, filename: str, verbose: bool = False) -> None:
    """
    Save the metrics to a text file using atomic write with retry mechanism.

    :param metrics: Dictionary containing metric values
    :param filename: Path to the output file
    :param verbose: Flag to enable verbose logging
    """
    try:
        content = "Metrics:\n"
        for key, value in metrics.items():
            content += f"{key}: {value}\n"

        atomic_write_with_retry(filename, content)

        if verbose:
            logging.info(f"Metrics saved to {filename}")
    except Exception as e:
        logging.error(f"Error saving metrics to {filename}: {str(e)}")


def save_metrics_bn_to_file(metrics: Dict, filename: str, verbose: bool = False) -> None:
    """
    Save the metrics of the BN to a text file using atomic write with retry mechanism.

    :param metrics: Dictionary containing metric values
    :param filename: Path to the output file
    :param verbose: Flag to enable verbose logging
    """
    try:
        content = "Metrics from Bayesian Network:\n"
        for key, value in metrics.items():
            content += f"{key}: {value}\n"

        atomic_write_with_retry(filename, content)

        if verbose:
            logging.info(f"Bayesian Network metrics saved to {filename}")
    except Exception as e:
        logging.error(f"Error saving Bayesian Network metrics to {filename}: {str(e)}")

# def save_metrics_to_file(metrics: dict, filename: str) -> None:
#     """
#     Save the metrics to a text file.
#     :param metrics:
#     :param filename:
#     :return:
#     """
#     # class_labels = ['Class 0', 'Class 1']
#     # formatted_cm = format_confusion_matrix(metrics['confusion_matrix'], class_labels)
#
#     # check if the file exists otherwise create it
#     if not os.path.exists(filename):
#         with open(filename, 'w') as f:
#             f.write("Metrics:\n")
#             f.write(f"Accuracy: {metrics['accuracy']}\n")
#             f.write(f"Precision: {metrics['precision']}\n")
#             f.write(f"Sensitivity: {metrics['sensitivity']}\n")
#             f.write(f"Specificity: {metrics['specificity']}\n")
#             f.write(f"F1 Score: {metrics['f1_score']}\n")
#             f.write(f"MCC: {metrics['mcc']}\n")
#
#
# def save_metrics_bn_to_file(metrics: dict, filename: str, verbose: bool = False) -> None:
#     """
#     Save the metrics of the BN to a text file.
#     :param metrics:
#     :param filename:
#     :param verbose:
#     :return:
#     """
#     # class_labels = ['Class 0', 'Class 1']
#     # formatted_cm = format_confusion_matrix(metrics['confusion_matrix'], class_labels)
#     logging.info(f"Metrics: {metrics}") if verbose else None
#
#     # check if the file exists otherwise create it
#     if not os.path.exists(filename):
#         with open(filename, 'w') as f:
#             f.write("Metrics from Bayesian Network:\n")
#             f.write(f"Accuracy: {metrics['accuracy']}\n")
#             f.write(f"Precision: {metrics['precision']}\n")
#             f.write(f"Sensitivity: {metrics['sensitivity']}\n")
#             f.write(f"Specificity: {metrics['specificity']}\n")
#             f.write(f"F1 Score: {metrics['f1_score']}\n")
#             f.write(f"MCC: {metrics['mcc']}\n")
#             f.write(f"Log Likelihood: {metrics['log_likelihood']}\n")
#             f.write(f"BIC: {metrics['bic_score']}\n")


def calculate_average_metrics(root_output_folder):
    """
    Calculate the average metrics for each approach.
    :param root_output_folder:
    :return:
    """
    approach_metrics = defaultdict(lambda: defaultdict(float))
    approach_counts = defaultdict(int)

    for run_folder in root_output_folder.glob('run_*'):
        for metrics_file in run_folder.glob('*_metrics_*.txt'):
            approach = metrics_file.name.split('_metrics_')[0]
            approach_counts[approach] += 1

            with open(metrics_file, 'r') as f:
                content = f.read()
                metrics = re.findall(r'(\w+): ([\d.]+)', content)
                for metric, value in metrics:
                    approach_metrics[approach][metric] += float(value)

    average_metrics = {}
    for approach, metrics in approach_metrics.items():
        count = approach_counts[approach]
        average_metrics[approach] = {metric: value / count for metric, value in metrics.items()}

    return average_metrics
