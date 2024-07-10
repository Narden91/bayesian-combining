import os
import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import logging

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef


def get_output_folder(output_path, cfg: dict) -> Path:
    """
    Get the output folder based on the configuration parameters.
    :param output_path:
    :param cfg: dict
    :return:
    """
    root_output_folder = output_path / cfg.data.type

    if cfg.experiment.calibration:
        root_output_folder = root_output_folder / "Calibration" / cfg.experiment.stacking_method
    else:
        root_output_folder = root_output_folder / "No_Calibration" / cfg.experiment.stacking_method

    if cfg.experiment.stacking_method == 'Classification':
        root_output_folder = root_output_folder / cfg.experiment.stacking_model
    elif cfg.experiment.stacking_method == 'Bayesian':
        root_output_folder = root_output_folder / f"{cfg.bayesian_net.algorithm}_{cfg.bayesian_net.prior_type}"
        if cfg.bayesian_net.use_parents:
            root_output_folder = root_output_folder / f"max_parents_{cfg.experiment.max_parents}"
        else:
            root_output_folder = root_output_folder / "no_max_parents"
    else:
        raise ValueError(f"Invalid stacking method: {cfg.experiment.stacking_method}")

    return root_output_folder


def load_csv_file(folder_path: str = "", extension=",") -> list[Path]:
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


def save_metrics_to_file(metrics: dict, filename: str) -> None:
    """
    Save the metrics to a text file.
    :param metrics:
    :param filename:
    :return:
    """
    # class_labels = ['Class 0', 'Class 1']
    # formatted_cm = format_confusion_matrix(metrics['confusion_matrix'], class_labels)

    # check if the file exists otherwise create it
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write("Metrics:\n")
            f.write(f"Accuracy: {metrics['accuracy']}\n")
            f.write(f"Precision: {metrics['precision']}\n")
            f.write(f"Sensitivity: {metrics['sensitivity']}\n")
            f.write(f"Specificity: {metrics['specificity']}\n")
            f.write(f"F1 Score: {metrics['f1_score']}\n")
            f.write(f"MCC: {metrics['mcc']}\n")


def save_metrics_bn_to_file(metrics: dict, filename: str) -> None:
    """
    Save the metrics of the BN to a text file.
    :param metrics:
    :param filename:
    :return:
    """
    # class_labels = ['Class 0', 'Class 1']
    # formatted_cm = format_confusion_matrix(metrics['confusion_matrix'], class_labels)
    logging.info(f"Metrics: {metrics}")

    # check if the file exists otherwise create it
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write("Metrics from Bayesian Network:\n")
            f.write(f"Accuracy: {metrics['accuracy']}\n")
            f.write(f"Precision: {metrics['precision']}\n")
            f.write(f"Sensitivity: {metrics['sensitivity']}\n")
            f.write(f"Specificity: {metrics['specificity']}\n")
            f.write(f"F1 Score: {metrics['f1_score']}\n")
            f.write(f"MCC: {metrics['mcc']}\n")
            f.write(f"Log Likelihood: {metrics['log_likelihood']}\n")
            f.write(f"BIC: {metrics['bic_score']}\n")


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
