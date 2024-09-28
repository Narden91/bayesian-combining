import pandas as pd
import numpy as np
import logging
from pathlib import Path


def read_and_process_file(file_path: Path, task_list: list, suffix: str) -> pd.DataFrame:
    """
    Reads a CSV file, filters columns based on task_list, and renames columns with a suffix.

    :param file_path: Path of the file to read.
    :param task_list: List of columns to retain.
    :param suffix: Suffix to add to the column names.
    :return: Processed DataFrame.
    """
    if file_path.exists():
        df = pd.read_csv(file_path)
        df = df[task_list]
        df.columns = [f"{col}_{suffix}" for col in df.columns]
        # logging.info(f"Processed DataFrame: {df.to_string()}")
        return df
    else:
        logging.error(f"File not found: {file_path}")
        return pd.DataFrame()


def process_data(base_path: Path, run: int, data_type: str, element_list: list, task_list: list,
                 clf_name: str) -> tuple:
    """
    Process data for a specific type (ML or DL) and return concatenated DataFrames.

    :param base_path: Base path of the data.
    :param run: Run number.
    :param data_type: Data type (e.g., ML, DL).
    :param element_list: List of elements (e.g., models or techniques).
    :param task_list: List of tasks to retain.
    :param clf_name: Classifier name.
    :return: Tuple of concatenated DataFrames for test, test probabilities, train, and train probabilities.
    """
    test_df_combined = pd.DataFrame()
    test_proba_df_combined = pd.DataFrame()
    train_df_combined = pd.DataFrame()
    train_proba_df_combined = pd.DataFrame()

    for idx, elem in enumerate(element_list):
        data_path = base_path / data_type / elem / clf_name / "Bayesian_stacking_clf" / f"run_{run}" / "First_level_data"
        # logging.info(f"Data path: {data_path}")

        # File paths
        test_data_path = data_path / "Test_data.csv"
        test_proba_path = data_path / "Test_data_probabilities.csv"
        train_data_path = data_path / "Trainings_data.csv"
        train_proba_path = data_path / "Trainings_data_proba.csv"

        if idx == 0:
            dummy_df_test = pd.read_csv(test_data_path)
            label = dummy_df_test["Label"]
            dummy_df_train = pd.read_csv(train_data_path)
            label_train = dummy_df_train["Label"]

        # Read and process each file
        test_df = read_and_process_file(test_data_path, task_list, elem)
        test_proba_df = read_and_process_file(test_proba_path, task_list, elem)
        train_df = read_and_process_file(train_data_path, task_list, elem)
        train_proba_df = read_and_process_file(train_proba_path, task_list, elem)

        # Concatenate dataframes
        test_df_combined = pd.concat([test_df_combined, test_df], axis=1)
        test_proba_df_combined = pd.concat([test_proba_df_combined, test_proba_df], axis=1)
        train_df_combined = pd.concat([train_df_combined, train_df], axis=1)
        train_proba_df_combined = pd.concat([train_proba_df_combined, train_proba_df], axis=1)

    # Concatenate the label to the test_df_ml if it exists
    test_df_combined = pd.concat([test_df_combined, label], axis=1)
    test_proba_df_combined = pd.concat([test_proba_df_combined, label], axis=1)
    train_df_combined = pd.concat([train_df_combined, label_train], axis=1)
    train_proba_df_combined = pd.concat([train_proba_df_combined, label_train], axis=1)

    return test_df_combined, test_proba_df_combined, train_df_combined, train_proba_df_combined


def get_predictions_df(path: str, run: int, task_list: list, clf_name: str) -> pd.DataFrame:
    """
    Read and process predictions data from the given path.

    :param path: Base path of the data.
    :param run: Run number.
    :param task_list: List of tasks to retain.
    :param clf_name: Classifier name.
    :return: Combined DataFrame of test data and labels.
    """
    base_path = Path(path)
    logging.info(f"Output path: {base_path}")

    ml_list = ["All", "InAir", "OnPaper", "InAirOnPaper"]
    dl_list = ["ConvNeXtSmall", "EfficientNetV2S", "InceptionResNetV2", "InceptionV3"]

    # Process ML data
    test_df_ml, test_proba_df_ml, train_df_ml, train_proba_df_ml = process_data(
        base_path, run, "ML", ml_list, task_list, clf_name)

    # Drop the label from the ml dataframes
    test_df_ml = test_df_ml.drop(columns=["Label"])
    test_proba_df_ml = test_proba_df_ml.drop(columns=["Label"])
    train_df_ml = train_df_ml.drop(columns=["Label"])
    train_proba_df_ml = train_proba_df_ml.drop(columns=["Label"])

    # Process DL data
    test_df_dl, test_proba_df_dl, train_df_dl, train_proba_df_dl = process_data(
        base_path, run, "DL", dl_list, task_list, clf_name)

    # Combine ML and DL test data
    combined_test_df = pd.concat([test_df_ml, test_df_dl], axis=1)
    combined_test_proba_df = pd.concat([test_proba_df_ml, test_proba_df_dl], axis=1)
    combined_train_df = pd.concat([train_df_ml, train_df_dl], axis=1)
    combined_train_proba_df = pd.concat([train_proba_df_ml, train_proba_df_dl], axis=1)

    # logging.info(f"Combined Test DataFrame: {combined_test_df.to_string()}")
    # logging.info(f"Combined Test Probabilities DataFrame: {combined_test_proba_df.to_string()}")
    # logging.info(f"Combined Train DataFrame: {combined_train_df.to_string()}")
    # logging.info(f"Combined Train Probabilities DataFrame: {combined_train_proba_df.to_string()}")

    return combined_test_df, combined_test_proba_df, combined_train_df, combined_train_proba_df


