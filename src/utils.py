import os
from pathlib import Path

import pandas as pd
import logging


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
