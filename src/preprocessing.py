import os
import logging
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def data_cleaning_ml(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Clean the data for the Machine Learning task.
    :param df: pd.DataFrame
    :param verbose: bool
    :return: pd.DataFrame
    """
    # Drop columns with all NaN values or Zeros
    df.dropna(axis=1, how='all', inplace=True)
    task_df = df.loc[:, (df != 0).any(axis=0)]

    categorical_columns = ['Sex', 'Work']
    label_column = 'Label'

    # Separate the categorical columns and the rest of the dataframe
    df_categorical = task_df[categorical_columns]
    df_rest = task_df.drop(columns=categorical_columns)

    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_categorical = encoder.fit_transform(df_categorical)
    encoded_categorical_df = pd.DataFrame(encoded_categorical,
                                          columns=encoder.get_feature_names_out(categorical_columns))

    # remove white spaces from column names
    encoded_categorical_df.columns = encoded_categorical_df.columns.str.replace(' ', '')
    df_rest.drop(columns=label_column, inplace=True)

    task_df.loc[:, 'Label'] = task_df['Label'].str.strip()
    task_df.loc[:, 'Label'] = task_df['Label'].map({label: key for key, label in enumerate(['Sano', 'Malato'])})

    # Combine the encoded columns with the rest of the dataframe
    df_encoded = pd.concat([df_rest, encoded_categorical_df, task_df[label_column]], axis=1)
    df_encoded['Id'] = df_encoded['Id'].astype(int)
    df_encoded['Sex_Maschile'] = df_encoded['Sex_Maschile'].astype(int)
    df_encoded['Work_Manuale'] = df_encoded['Work_Manuale'].astype(int)

    print(df_encoded.head().to_string()) if verbose else None

    return df_encoded


def data_split(df: pd.DataFrame, target: str = 'Label', test_size: float = 0.2,
               seed: int = 42, verbose: bool = False) -> tuple[pd.DataFrame]:
    """
    Split the data into train and test sets.
    :param df: pd.DataFrame
    :param target: str (default: "Label")
    :param test_size: float
    :param seed: int
    :param verbose: bool
    :return: tuple[pd.DataFrame]
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[target])

    return train_df, test_df


def data_scaling(df: pd.DataFrame, id_column: str = 'Id',
                 scaler_type: str = "Standard", verbose: bool = False) -> pd.DataFrame:
    """
    Scale the data
    :param df: pd.DataFrame
    :param id_column: str
    :param scaler_type: str (Standard, Robust)
    :param verbose: bool
    :return: pd.DataFrame
    """
    scaler = StandardScaler() if scaler_type == "Standard" else RobustScaler()

    columns_to_scale = df.columns.drop(id_column)
    scaled_columns = scaler.fit_transform(df[columns_to_scale])

    scaled_df = pd.DataFrame(scaled_columns, columns=columns_to_scale, index=df.index)
    scaled_df = pd.concat([df[[id_column]], scaled_df], axis=1)

    logging.info(f"Data Scaled: \n {scaled_df}") if verbose else None

    return scaled_df, scaler


def apply_scaling(df: pd.DataFrame, scaler, id_column: str = 'Id', verbose: bool = False) -> pd.DataFrame:
    """
    Apply the scaling to the data
    :param df: pd.DataFrame
    :param scaler: StandardScaler
    :param id_column: str
    :param verbose: bool
    :return: pd.DataFrame
    """
    columns_to_scale = df.columns.drop(id_column)
    scaled_columns = scaler.transform(df[columns_to_scale])

    scaled_df = pd.DataFrame(scaled_columns, columns=columns_to_scale, index=df.index)
    scaled_df = pd.concat([df[[id_column]], scaled_df], axis=1)

    logging.info(f"Data Scaled: \n {scaled_df}") if verbose else None

    return scaled_df