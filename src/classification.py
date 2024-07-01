import numpy as np
import pandas as pd


def majority_vote(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform majority vote on the predictions dataframe.
    :param df: pd.DataFrame
    :return: pd.DataFrame
    """
    return df.mode(axis=1)[0].astype(int)
