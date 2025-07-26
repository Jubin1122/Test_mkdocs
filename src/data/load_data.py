# src/data/load_data.py

import pandas as pd
from configs import settings

def load_data() -> pd.DataFrame:
    """
    Load the full dataset into a pandas DataFrame.

    Reads CSV data from the path specified in `configs.settings.DATA_PATH`.

    Returns
    -------
    pd.DataFrame
        The raw, unprocessed dataset.
    """
    return pd.read_csv(settings.DATA_PATH)


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split a DataFrame into features and target.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset containing a column named 'target'.

    Returns
    -------
    X : pd.DataFrame
        All columns except 'target'.
    y : pd.Series
        The 'target' column.
    """
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y
