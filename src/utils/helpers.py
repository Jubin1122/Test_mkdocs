# src/utils/helpers.py

def ensure_directory(path: str) -> None:
    """
    Create a directory if it doesn’t already exist.

    Parameters
    ----------
    path : str
        Filesystem path to create.

    Returns
    -------
    None
    """
    import os
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> dict:
    """
    Load a JSON file into a Python dict.

    Parameters
    ----------
    path : str
        Path to a `.json` file.

    Returns
    -------
    dict
        Parsed JSON contents.
    """
    import json
    with open(path, "r") as f:
        return json.load(f)
