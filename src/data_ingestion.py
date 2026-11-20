import pandas as pd
from pathlib import Path


def load_data(file):
    """
    Load a CSV dataset from a file path
    or an uploaded Streamlit file.
    """

    if isinstance(file, (str, Path)):

        path = Path(file)

        if not path.exists():
            raise FileNotFoundError("Dataset file not found.")

        if path.suffix.lower() != ".csv":
            raise ValueError("Only CSV files are supported.")

        file = path

    else:

        if not hasattr(file, "name"):
            raise ValueError("Invalid uploaded file.")

        if not file.name.lower().endswith(".csv"):
            raise ValueError("Only CSV files are supported.")

        file.seek(0)

    try:
        df = pd.read_csv(file)

    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded CSV file is empty.")

    except pd.errors.ParserError:
        raise ValueError("Could not read the CSV file.")

    if df.empty:
        raise ValueError("The dataset contains no data.")

    return df