import pandas as pd


def validate_data(df):
    """
    Validate the uploaded dataset and return a validation report.
    """

    # 1. Check if the input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    # 2. Check if the dataset is empty
    if df.empty:
        raise ValueError("The dataset is empty.")

    # 3. Check for duplicate column names
    if df.columns.duplicated().any():
        raise ValueError("The dataset contains duplicate column names.")

    # 4. Check for missing values
    missing_values = {
        column: int(count)
        for column, count in df.isnull().sum().items()
        if count > 0
    }

    # 5. Check for duplicate rows
    duplicate_rows = int(df.duplicated().sum())

    # 6. Detect constant columns
    constant_columns = [
        column
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]

    # 7. Get data types
    data_types = {
        column: str(dtype)
        for column, dtype in df.dtypes.items()
    }

    # 8. Get dataset shape
    rows, columns = df.shape

    # 9. Build validation report
    validation_report = {
        "rows": rows,
        "columns": columns,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "data_types": data_types
    }

    return validation_report