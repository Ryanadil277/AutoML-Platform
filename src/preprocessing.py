import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def split_features_target(df, target_column):
    if target_column not in df.columns:
        raise ValueError("Target column not found.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def create_preprocessor(X):
    numerical_columns = X.select_dtypes(include="number").columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    return preprocessor


def prepare_data(df, target_column, test_size=0.2, random_state=42):
    X, y = split_features_target(df, target_column)

    if y.isnull().any():
        raise ValueError("Target column contains missing values.")

    if y.nunique() < 2:
        raise ValueError("Target column must contain at least two classes or values.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    preprocessor = create_preprocessor(X_train)

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test, preprocessor