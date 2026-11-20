import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. Correlation Analysis
# ==========================================

def correlation_analysis(df):

    numerical_df = df.select_dtypes(include="number")

    if numerical_df.shape[1] < 2:
        return None, None

    correlation = numerical_df.corr()

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        center=0,
        ax=ax
    )

    ax.set_title(
        "Features Correlation",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()

    return correlation, fig


# ==========================================
# 2. Distribution Analysis
# ==========================================

def distribution_analysis(df, column):

    if column not in df.columns:
        raise ValueError("Column not found in dataset.")

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError("Distribution requires a numerical column.")

    values = df[column].dropna()

    if values.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.histplot(
        values,
        bins=20,
        kde=len(values) > 1,
        ax=ax
    )

    ax.set_title(
        f"Distribution of {column}",
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

    plt.tight_layout()

    return fig