
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.data_ingestion import load_data
from src.data_validation import validate_data

from src.EDA import (
    correlation_analysis,
    distribution_analysis
)

from src.model_training import get_available_models
from src.pipeline import run_automl
from src.model_saving import load_model


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AutoML Platform",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AutoML Platform")

st.markdown("""
### Automated Machine Learning Platform

Upload your dataset, explore your data, train machine learning
models, evaluate performance, tune hyperparameters,
and save your best model.
""")

st.divider()


# ==========================================
# 2. DATA INGESTION
# ==========================================

st.header("1. Data Ingestion")

uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df = load_data(uploaded_file)

        st.session_state["df"] = df
        st.session_state["uploaded_file"] = uploaded_file

        st.success("Dataset uploaded successfully!")

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(),
            use_container_width=True
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

    except Exception as e:
        st.error(f"Data loading error: {e}")


# ==========================================
# 3. DATA VALIDATION
# ==========================================

if "df" in st.session_state:

    st.divider()

    st.header("2. Data Validation")

    if st.button("Validate Dataset"):

        try:
            validation_report = validate_data(
                st.session_state["df"]
            )

            st.session_state["validation_report"] = (
                validation_report
            )

            st.success("Dataset validation completed!")

            st.subheader("Validation Report")

            st.json(validation_report)

        except Exception as e:
            st.error(f"Validation error: {e}")


# ==========================================
# 4. EXPLORATORY DATA ANALYSIS
# ==========================================

if "df" in st.session_state:

    st.divider()

    st.header("3. Exploratory Data Analysis")

    df = st.session_state["df"]

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    # Correlation Analysis
    st.subheader("Correlation Analysis")

    if st.button("Generate Correlation Heatmap"):

        try:
            correlation, correlation_fig = (
                correlation_analysis(df)
            )

            if correlation_fig is not None:

                st.pyplot(correlation_fig)
                plt.close(correlation_fig)

                st.subheader("Correlation Matrix")

                st.dataframe(
                    correlation,
                    use_container_width=True
                )

            else:
                st.warning(
                    "At least two numerical columns are required."
                )

        except Exception as e:
            st.error(f"Correlation error: {e}")

    # Distribution Analysis
    st.subheader("Distribution Analysis")

    if numeric_columns:

        selected_column = st.selectbox(
            "Select a numerical column",
            numeric_columns
        )

        if st.button("Generate Distribution Plot"):

            try:
                distribution_fig = distribution_analysis(
                    df,
                    selected_column
                )

                if distribution_fig is not None:

                    st.pyplot(distribution_fig)
                    plt.close(distribution_fig)

                else:
                    st.warning(
                        "No available values for this column."
                    )

            except Exception as e:
                st.error(f"Distribution error: {e}")

    else:
        st.info("No numerical columns found.")


# ==========================================
# 5. AUTOML CONFIGURATION
# ==========================================

if "df" in st.session_state:

    st.divider()

    st.header("4. AutoML Configuration")

    df = st.session_state["df"]

    target_column = st.selectbox(
        "Select Target Column",
        df.columns.tolist()
    )

    task_type = st.selectbox(
        "Select Machine Learning Task",
        ["regression", "classification"]
    )

    available_models = get_available_models(task_type)

    selected_models = st.multiselect(
        "Select Models",
        list(available_models.keys()),
        default=list(available_models.keys())
    )

    test_size = st.slider(
        "Test Data Size",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05
    )

    # Hyperparameter Tuning
    st.subheader("Hyperparameter Tuning")

    enable_tuning = st.checkbox(
        "Enable Hyperparameter Tuning",
        value=False
    )

    tuning_method = st.selectbox(
        "Tuning Method",
        ["grid", "random"],
        disabled=not enable_tuning
    )

    cv = st.slider(
        "Cross Validation Folds",
        min_value=2,
        max_value=10,
        value=5,
        disabled=not enable_tuning
    )

    n_iter = st.number_input(
        "Random Search Iterations",
        min_value=1,
        max_value=100,
        value=10,
        disabled=not enable_tuning
    )


# ==========================================
# 6. RUN AUTOML PIPELINE
# ==========================================

if "df" in st.session_state:

    st.divider()

    st.header("5. Run AutoML")

    if st.button(
        "🚀 Start AutoML",
        type="primary",
        use_container_width=True
    ):

        if not selected_models:

            st.warning("Please select at least one model.")

        else:

            try:

                with st.spinner(
                    "Running AutoML Pipeline..."
                ):

                    st.session_state.pop(
                        "automl_results",
                        None
                    )

                    results = run_automl(
                        file_path=st.session_state["uploaded_file"],
                        target_column=target_column,
                        task_type=task_type,
                        selected_models=selected_models,
                        test_size=test_size,
                        enable_tuning=enable_tuning,
                        tuning_method=tuning_method,
                        cv=cv,
                        n_iter=int(n_iter)
                    )

                    st.session_state["automl_results"] = results

                st.success(
                    "AutoML Pipeline Completed Successfully!"
                )

            except Exception as e:

                st.error(
                    f"AutoML Pipeline Error: {e}"
                )


# ==========================================
# 7. MODEL EVALUATION
# ==========================================

if "automl_results" in st.session_state:

    results = st.session_state["automl_results"]

    st.divider()

    st.header("6. Model Evaluation")

    evaluation_results = results["evaluation"]

    display_results = {}

    for model_name, metrics in evaluation_results.items():

        display_results[model_name] = {
            key: value
            for key, value in metrics.items()
            if key != "Confusion Matrix"
        }

    results_df = pd.DataFrame.from_dict(
        display_results,
        orient="index"
    )

    st.subheader("Model Comparison")

    st.dataframe(
        results_df,
        use_container_width=True
    )

    # Best Model
    st.subheader("Best Model")

    best_model_name = results["best_model_name"]

    st.success(
        f"Best Model: {best_model_name}"
    )

    best_metrics = results["best_model_results"]

    metric_values = {
        key: value
        for key, value in best_metrics.items()
        if key != "Confusion Matrix"
    }

    metric_columns = st.columns(
        max(1, len(metric_values))
    )

    for index, (metric_name, metric_value) in enumerate(
        metric_values.items()
    ):

        metric_columns[index].metric(
            metric_name,
            f"{metric_value:.4f}"
        )

    # Confusion Matrix
    if "Confusion Matrix" in best_metrics:

        st.subheader("Confusion Matrix")

        confusion_df = pd.DataFrame(
            best_metrics["Confusion Matrix"]
        )

        st.dataframe(
            confusion_df,
            use_container_width=True
        )


# ==========================================
# 8. HYPERPARAMETER TUNING RESULTS
# ==========================================

if "automl_results" in st.session_state:

    results = st.session_state["automl_results"]

    st.divider()

    st.header("7. Hyperparameter Tuning")

    tuning_results = results["tuning"]

    if tuning_results is not None:

        st.success("Tuning completed successfully!")

        col1, col2 = st.columns(2)

        col1.metric(
            "Best CV Score",
            f"{tuning_results['best_score']:.4f}"
        )

        col2.metric(
            "Number of Best Parameters",
            len(tuning_results["best_params"])
        )

        st.subheader("Best Parameters")

        st.json(
            tuning_results["best_params"]
        )

    else:

        st.info(
            "Hyperparameter tuning was not enabled."
        )


# ==========================================
# 9. MODEL SAVING
# ==========================================

if "automl_results" in st.session_state:

    results = st.session_state["automl_results"]

    st.divider()

    st.header("8. Model Saving")

    st.success("Best model saved successfully!")

    st.write(
        "**Model Path:**",
        str(results["model_path"])
    )

    st.write(
        "**Metadata Path:**",
        str(results["metadata_path"])
    )


# ==========================================
# 10. EXPERIMENT TRACKING
# ==========================================

if "automl_results" in st.session_state:

    st.divider()

    st.header("9. Experiment Tracking")

    st.info(
        "Experiments and numerical evaluation metrics "
        "are automatically tracked using MLflow."
    )

    st.markdown("""
    Open your terminal in the project directory and run:

    ```bash
    mlflow ui
    ```

    Then visit:

    http://localhost:5000
    """)


# ==========================================
# 11. LOAD SAVED MODEL
# ==========================================

st.divider()

st.header("10. Load Saved Model")

model_path_input = st.text_input(
    "Enter Model File Path",
    placeholder="models/random_forest.joblib"
)

if st.button("Load Model"):

    if model_path_input:

        try:

            loaded_model = load_model(
                model_path_input
            )

            st.session_state["loaded_model"] = loaded_model

            st.success("Model loaded successfully!")

            st.write(
                "Loaded Object:",
                type(loaded_model).__name__
            )

            if isinstance(loaded_model, dict):

                st.write(
                    "Model Bundle Components:",
                    list(loaded_model.keys())
                )

        except Exception as e:

            st.error(
                f"Model loading error: {e}"
            )

    else:

        st.warning("Please enter a model path.")


# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <div style="text-align:center">

    **AutoML Platform**

    Automated Machine Learning | Data Science | Python

    </div>
    """,
    unsafe_allow_html=True
)
