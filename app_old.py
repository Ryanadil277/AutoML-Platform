import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from src.data_ingestion import load_data
from src.data_validation import validate_data
from src.EDA import correlation_analysis, distribution_analysis
from src.preprocessing import prepare_data
from src.model_training import get_available_models, train_models
from src.evaluation import evaluate_models,get_best_model
from src.tuning import ( tune_model,get_default_scoring)
from src.model_saving import save_model, save_model_metadata
from src.tuning import get_parameter_grid
# ==========================================
# 1. Page Configuration
# ==========================================

st.set_page_config(
    page_title="AutoML Platform",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AutoML Platform")
st.write(
    "An automated machine learning platform "
    "for data analysis and model training."
)

st.divider()


# ==========================================
# 2. Data Upload
# ==========================================

st.header("1. Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        # Load Dataset
        df = load_data(uploaded_file)

        st.success("Dataset uploaded successfully!")

        # ==========================================
        # 3. Dataset Preview
        # ==========================================

        st.header("2. Dataset Preview")

        st.dataframe(df, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Number of Rows", df.shape[0])

        with col2:
            st.metric("Number of Columns", df.shape[1])

        st.divider()

        # ==========================================
        # 4. Data Validation
        # ==========================================

        st.header("3. Data Validation")

        validation_report = validate_data(df)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Rows", validation_report["rows"])

        with col2:
            st.metric("Columns", validation_report["columns"])

        with col3:
            st.metric(
                "Duplicate Rows",
                validation_report["duplicate_rows"]
            )

        st.subheader("Missing Values")

        if validation_report["missing_values"]:
            st.json(validation_report["missing_values"])
        else:
            st.success("No missing values found.")

        st.subheader("Constant Columns")

        if validation_report["constant_columns"]:
            st.write(validation_report["constant_columns"])
        else:
            st.success("No constant columns found.")

        st.subheader("Data Types")
        st.json(validation_report["data_types"])

        st.divider()

        # ==========================================
        # 5. Exploratory Data Analysis
        # ==========================================

        st.header("4. Exploratory Data Analysis")

        tab1, tab2 = st.tabs([
            "Correlation",
            "Distribution"
        ])

        with tab1:

            st.subheader("Correlation Heatmap")

            correlation, correlation_fig = correlation_analysis(df)

            if correlation_fig is not None:
                st.pyplot(correlation_fig)
                plt.close(correlation_fig)
            else:
                st.info(
                    "At least two numerical columns "
                    "are required for correlation analysis."
                )

        with tab2:

            st.subheader("Feature Distribution")

            numerical_columns = (
                df.select_dtypes(include="number").columns.tolist()
            )

            if numerical_columns:

                selected_column = st.selectbox(
                    "Select a numerical feature",
                    numerical_columns
                )

                distribution_fig = distribution_analysis(
                    df,
                    selected_column
                )

                if distribution_fig is not None:
                    st.pyplot(distribution_fig)
                    plt.close(distribution_fig)
                else:
                    st.info("No valid values available.")
            else:
                st.info("No numerical columns found.")

        st.divider()

        # ==========================================
        # 6. Data Preprocessing
        # ==========================================
        # ==========================================
# 5. DATA PREPROCESSING & TARGET SELECTION
# ==========================================

st.divider()
st.header("5. Data Preprocessing")

st.write(
    "Select the target column that you want the "
    "Machine Learning model to predict."
)

# Check available columns
if df.shape[1] < 2:
    st.warning("Dataset must contain at least two columns.")

else:

    # 1. Target Selection
    target_column = st.selectbox(
        "Select Target Column",
        options=df.columns.tolist(),
        help="Choose the column that represents the output you want to predict."
    )

    target_series = df[target_column]

    # 2. Target Explanation
    st.subheader("Target Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Data Type",
            str(target_series.dtype)
        )

    with col2:
        st.metric(
            "Unique Values",
            int(target_series.nunique())
        )

    with col3:
        st.metric(
            "Missing Values",
            int(target_series.isnull().sum())
        )

    # 3. Explain the selected target
    st.info(
        f"""
        Selected Target: {target_column}

        This is the column that your model will learn to predict.

        All other columns will be used as input features (X).
        """
    )

    # 4. Target Preview
    with st.expander("View Target Values"):
        st.dataframe(
            target_series.head(10).to_frame(),
            use_container_width=True
        )

    # 5. Target Suitability Information
    if target_series.isnull().all():
        st.error("The selected target contains no valid values.")

    elif target_series.nunique() < 2:
        st.error(
            "The target must contain at least two different values."
        )

    elif pd.api.types.is_numeric_dtype(target_series):
        st.success(
            "The target is numerical. It may be suitable for "
            "Regression or Classification, depending on your goal."
        )

    else:
        st.success(
            "The target is categorical or text-based. "
            "It may be suitable for Classification."
        )

    # 6. Show Features
    st.subheader("Input Features")

    feature_columns = [
        column for column in df.columns
        if column != target_column
    ]

    st.write(
        f"Number of Features: {len(feature_columns)}"
    )

    st.write("Features used for training:")

    st.write(feature_columns)

    # 7. Test Size Selection
    st.subheader("Train-Test Split")

    test_size = st.slider(
        "Test Data Percentage (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
        help="Percentage of data reserved for testing."
    )

    st.write(
        f"Training Data: {100 - test_size}%"
    )

    st.write(
        f"Testing Data: {test_size}%"
    )

    # 8. Prepare Data
    if st.button("Prepare Data"):

        try:

            if target_series.isnull().all():
                raise ValueError(
                    "Target column contains only missing values."
                )

            if target_series.isnull().any():
                raise ValueError(
                    "Target column contains missing values. "
                    "Please clean the target first."
                )

            if target_series.nunique() < 2:
                raise ValueError(
                    "Target must contain at least two unique values."
                )

            with st.spinner("Preparing Dataset..."):

                (
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    preprocessor
                ) = prepare_data(
                    df=df,
                    target_column=target_column,
                    test_size=test_size / 100,
                    random_state=42
                )
# Save results in session state
            st.session_state["X_train"] = X_train
            st.session_state["X_test"] = X_test
            st.session_state["y_train"] = y_train
            st.session_state["y_test"] = y_test
            st.session_state["preprocessor"] = preprocessor
            st.session_state["target_column"] = target_column

            # Clear old results
            keys_to_clear = [
                "trained_models",
                "evaluation_results",
                "best_model_name",
                "tuning_results",
                "tuned_model",
                "saved_model_path",
                "saved_metadata_path"
            ]

            for key in keys_to_clear:
                st.session_state.pop(key, None)

            st.success(
                "Data preprocessing completed successfully!"
            )

            # 9. Display Dataset Split
            st.subheader("Prepared Data Summary")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Training Samples",
                    X_train.shape[0]
                )

            with col2:
                st.metric(
                    "Testing Samples",
                    X_test.shape[0]
                )

            st.write(
                f"Number of Processed Features: {X_train.shape[1]}"
            )

            st.write(
                f"Selected Target: {target_column}"
            )

        except Exception as e:
            st.error(f"Preprocessing Error: {e}")

    # 10. Show preparation status
    if "X_train" in st.session_state:
        st.success("Dataset is ready for Model Training.")
    

        # ==========================================
        # 7. Model Training
        # ==========================================

        st.header("6. Model Training")

        task_type = st.selectbox(
            "Select Task Type",
            ["classification", "regression"]
        )

        available_models = get_available_models(task_type)

        selected_models = st.multiselect(
            "Select Models to Train",
            options=list(available_models.keys()),
            default=list(available_models.keys())
        )

        if st.button("Train Models"):

            if "X_train" not in st.session_state:

                st.warning(
                    "Please prepare the data first."
                )

            elif (
                st.session_state.get("target_column")
                != target_column
            ):

                st.warning(
                    "Please prepare the data again "
                    "after changing the target column."
                )

            elif not selected_models:

                st.warning(
                    "Please select at least one model."
                )

            else:

                try:

                    with st.spinner("Training models..."):

                        trained_models = train_models(
                            st.session_state["X_train"],
                            st.session_state["y_train"],
                            task_type,
                            selected_models
                        )

                    st.session_state["trained_models"] = trained_models
                    st.session_state["task_type"] = task_type

                    st.success("Models trained successfully!")

                    st.subheader("Trained Models")

                    for model_name in trained_models:
                        st.write(f"✅ {model_name}")

                except Exception as e:st.error(f"Training Error: {e}")

    except Exception as e:

        st.error(f"Error: {e}")

else:

    st.info("Please upload a CSV dataset to get started.")
    # ==========================================
# 8. Model Evaluation
# ==========================================

st.divider()
st.header("7. Model Evaluation")

if st.button("Evaluate Models"):

    if "trained_models" not in st.session_state:

        st.warning("Please train the models first.")

    else:

        try:

            evaluation_results = evaluate_models(
                st.session_state["trained_models"],
                st.session_state["X_test"],
                st.session_state["y_test"],
                st.session_state["task_type"]
            )

            st.session_state["evaluation_results"] = evaluation_results

            best_model_name = get_best_model(
                evaluation_results,
                st.session_state["task_type"]
            )

            st.session_state["best_model_name"] = best_model_name

            st.success("Model evaluation completed successfully!")

        except Exception as e:

            st.error(f"Evaluation Error: {e}")


# Display Evaluation Results

if "evaluation_results" in st.session_state:

    evaluation_results = st.session_state["evaluation_results"]
    task_type = st.session_state["task_type"]

    st.subheader("Models Performance Comparison")

    # Display metrics for all models
    results_df = pd.DataFrame(evaluation_results).T

    if "Confusion Matrix" in results_df.columns:
        results_df = results_df.drop(columns=["Confusion Matrix"])

    st.dataframe(
        results_df.style.format("{:.4f}"),
        use_container_width=True
    )

    # Best Model
    best_model_name = st.session_state["best_model_name"]

    st.subheader("Best Model")

    st.success(f"Best Model: {best_model_name}")

    st.write(
        "Performance Metrics:",
        evaluation_results[best_model_name]
    )

    # Confusion Matrix for Classification
    if task_type == "classification":

        st.subheader("Confusion Matrix")

        confusion = evaluation_results[best_model_name][
            "Confusion Matrix"
        ]

        confusion_df = pd.DataFrame(confusion)

        st.dataframe(confusion_df)

    # Regression Results
    elif task_type == "regression":

        st.subheader("Regression Performance")

        best_metrics = evaluation_results[best_model_name]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("MAE", f"{best_metrics['MAE']:.4f}")

        with col2:
            st.metric("RMSE", f"{best_metrics['RMSE']:.4f}")

        with col3:
            st.metric("R2 Score", f"{best_metrics['R2 Score']:.4f}")

            # ==========================================
# 9. Hyperparameter Tuning
# ==========================================

st.divider()
st.header("8. Hyperparameter Tuning")

if "best_model_name" in st.session_state:

    best_model_name = st.session_state["best_model_name"]
    task_type = st.session_state["task_type"]

    st.write(f"Best Model: {best_model_name}")

    tuning_method = st.selectbox(
        "Select Tuning Method",
        ["grid", "random"]
    )

    cv = st.slider(
        "Cross Validation (CV)",
        min_value=2,
        max_value=10,
        value=3
    )

    n_iter = 10

    if tuning_method == "random":
        n_iter = st.slider(
            "Number of Iterations",
            min_value=5,
            max_value=50,
            value=10
        )

    if st.button("Start Hyperparameter Tuning"):

        try:
            param_grid = get_parameter_grid(
                best_model_name,
                task_type
            )

            best_model = st.session_state["trained_models"][
                best_model_name
            ]

            with st.spinner("Tuning model..."):

                tuning_results = tune_model(
                    model=best_model,
                    param_grid=param_grid,
                    X_train=st.session_state["X_train"],
                    y_train=st.session_state["y_train"],
                    scoring=get_default_scoring(task_type),
                    cv=cv,
                    method=tuning_method,
                    n_iter=n_iter
                )

            st.session_state["tuning_results"] = tuning_results
            st.session_state["tuned_model"] = tuning_results["best_model"]

            st.success("Hyperparameter tuning completed!")

            st.subheader("Best Parameters")
            st.json(tuning_results["best_params"])

            st.metric(
                "Best Cross Validation Score",
                f"{tuning_results['best_score']:.4f}"
            )

        except Exception as e:
            st.error(f"Tuning Error: {e}")

else:
    st.info("Please train and evaluate models first.")
    # ==========================================
# 10. Model Saving
# ==========================================

st.divider()
st.header("9. Save Best Model")

if "best_model_name" in st.session_state:

    st.write(
        f"Best Model: {st.session_state['best_model_name']}"
    )

    if st.button("Save Best Model"):

        try:
            best_model_name = st.session_state["best_model_name"]
            task_type = st.session_state["task_type"]

            # Use tuned model if available
            if "tuned_model" in st.session_state:
                final_model = st.session_state["tuned_model"]
                best_params = st.session_state[
                    "tuning_results"
                ]["best_params"]
            else:
                final_model = st.session_state["trained_models"][
                    best_model_name
                ]
                best_params = {}

            # Evaluate the final model on test data
            final_metrics = evaluate_models(
                {best_model_name: final_model},
                st.session_state["X_test"],
                st.session_state["y_test"],
                task_type
            )[best_model_name]

            # Save model with preprocessor
            model_bundle = {
                "model": final_model,
                "preprocessor": st.session_state["preprocessor"],
                "target_column": st.session_state["target_column"],
                "task_type": task_type
            }

            model_path = save_model(
                model_bundle,
                best_model_name,
                save_dir="models"
            )

            # Save metadata
            metadata_path = save_model_metadata(
                model_name=best_model_name,
                task_type=task_type,
                metrics=final_metrics,
                best_params=best_params,
                save_dir="models"
            )

            st.session_state["saved_model_path"] = str(model_path)
            st.session_state["saved_metadata_path"] = str(metadata_path)

            st.success("Model saved successfully!")

            st.write("Model File:")
            st.code(str(model_path))

            st.write("Metadata File:")
            st.code(str(metadata_path))

            st.write("Final Model Performance:")
            st.json(final_metrics)

        except Exception as e:
            st.error(f"Model Saving Error: {e}")

else:
    st.info("Please train and evaluate models first.")
    