
from src.data_ingestion import load_data
from src.data_validation import validate_data

from src.EDA import (
    correlation_analysis,
    distribution_analysis
)

from src.preprocessing import prepare_data
from src.model_training import train_models

from src.evaluation import (
    evaluate_models,
    get_best_model
)

from src.tuning import (
    tune_model,
    get_parameter_grid,
    get_default_scoring
)

from src.model_saving import (
    save_model,
    save_model_metadata
)

from src.experiment_tracking import log_experiment


def run_automl(
    file_path,
    target_column,
    task_type,
    selected_models=None,
    test_size=0.2,
    enable_tuning=False,
    tuning_method="grid",
    cv=5,
    n_iter=10,
    save_dir="models"
):

    # 1. Data Ingestion
    df = load_data(file_path)

    # 2. Data Validation
    validation_report = validate_data(df)

    if target_column not in df.columns:
        raise ValueError("Target column not found.")

    # 3. Exploratory Data Analysis
    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    correlation, correlation_fig = correlation_analysis(df)

    distribution_fig = (
        distribution_analysis(df, numeric_columns[0])
        if numeric_columns else None
    )

    eda_results = {
        "correlation": correlation,
        "correlation_fig": correlation_fig,
        "distribution_fig": distribution_fig
    }

    # 4. Data Preprocessing
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(
        df,
        target_column,
        test_size=test_size
    )

    # 5. Model Training
    trained_models = train_models(
        X_train,
        y_train,
        task_type,
        selected_models
    )

    if not trained_models:
        raise ValueError("No models were trained.")

    # 6. Model Evaluation
    evaluation_results = evaluate_models(
        trained_models,
        X_test,
        y_test,
        task_type
    )

    # 7. Select Best Model
    best_model_name = get_best_model(
        evaluation_results,
        task_type
    )

    best_model = trained_models[best_model_name]

    # 8. Hyperparameter Tuning
    tuning_results = None

    if enable_tuning:

        param_grid = get_parameter_grid(
            best_model_name,
            task_type
        )

        tuning_results = tune_model(
            model=best_model,
            param_grid=param_grid,
            X_train=X_train,
            y_train=y_train,
            scoring=get_default_scoring(task_type),
            cv=cv,
            method=tuning_method,
            n_iter=n_iter
        )

        best_model = tuning_results["best_model"]

    # 9. Evaluate Final Best Model
    best_model_results = evaluate_models(
        {best_model_name: best_model},
        X_test,
        y_test,
        task_type
    )[best_model_name]

    # 10. Save Model Bundle
    model_bundle = {
        "model": best_model,
        "preprocessor": preprocessor,
        "target_column": target_column,
        "task_type": task_type
    }

    model_path = save_model(
        model_bundle,
        best_model_name,
        save_dir
    )

    # 11. Save Model Metadata
    best_params = (
        tuning_results["best_params"]
        if tuning_results is not None
        else best_model.get_params(deep=False)
    )

    metadata_path = save_model_metadata(
        model_name=best_model_name,
        task_type=task_type,
        metrics=best_model_results,
        best_params=best_params,
        save_dir=save_dir
    )

    # 12. Experiment Tracking
    for model_name, model in trained_models.items():

        log_experiment(
            model_name=model_name,
            task_type=task_type,
            metrics=evaluation_results[model_name],
            params=model.get_params(deep=False)
        )

    if enable_tuning:

        log_experiment(
            model_name=f"{best_model_name}_Tuned",
            task_type=task_type,
            metrics=best_model_results,
            params=best_model.get_params(deep=False),
            model_path=model_path,
            metadata_path=metadata_path
        )

    else:

        log_experiment(
            model_name=best_model_name,
            task_type=task_type,
            metrics=best_model_results,
            params=best_model.get_params(deep=False),
            model_path=model_path,
            metadata_path=metadata_path
        )

    # 13. Return Results
    return {
        "validation": validation_report,
        "eda": eda_results,
        "evaluation": evaluation_results,
        "best_model_name": best_model_name,
        "best_model_results": best_model_results,
        "tuning": tuning_results,
        "preprocessor": preprocessor,
        "model_path": model_path,
        "metadata_path": metadata_path
    }
