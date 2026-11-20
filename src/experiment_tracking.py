import mlflow
from numbers import Real


def log_experiment(
    model_name,
    task_type,
    metrics,
    params=None,
    model_path=None,
    metadata_path=None
):

    mlflow.set_experiment("AutoML Experiments")

    with mlflow.start_run(run_name=model_name) as run:

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("task_type", task_type)

        if params:

            safe_params = {
                key: str(value)
                for key, value in params.items()
            }

            mlflow.log_params(safe_params)

        numeric_metrics = {}

        for key, value in metrics.items():

            if isinstance(value, Real) and not isinstance(value, bool):
                numeric_metrics[key] = float(value)

        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics)

        if model_path:
            mlflow.log_artifact(
                str(model_path),
                artifact_path="model"
            )

        if metadata_path:
            mlflow.log_artifact(
                str(metadata_path),
                artifact_path="metadata"
            )

        return run.info.run_id