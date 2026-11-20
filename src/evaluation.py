import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_models(trained_models, X_test, y_test, task_type):

    evaluation_results = {}

    for model_name, model in trained_models.items():

        predictions = model.predict(X_test)

        if task_type == "classification":

            metrics = {
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(
                    y_test, predictions,
                    average="weighted",
                    zero_division=0
                ),
                "Recall": recall_score(
                    y_test, predictions,
                    average="weighted",
                    zero_division=0
                ),
                "F1 Score": f1_score(
                    y_test, predictions,
                    average="weighted",
                    zero_division=0
                ),
                "Confusion Matrix": confusion_matrix(
                    y_test, predictions
                ).tolist()
            }

        elif task_type == "regression":

            mse = mean_squared_error(y_test, predictions)

            metrics = {
                "MAE": mean_absolute_error(y_test, predictions),
                "MSE": mse,
                "RMSE": np.sqrt(mse),
                "R2 Score": r2_score(y_test, predictions)
            }

        else:
            raise ValueError(
                "Task type must be classification or regression."
            )

        evaluation_results[model_name] = metrics

    return evaluation_results


def get_best_model(evaluation_results, task_type):

    if not evaluation_results:
        raise ValueError("No evaluation results available.")

    if task_type == "classification":
        best_model = max(
            evaluation_results,
            key=lambda name: evaluation_results[name]["F1 Score"]
        )

    elif task_type == "regression":
        best_model = min(
            evaluation_results,
            key=lambda name: evaluation_results[name]["RMSE"]
        )

    else:
        raise ValueError("Invalid task type.")

    return best_model