from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import clone


# 1. Get parameters for each model
def get_parameter_grid(model_name, task_type):

    classification_grids = {
        "Logistic Regression": {
            "C": [0.01, 0.1, 1, 10, 100],
            "solver": ["lbfgs", "liblinear"]
        },

        "Decision Tree": {
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        },

        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        },

        "SVM": {
            "C": [0.1, 1, 10, 100],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto"]
        }
    }

    regression_grids = {
        "Linear Regression": {
            "fit_intercept": [True, False]
        },

        "Decision Tree": {
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        },

        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        },

        "SVR": {
            "C": [0.1, 1, 10, 100],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto"]
        }
    }

    if task_type == "classification":
        parameter_grids = classification_grids

    elif task_type == "regression":
        parameter_grids = regression_grids

    else:
        raise ValueError(
            "Task type must be classification or regression."
        )

    if model_name not in parameter_grids:
        raise ValueError(
            f"No parameter grid available for {model_name}."
        )

    return parameter_grids[model_name]


# 2. Default scoring metric
def get_default_scoring(task_type):

    if task_type == "classification":
        return "f1_weighted"

    elif task_type == "regression":
        return "neg_root_mean_squared_error"

    else:
        raise ValueError("Invalid task type.")


# 3. Hyperparameter tuning
def tune_model(
    model,
    param_grid,
    X_train,
    y_train,
    scoring=None,
    cv=5,
    method="grid",
    n_iter=10
):

    # Validate inputs
    if model is None:
        raise ValueError("Model is not provided.")

    if param_grid is None or not param_grid:
        raise ValueError("Parameter grid is empty.")

    if len(y_train) < 2:
        raise ValueError("Not enough training samples.")

    if cv < 2:
        raise ValueError("CV must be at least 2.")

    # Adjust CV if dataset is small
    cv = min(cv, len(y_train))

    # Set default number of iterations
    n_iter = max(1, int(n_iter))

    # Set default scoring if not provided
    if scoring is None:
        scoring = "accuracy"

    # Create independent model copy
    model_copy = clone(model)

    # Select search method
    if method == "grid":

        search = GridSearchCV(
            estimator=model_copy,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            verbose=1,
            error_score="raise"
        )

    elif method == "random":

        search = RandomizedSearchCV(
            estimator=model_copy,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            random_state=42,
            verbose=1,
            error_score="raise"
        )

    else:
        raise ValueError(
            "Method must be 'grid' or 'random'."
        )

    # Train and search
    search.fit(X_train, y_train)

    # Return results
    return {
        "best_model": search.best_estimator_,
        "best_params": search.best_params_,
        "best_score": search.best_score_,
        "search_results": search.cv_results_
    }