from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR


def get_available_models(task_type):

    if task_type == "classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "SVM": SVC()
        }

    elif task_type == "regression":
        return {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(random_state=42),
            "SVR": SVR()
        }

    else:
        raise ValueError(
            "Task type must be classification or regression."
        )


def train_models(X_train, y_train, task_type, selected_models=None):

    available_models = get_available_models(task_type)

    if selected_models is None or "All Models" in selected_models:
        selected_models = list(available_models.keys())

    trained_models = {}

    for model_name in selected_models:

        if model_name not in available_models:
            raise ValueError(f"Unknown model: {model_name}")

        model = available_models[model_name]

        model.fit(X_train, y_train)

        trained_models[model_name] = model

    return trained_models