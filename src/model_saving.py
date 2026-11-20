import joblib
import json
from pathlib import Path
from datetime import datetime


def save_model(model, model_name, save_dir="models"):

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    file_name = model_name.lower().replace(" ", "_") + ".joblib"
    model_path = save_path / file_name

    joblib.dump(model, model_path)

    return model_path


def load_model(model_path):

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    return joblib.load(model_path)


def save_model_metadata(
    model_name,
    task_type,
    metrics,
    best_params=None,
    save_dir="models"
):

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model_name": model_name,
        "task_type": task_type,
        "metrics": metrics,
        "best_params": best_params or {},
        "saved_at": datetime.now().isoformat()
    }

    metadata_file = (
        model_name.lower().replace(" ", "_") + "_metadata.json"
    )

    metadata_path = save_path / metadata_file

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False, default=str)

    return metadata_path