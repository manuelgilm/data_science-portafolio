from typing import Dict
from typing import Any
from typing import Optional

import mlflow


def check_active_run() -> Optional[str]:
    """
    Check if there is an active run and return the run_id
    """
    active_run = mlflow.active_run()
    if active_run:
        run_id = active_run.info.run_id
        return run_id
    else:
        print("No active run found")
        return None


def log_dictionary(dictionary: Dict[str, Any], type: str) -> None:
    """
    Log a dictionary to MLflow

    :param dictionary: Dictionary to log
    :param type: Type of dictionary to log (metrics, tags, params)

    """
    run_id = check_active_run()
    if not run_id:
        print("No active run found")
        return
    if type == "metrics":
        mlflow.log_metrics(dictionary)
    elif type == "tags":
        mlflow.set_tags(dictionary)
    elif type == "params":
        mlflow.log_params(dictionary)
    elif type == "artifacts":
        for local_path, artifact_path in dictionary.items():
            mlflow.log_artifact(local_path=local_path, artifact_path=artifact_path)
    else:
        raise ValueError(f"Invalid type: {type}")
