"""MLflow setup: runs in mlflow.db, artifacts in mlruns/<experiment>/<run_id>/."""

import mlflow


def setup_mlflow(experiment):
    """Selects `experiment`, creating it with its own artifact folder if needed."""
    # Relative URI: the absolute default encodes spaces in the path as %20 and breaks on Windows
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=f"mlruns/{experiment}")

    mlflow.set_experiment(experiment)


def delete_runs(experiment, run_names):
    """Deletes earlier runs with these names so re-runs do not leave duplicates."""
    setup_mlflow(experiment)
    for run in mlflow.search_runs(experiment_names=[experiment], output_format="list"):
        if run.info.run_name in run_names:
            mlflow.delete_run(run.info.run_id)
