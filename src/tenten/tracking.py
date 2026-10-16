"""MLflow: runs en mlflow.db y artefactos en mlruns/<experimento>/<run_id>/."""

import mlflow


def setup_mlflow(experiment):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=f"mlruns/{experiment}")

    mlflow.set_experiment(experiment)


def delete_runs(experiment, run_names):
    """Borra de MLflow los runs anteriores con estos nombres, para no tener duplicados."""
    setup_mlflow(experiment)
    for run in mlflow.search_runs(experiment_names=[experiment], output_format="list"):
        if run.info.run_name in run_names:
            mlflow.delete_run(run.info.run_id)
