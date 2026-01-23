"""Utility functions for MLflow."""
import os
import ast

import mlflow
import pandas as pd

from src.util.train_utils import set_seeds, _to_numpy

mlflow.set_tracking_uri("http://localhost:5000")


def set_client(host: str = 'http://localhost', port: str = '5000'):
    """Set MLFlow tracking server, default is localhost."""
    return mlflow.MlflowClient(tracking_uri=f'{host}:{port}')


def create_experiment_if_not_exists(experiment_name: str) -> str:
    """Create a new MLFlow experiment if it doesn't exist."""
    new_exp = mlflow.get_experiment_by_name(experiment_name)
    if new_exp:
        print(
            f'Experiment {experiment_name} already exists with ID: '
            f'{new_exp.experiment_id}')
        return new_exp.experiment_id
    else:
        # If it doesn't exist, create the experiment with the provided name
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f'Experiment {experiment_name} created with ID: {experiment_id}')
        return experiment_id


def log_parameters(params: dict):
    """Log parameters to tracking server."""
    for key, val in params.items():
        mlflow.log_param(key, val)


def log_param(name, param):
    mlflow.log_param(name, param)


def log_metrics(metrics: dict):
    """Log metrics to tracking server."""
    for key, val in metrics.items():
        mlflow.log_metric(key, val)


def log_model(model, model_name: str):
    """Log model to tracking server."""
    mlflow.pytorch.log_model(model, model_name)


def log_artifact(artifact_path: str):
    mlflow.log_artifact(artifact_path)

def log_results_mlflow(results: dict, prefix="test"):

    for name, val in results.items():
        # normalize to (seq, steps or None)
        if isinstance(val, (tuple, list)) and len(val) == 2:
            seq, steps = val
        else:
            seq, steps = val, None

        seq_np = _to_numpy(seq)
        if seq_np.size == 1:
            mlflow.log_metric(f"{prefix}_{name.lower()}", float(seq_np))
        else:
            # rare, but handle non-scalar seq
            for i, v in enumerate(seq_np.ravel(), start=1):
                mlflow.log_metric(f"{prefix}_{name.lower()}_{i}", float(v))

        if steps is not None:
            steps_np = _to_numpy(steps).ravel()
            for i, v in enumerate(steps_np, start=1):
                mlflow.log_metric(f"{prefix}_{name.lower()}_step_{i}", float(v))


def get_experiment_id(exp_name):
    """Get experiment ID from mlflow."""
    return mlflow.get_experiment_by_name(exp_name).experiment_id


def get_run_data(exp_id):
    """Collect all runs of a specific experiment."""
    all_runs = mlflow.search_runs([exp_id])
    return pd.DataFrame(all_runs)


def get_best_run_per_exp(exp_id: str, criterion: str):
    """Collect best runs of a specific experiment."""
    all_runs = mlflow.search_runs([exp_id])
    all_run_ids = all_runs['run_id']
    all_metrics = all_runs[f'metrics.{criterion}']
    run_df = pd.DataFrame({'Run ID': all_run_ids, 'Metrics': all_metrics})
    best_run = run_df.sort_values('Metrics',
                                  ascending=False).iloc[0]['Run ID']
    best_model_path = os.path.join(best_run, 'artifacts', 'trained_model',
                                   'data', 'model.pth')
    return best_run, best_model_path


def get_best_run_per_pred_len(exp_id: str, pred_len: int, results_list: list):
    """Collect best runs of a specific experiment and per prediction length."""
    all_runs = mlflow.search_runs([exp_id])
    cols = ['Run ID', 'MSE', 'MAE']
    qualified_runs = pd.DataFrame(columns=cols)

    def get_config(run_data):
        """Collect config from run_data."""
        config_str = run_data.data.params.get("TrainingConfig")

        if config_str:
            try:
                configuration = ast.literal_eval(config_str)
                return configuration
            except ValueError as e:
                print(f"Error parsing TrainingConfig for run {run_id}: {e}")
                return None
        else:
            print(f"TrainingConfig not found for run {run_id}.")
            return None

    for _, run in all_runs.iterrows():
        run_id = run['run_id']
        run_data_df = mlflow.get_run(run_id)

        config = get_config(run_data_df)
        if config:
            pred_len_value = config.get("pred_len")
            if pred_len_value == pred_len:
                mse_value = run.get('metrics.mse')
                mae_value = run.get('metrics.mae')
                if mse_value:
                    new_data = [[run_id, mse_value, mae_value]]
                    new_df = pd.DataFrame(new_data, columns=cols)
                    qualified_runs = pd.concat([qualified_runs, new_df],
                                               ignore_index=True)

    # Sorting to find the best run based on the criterion
    best_run = qualified_runs.sort_values('MSE', ascending=True).iloc[0][
        'Run ID']
    best_mse = qualified_runs.sort_values('MSE',
                                          ascending=True).iloc[0]['MSE']
    best_mae = qualified_runs.sort_values('MSE',
                                          ascending=True).iloc[0]['MAE']
    best_model_path = os.path.join(best_run, 'artifacts', 'trained_model',
                                   'data', 'model.pth')
    results_list.append({
        'Experiment ID': exp_id,
        'Prediction Length': pred_len,
        'Best Run ID': best_run,
        'mse': best_mse,
        'mae': best_mae,
        'Model Path': best_model_path
    })


def get_all_runs(exp_name):
    out_dir = os.path.join('key_files', 'results')
    # check for existence
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is not None:
        exp_id = exp.experiment_id
        runs_df = mlflow.search_runs(experiment_ids=[exp_id])
        csv_file = os.path.join(out_dir, f"{exp_name}_runs.csv")
        runs_df.to_csv(csv_file, index=False)
        print(f"Run data for '{exp_name}' has been saved to '{csv_file}'.")
    else:
        raise LookupError(f"Experiment with name '{exp_name}' does not exist.")


def retain_lowest_mae_run(experiment_name):
    """
    Retains only the run with the lowest metrics.mae for each run_name in the
    specified experiment.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found!")
    experiment_id = experiment.experiment_id
    runs = mlflow.search_runs(experiment_ids=experiment_id,
                              output_format="list")
    run_dict = {}
    for run in runs:
        run_name = run.data.tags.get("mlflow.runName")
        source_name = run.data.tags.get("mlflow.source.name")
        mae = run.data.metrics.get(
            "mae")  # Adjust the metric name if necessary

        if run_name and source_name:
            key = (run_name, source_name)
            if key not in run_dict or (
                    mae is not None and mae < run_dict[key]["mae"]):
                run_dict[key] = {"run_id": run.info.run_id, "mae": mae}

    # Collect run IDs to delete
    to_delete = set(run.info.run_id for run in runs) - set(
        data["run_id"] for data in run_dict.values())

    # Delete runs
    for run_id in to_delete:
        mlflow.delete_run(run_id)

    print(
        f"Deleted {len(to_delete)} runs.")


if __name__ == '__main__':
    flag = 'get_all'
    experiment1 = 'mm_exp_sampled'
    experiment2 = 'mm_exp_bench'
    if flag == 'create':
        new_id = create_experiment_if_not_exists(experiment1)
        new_id2 = create_experiment_if_not_exists(experiment2)
    elif flag == 'get_all':
        get_all_runs(experiment1)
        get_all_runs(experiment2)
    elif flag == 'delete':
        retain_lowest_mae_run(experiment1)
        retain_lowest_mae_run(experiment2)
    elif flag == 'create_local':
        new_id = create_experiment_if_not_exists('local-test')
    else:
        print('Invalid flag')
