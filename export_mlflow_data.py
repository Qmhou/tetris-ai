# export_mlflow_data.py
# (版本：支持导出指定的Run ID)

import mlflow
import pandas as pd
import os
import argparse # 引入argparse来处理命令行参数

# --- 配置区 ---
EXPERIMENT_NAME = "Tetris CNN T-Spin Training"
MLFLOW_TRACKING_URI = "mlruns"
OUTPUT_CSV_NAME = "tetris_training_metrics_export.csv"
# --- 配置区结束 ---

def export_metrics_to_csv(run_ids_to_export=None):
    """
    Connects to MLflow and exports specified runs (or all runs) to a single CSV.
    """
    print(f"Connecting to MLflow at: {os.path.abspath(MLFLOW_TRACKING_URI)}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            print(f"Error: Experiment '{EXPERIMENT_NAME}' not found.")
            return

        print(f"Found experiment '{EXPERIMENT_NAME}' with ID: {experiment.experiment_id}")

        # --- 核心修改：根据是否提供了run_ids来构建搜索条件 ---
        filter_string = ""
        if run_ids_to_export:
            # Create a filter string like: "run_id IN ('id1', 'id2', ...)"
            filter_string = "run_id IN ({})".format(
                ", ".join([f"'{run_id}'" for run_id in run_ids_to_export])
            )
            print(f"Filtering for specific runs: {run_ids_to_export}")
        else:
            print("No specific run IDs provided. Exporting all runs in the experiment.")

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string
        )
        # --- 修改结束 ---

        if runs.empty:
            print("No runs found matching the criteria.")
            return

        print(f"Found {len(runs)} runs to process.")
        all_runs_data = []
        client = mlflow.tracking.MlflowClient()

        for index, run_info in runs.iterrows():
            run_id = run_info['run_id']
            print(f"Processing Run ID: {run_id}...")

            run_metrics = {}
            for metric_key in run_info.keys():
                if metric_key.startswith('metrics.'):
                    clean_metric_key = metric_key.replace('metrics.', '')
                    metric_history = client.get_metric_history(run_id, clean_metric_key)
                    run_metrics[clean_metric_key] = {m.step: m.value for m in metric_history}

            run_df = pd.DataFrame(run_metrics).rename_axis('step').reset_index()

            run_df['run_id'] = run_id
            for param_key in run_info.keys():
                if param_key.startswith('params.'):
                    clean_param_key = param_key.replace('params.', '')
                    run_df[clean_param_key] = run_info[param_key]

            all_runs_data.append(run_df)

        if all_runs_data:
            master_df = pd.concat(all_runs_data, ignore_index=True)

            id_cols = ['run_id', 'step']
            param_cols = sorted([col for col in master_df.columns if col.startswith('per_') or col in ['learning_rate', 'gamma', 'batch_size']])
            metric_cols = sorted([col for col in master_df.columns if col not in id_cols and col not in param_cols])
            master_df = master_df[id_cols + param_cols + metric_cols]

            master_df.to_csv(OUTPUT_CSV_NAME, index=False)
            print(f"\nSuccess! All data has been exported to '{OUTPUT_CSV_NAME}'")
        else:
            print("No data was processed.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    try:
        import pandas
    except ImportError:
        print("Pandas is not installed. Please run 'pip install pandas' to use this script.")
        exit()

    # --- 核心修改：使用 argparse 解析命令行参数 ---
    parser = argparse.ArgumentParser(description="Export MLflow experiment data to a CSV file.")
    parser.add_argument(
        '--run_ids',
        nargs='+',  # This allows accepting one or more values for this argument
        type=str,
        help="A list of specific Run IDs to export. If not provided, all runs will be exported."
    )
    args = parser.parse_args()

    export_metrics_to_csv(run_ids_to_export=args.run_ids)