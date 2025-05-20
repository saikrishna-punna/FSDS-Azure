import argparse
import json
from datetime import datetime

import mlflow
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from housing.ingest_data import load_data
from housing.score import scorer
from housing.train import training


def main():
    def parse_bool(s: str) -> bool:
        try:
            return {"true": True, "false": False}[s.lower()]
        except KeyError:
            raise argparse.ArgumentTypeError(f"expected true/false, got {s}")

    # Default Configurations
    with open("config.json") as config_file:
        data = json.load(config_file)

    # Argparser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_path",
        help="path to save the output of created data",
        default=data["output_path"],
    )
    parser.add_argument(
        "-om",
        "--output_path_model",
        help="path to save the output of created model",
        default=data["output_path_model"],
    )
    parser.add_argument(
        "--log_level",
        help="specify the log level",
        default=data["log_level"],
    )
    parser.add_argument(
        "--log_path",
        help="specify the file to be used for logging",
        default=data["log_path"],
    )
    parser.add_argument(
        "--console_log",
        type=parse_bool,
        help="Boolean | To write logs to the console, default(True)",
        default=data["console_log"],
    )

    args = parser.parse_args()
    output_path = args.output_path
    output_path_model = args.output_path_model
    log_level = args.log_level
    log_path = args.log_path
    console_log = args.console_log

    # Create experiment and capture the returned experiment ID
    exp_name = f"Housing_{datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]}"
    exp_id = mlflow.create_experiment(exp_name)

    # Now use the returned experiment ID
    with mlflow.start_run(
        run_name="Housing",
        experiment_id=exp_id,
        tags={"version": "v1", "priority": "p1"},
        description="parent",
    ):
        mlflow.log_param("Housing", "yes")
        with mlflow.start_run(
            run_name="Data_Ingestion",
            experiment_id=exp_id,
            tags={"child": "true"},  # Tags should be a dictionary
            nested=True,
        ):
            mlflow.log_param("Data_Ingestion", "yes")
            load_data(output_path, log_level, console_log, log_path)

        with mlflow.start_run(
            run_name="Model_Training",
            experiment_id=exp_id,
            tags={"child": "true"},  # Tags should be a dictionary
            nested=True,
        ):
            mlflow.log_param("Model_Training", "yes")
            training(output_path, output_path_model, log_level, console_log, log_path)

        with mlflow.start_run(
            run_name="Scoring",
            experiment_id=exp_id,
            tags={"child": "true"},  # Tags should be a dictionary
            nested=True,
        ):
            mlflow.log_param("Scoring", "yes")
            scorer(output_path, output_path_model, log_level, console_log, log_path)


if __name__ == "__main__":
    main()
