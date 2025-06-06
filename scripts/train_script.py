#!.venv-dev/bin/python

import argparse
import json

from housing.train import training


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

training(output_path, output_path_model, log_level, console_log, log_path)
