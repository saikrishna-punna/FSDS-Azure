import logging
import os
import pickle

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def scorer(output_path, output_path_model, log_level, console_log, log_path):
    """This module is used to calculate the performance of the model.

    Parameters
    ----------
    *args list ->
    output_path : obj:`str`, optional
        path to save the output of created data
        default - ../data/
    output_path_model : obj:`str`, optional
        path to save the output of created data
        default - ../artifacts/
    log_level : obj:`str`, optional
        specify the log level
        deafult - DEBUG
    console_log : boolean, optional
        Boolean | To write logs to the console, default(True)
        deafult - True
    log_path : obj:`str`, optional
        specify the file to be used for logging
        default - None

    Returns
    -------
    final_rmse : float
        Root Mean Squared Error Score

    """

    logger = logging.getLogger()
    if log_path:
        fh = logging.FileHandler(log_path)
        logger.addHandler(fh)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    if console_log is False:
        logger.disable(logging.CRITICAL)

    logger.info("Unpickling the model pickle file")
    filename = os.path.join(output_path_model, "final_model.pkl")
    final_model = pickle.load(open(filename, "rb"))
    if final_model:
        logger.info("Unplickling Completed")
    else:
        logger.error("Unpickling Failed")

    mlflow.sklearn.autolog()

    logger.info("Reading the Test Files")
    X_test_prepared = pd.read_csv(os.path.join(output_path, "X_test_prepared.csv"))
    y_test = pd.read_csv(os.path.join(output_path, "y_test.csv"))
    final_predictions = final_model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    logger.info("Calculated the error")

    print(final_rmse)
    return final_rmse
