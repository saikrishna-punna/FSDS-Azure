import logging
import os
import pickle

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor


def training(output_path, output_path_model, log_level, console_log, log_path):
    """This model is used to train the model on housing_prepared & housing_labels

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
    final_model.pkl : pickle file
        final_model pickle file is generated and stored at output_path_model

    """
    mlflow.sklearn.autolog()
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

    logger.info("Reading the Training Files")
    housing_prepared = pd.read_csv(os.path.join(output_path, "housing_prepared.csv"))
    if housing_prepared.empty:
        logger.error("Unable to read Xtrain")
    else:
        logger.info("Read Xtrain")
    housing_labels = pd.read_csv(os.path.join(output_path, "housing_labels.csv"))
    if housing_labels.empty:
        logger.error("Unable to read ytrain")
    else:
        logger.info("Read ytrain")

    logger.info("Initialising Linear Regression Model")
    lin_reg = LinearRegression()
    logger.info("Fitting Linear Regression Model")
    lin_reg.fit(housing_prepared, housing_labels)

    logger.info("Predicting Linear Regression Model")
    housing_predictions = lin_reg.predict(housing_prepared)
    logger.info("Calculating Linear Regression Model MSE")
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    lin_rmse

    logger.info("Calculating Linear Regression Model MSE")
    lin_mae = mean_absolute_error(housing_labels, housing_predictions)
    lin_mae

    tree_reg = DecisionTreeRegressor(random_state=42)
    logger.info("Fitting Decision Tree Regression Model")
    tree_reg.fit(housing_prepared, housing_labels)

    logger.info("Predicting Decision Tree Regression Model")
    housing_predictions = tree_reg.predict(housing_prepared)
    logger.info("Calculating Decision Tree Regression Model MSE")
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    tree_rmse

    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }

    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    logger.info("Fitting Random Forest Regression Model")
    rnd_search.fit(housing_prepared, housing_labels)
    cvres = rnd_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]

    logger.info("Finding best parameters for Random Forest Regression Model")
    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    logger.info("Fitting Random Forest Regression Model with best params")
    grid_search.fit(housing_prepared, housing_labels)

    grid_search.best_params_
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    feature_importances = grid_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, housing_prepared.columns), reverse=True)

    final_model = grid_search.best_estimator_

    logger.info("Creating model pickle for best Model")
    os.makedirs(output_path_model, exist_ok=True)
    filename = os.path.join(output_path_model, "final_model.pkl")
    pickle.dump(final_model, open(filename, "wb"))

    print("Final Model pickle created")
