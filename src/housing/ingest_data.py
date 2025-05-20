import logging
import os
import tarfile
import urllib.request

import mlflow

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


def load_data(output_path, log_level, console_log, log_path):
    """This function is used to load the data and preprocess it.

    Parameters
    ----------
    *args list ->
    output_path : obj:`str`, optional
        path to save the output of created data
        default - ../data/
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
    housing_prepared, housing_labels, X_test_prepared, y_test : csv
        Xtrain, ytrain, Xtest and ytest CSV are generated and stored at the output_path

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

    # DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    housing_path = os.path.join("data", "housing")
    housing_url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"

    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

    csv_path = os.path.join(housing_path, "housing.csv")
    logger.info("Reading the housing.csv File")
    housing = pd.read_csv(csv_path)
    mlflow.log_artifact(csv_path)
    if housing.empty:
        logger.error("Unable to read housing.csv")
    else:
        logger.info("Read housing.csv")

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    try:
        for train_index, test_index in split.split(housing, housing["income_cat"]):
            strat_train_set = housing.loc[train_index]
            strat_test_set = housing.loc[test_index]
    except Exception as e:
        strat_train_set = housing
        strat_test_set = housing

    def income_cat_proportions(data):
        return data["income_cat"].value_counts() / len(data)

    logger.info("Creating train and test set")
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    if train_set.empty:
        logger.error("Unable to create train")
    else:
        logger.info("created train")
    if test_set.empty:
        logger.error("Unable to create test")
    else:
        logger.info("created test")

    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing),
            "Stratified": income_cat_proportions(strat_test_set),
            "Random": income_cat_proportions(test_set),
        }
    ).sort_index()
    compare_props["Rand. %error"] = (
        100 * compare_props["Random"] / compare_props["Overall"] - 100
    )
    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.copy()
    housing.plot(kind="scatter", x="longitude", y="latitude")
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

    numeric_housing = housing.select_dtypes(include=["number"])
    corr_matrix = numeric_housing.corr()
    # corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    housing = strat_train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    logger.info("Imputing data")
    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)
    logger.info("Data Imputed")

    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_num = X_test.drop("ocean_proximity", axis=1)
    X_test_prepared = imputer.transform(X_test_num)
    X_test_prepared = pd.DataFrame(
        X_test_prepared, columns=X_test_num.columns, index=X_test.index
    )
    X_test_prepared["rooms_per_household"] = (
        X_test_prepared["total_rooms"] / X_test_prepared["households"]
    )
    X_test_prepared["bedrooms_per_room"] = (
        X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
    )
    X_test_prepared["population_per_household"] = (
        X_test_prepared["population"] / X_test_prepared["households"]
    )

    X_test_cat = X_test[["ocean_proximity"]]
    X_test_prepared = X_test_prepared.join(pd.get_dummies(X_test_cat, drop_first=True))

    logger.info("Generating Xtrain, Xtest, ytrain, ytest csv")
    housing_prepared.to_csv(
        os.path.join(output_path, "housing_prepared.csv"), index=False
    )
    housing_labels.to_csv(os.path.join(output_path, "housing_labels.csv"), index=False)
    X_test_prepared.to_csv(
        os.path.join(output_path, "X_test_prepared.csv"), index=False
    )
    y_test.to_csv(os.path.join(output_path, "y_test.csv"), index=False)
    logger.info("Generated Xtrain, Xtest, ytrain, ytest csv")

    return housing_prepared, housing_labels, X_test_prepared, y_test
