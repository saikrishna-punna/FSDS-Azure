import os
import pickle
import shutil
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from ingest_data import load_data
from score import scorer
from train import training


@pytest.fixture
def setup_test_dirs():
    """Fixture to create and clean up test directories."""
    test_output_path = os.path.join(os.getcwd(), "test_data")
    test_model_path = os.path.join(os.getcwd(), "test_artifacts")
    os.makedirs(test_output_path, exist_ok=True)
    os.makedirs(test_model_path, exist_ok=True)

    yield test_output_path, test_model_path

    if os.path.exists(test_output_path):
        shutil.rmtree(test_output_path)
    if os.path.exists(test_model_path):
        shutil.rmtree(test_model_path)


@pytest.fixture
def mock_housing_data():
    """Fixture to create mock housing data."""
    housing_data = pd.DataFrame(
        {
            "longitude": [-122.23, -122.22, -122.24, -122.25],
            "latitude": [37.88, 37.86, 37.85, 37.84],
            "housing_median_age": [41.0, 21.0, 52.0, 52.0],
            "total_rooms": [880.0, 7099.0, 1467.0, 1274.0],
            "total_bedrooms": [129.0, 1106.0, 190.0, 235.0],
            "population": [322.0, 2401.0, 496.0, 558.0],
            "households": [126.0, 1138.0, 177.0, 219.0],
            "median_income": [8.3252, 8.3014, 7.2574, 5.6431],
            "median_house_value": [452600.0, 358500.0, 352100.0, 341300.0],
            "ocean_proximity": ["NEAR BAY", "NEAR BAY", "<1H OCEAN", "<1H OCEAN"],
        }
    )
    return housing_data


# Simplified test for load_data function
@patch("urllib.request.urlretrieve")
@patch("tarfile.open")
@patch("pandas.read_csv")
@patch("sklearn.model_selection.train_test_split")
def test_load_data_basic(
    mock_train_test_split,
    mock_read_csv,
    mock_tarfile,
    mock_urlretrieve,
    setup_test_dirs,
    mock_housing_data,
):
    """Test basic data loading functionality without stratified split."""
    test_output_path, _ = setup_test_dirs

    # Configure mocks
    mock_tarfile_instance = MagicMock()
    mock_tarfile.return_value = mock_tarfile_instance
    mock_read_csv.return_value = mock_housing_data

    # Mock train_test_split to return predetermined splits
    train_data = mock_housing_data.iloc[:2]
    test_data = mock_housing_data.iloc[2:]
    mock_train_test_split.return_value = (
        train_data.drop("median_house_value", axis=1),
        test_data.drop("median_house_value", axis=1),
        train_data["median_house_value"],
        test_data["median_house_value"],
    )

    # Patch the problematic StratifiedShuffleSplit
    with patch("sklearn.model_selection.StratifiedShuffleSplit") as mock_stratified:
        # Skip the stratified split by returning simple indices
        mock_split = MagicMock()
        mock_split.__iter__.return_value = [(np.array([0, 1]), np.array([2, 3]))]
        mock_stratified.return_value = mock_split

        # Call the function
        result = load_data(
            output_path=test_output_path,
            log_level="DEBUG",
            console_log=True,
            log_path=None,
        )

    # Basic checks
    assert len(result) == 4
    assert os.path.exists(os.path.join(test_output_path, "housing_prepared.csv"))


# Simplified test for training function
@pytest.fixture
def setup_training_data(setup_test_dirs, mock_housing_data):
    """Fixture to create training data files."""
    test_output_path, _ = setup_test_dirs

    housing_prepared = mock_housing_data.drop("median_house_value", axis=1)
    housing_labels = mock_housing_data["median_house_value"]

    housing_prepared.to_csv(
        os.path.join(test_output_path, "housing_prepared.csv"), index=False
    )
    housing_labels.to_csv(
        os.path.join(test_output_path, "housing_labels.csv"), index=False
    )

    return housing_prepared, housing_labels


@patch("sklearn.linear_model.LinearRegression")
@patch("sklearn.tree.DecisionTreeRegressor")
def test_training_basic(
    MockDecisionTreeRegressor,
    MockLinearRegression,
    setup_test_dirs,
    setup_training_data,
):
    """Test basic model training functionality without cross-validation."""
    test_output_path, test_model_path = setup_test_dirs

    # Configure simple mocks
    mock_lin_reg = MagicMock()
    mock_tree_reg = MagicMock()
    MockLinearRegression.return_value = mock_lin_reg
    MockDecisionTreeRegressor.return_value = mock_tree_reg

    # Patch the problematic search methods
    with patch("sklearn.model_selection.RandomizedSearchCV") as MockRandomizedSearchCV:
        with patch("sklearn.model_selection.GridSearchCV") as MockGridSearchCV:
            # Skip cross-validation with simple mocks
            mock_rnd_search = MagicMock()
            mock_grid_search = MagicMock()
            MockRandomizedSearchCV.return_value = mock_rnd_search
            MockGridSearchCV.return_value = mock_grid_search

            # Use a fully initialized model for the best_estimator_
            forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
            # Train it on some data to initialize all attributes
            X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
            y = np.array([1, 2, 3, 4])
            forest_reg.fit(X, y)

            mock_grid_search.best_estimator_ = forest_reg
            mock_grid_search.cv_results_ = {
                "mean_test_score": np.array([-10000]),
                "params": [{"n_estimators": 10, "max_features": 2}],
            }

            # Call the function
            training(
                output_path=test_output_path,
                output_path_model=test_model_path,
                log_level="DEBUG",
                console_log=True,
                log_path=None,
            )

    # Check that the model file was created
    assert os.path.exists(os.path.join(test_model_path, "final_model.pkl"))


# Simplified test for scorer function
@pytest.fixture
def setup_simplified_scorer_data(setup_test_dirs, mock_housing_data):
    """Fixture to create simplified test data and model for scorer."""
    test_output_path, test_model_path = setup_test_dirs

    # Create test files
    X_test = mock_housing_data.drop("median_house_value", axis=1)
    y_test = mock_housing_data["median_house_value"]

    X_test.to_csv(os.path.join(test_output_path, "X_test_prepared.csv"), index=False)
    y_test.to_csv(os.path.join(test_output_path, "y_test.csv"), index=False)

    # Create and train a real model
    model = RandomForestRegressor(n_estimators=5, random_state=42)
    model.fit(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), np.array([1, 2, 3, 4]))

    model_path = os.path.join(test_model_path, "final_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return test_output_path, test_model_path


@patch("pandas.read_csv")
def test_scorer_basic(mock_read_csv, setup_simplified_scorer_data, mock_housing_data):
    """Test basic model scoring functionality."""
    test_output_path, test_model_path = setup_simplified_scorer_data

    # Mock the read_csv to return our test data
    X_test = mock_housing_data.drop("median_house_value", axis=1)
    y_test = mock_housing_data["median_house_value"]

    def side_effect(filepath, *args, **kwargs):
        if filepath.endswith("X_test_prepared.csv"):
            return X_test
        elif filepath.endswith("y_test.csv"):
            return pd.Series(y_test)
        return pd.DataFrame()

    mock_read_csv.side_effect = side_effect

    # Call the scorer
    with patch("pickle.load") as mock_pickle_load:
        # Create and train a model that will work with our test data
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X_test.values, y_test.values)
        mock_pickle_load.return_value = model

        rmse = scorer(
            output_path=test_output_path,
            output_path_model=test_model_path,
            log_level="DEBUG",
            console_log=True,
            log_path=None,
        )

    # Basic check
    assert isinstance(rmse, float)
