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
    # Create test directories
    test_output_path = os.path.join(os.getcwd(), "test_data")
    test_model_path = os.path.join(os.getcwd(), "test_artifacts")
    os.makedirs(test_output_path, exist_ok=True)
    os.makedirs(test_model_path, exist_ok=True)

    # Return paths for tests to use
    yield test_output_path, test_model_path

    # Clean up after tests
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
            "ocean_proximity_code": [1, 2, 2, 1],  # Numeric encoding
        }
    )
    return housing_data


# Main test for load_data function
@patch("urllib.request.urlretrieve")
@patch("tarfile.open")
@patch("pandas.read_csv")
@patch("sklearn.model_selection.train_test_split")
@patch("sklearn.model_selection.StratifiedShuffleSplit")
def test_load_data_success(
    mock_stratified,
    mock_train_test_split,
    mock_read_csv,
    mock_tarfile,
    mock_urlretrieve,
    setup_test_dirs,
    mock_housing_data,
):
    """Test successful data loading and processing."""
    test_output_path, _ = setup_test_dirs

    # Configure mocks
    mock_tarfile_instance = MagicMock()
    mock_tarfile.return_value = mock_tarfile_instance
    mock_read_csv.return_value = mock_housing_data

    # Mock the stratified split to avoid the "too few members" error
    mock_split = MagicMock()
    mock_indices = [(np.array([0, 1]), np.array([2, 3]))]  # train, test indices
    mock_split.__iter__.return_value = mock_indices
    mock_stratified.return_value = mock_split

    # Mock train_test_split to return predetermined splits
    train_data = mock_housing_data.iloc[:2]
    test_data = mock_housing_data.iloc[2:]
    mock_train_test_split.return_value = (
        train_data.drop("median_house_value", axis=1),
        test_data.drop("median_house_value", axis=1),
        train_data["median_house_value"],
        test_data["median_house_value"],
    )

    # Call the function
    result = load_data(
        output_path=test_output_path,
        log_level="DEBUG",
        console_log=True,
        log_path=None,
    )

    # Check that the function returns expected tuple with 4 elements
    assert len(result) == 4
    housing_prepared, housing_labels, X_test_prepared, y_test = result

    # Verify the function called the expected methods
    mock_urlretrieve.assert_called_once()
    mock_tarfile.assert_called_once()
    mock_tarfile_instance.extractall.assert_called_once()

    # Check that the output files were created
    assert os.path.exists(os.path.join(test_output_path, "housing_prepared.csv"))
    assert os.path.exists(os.path.join(test_output_path, "housing_labels.csv"))
    assert os.path.exists(os.path.join(test_output_path, "X_test_prepared.csv"))
    assert os.path.exists(os.path.join(test_output_path, "y_test.csv"))


# Main test for training function
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
@patch("sklearn.model_selection.RandomizedSearchCV")
@patch("sklearn.model_selection.GridSearchCV")
def test_training_success(
    MockGridSearchCV,
    MockRandomizedSearchCV,
    MockDecisionTreeRegressor,
    MockLinearRegression,
    setup_test_dirs,
    setup_training_data,
):
    """Test successful model training and pickle creation."""
    test_output_path, test_model_path = setup_test_dirs

    # Configure mocks for model classes
    mock_lin_reg = MagicMock()
    mock_tree_reg = MagicMock()
    mock_rnd_search = MagicMock()
    mock_grid_search = MagicMock()

    # Return the mocks when constructors are called
    MockLinearRegression.return_value = mock_lin_reg
    MockDecisionTreeRegressor.return_value = mock_tree_reg
    MockRandomizedSearchCV.return_value = mock_rnd_search
    MockGridSearchCV.return_value = mock_grid_search

    real_model = RandomForestRegressor(n_estimators=10, random_state=42)

    mock_grid_search.best_estimator_ = real_model
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

    # Verify the function called the expected methods
    mock_lin_reg.fit.assert_called_once()
    mock_tree_reg.fit.assert_called_once()
    mock_rnd_search.fit.assert_called_once()
    mock_grid_search.fit.assert_called_once()

    # Check that the model pickle was created
    assert os.path.exists(os.path.join(test_model_path, "final_model.pkl"))


# Main test for scorer function
@pytest.fixture
def setup_scorer_data(setup_test_dirs, mock_housing_data):
    """Fixture to create test data and model for scorer."""
    test_output_path, test_model_path = setup_test_dirs

    # Create dummy test files with numeric-only data
    X_test = mock_housing_data.drop("median_house_value", axis=1)
    y_test = mock_housing_data["median_house_value"]

    X_test.to_csv(os.path.join(test_output_path, "X_test_prepared.csv"), index=False)
    y_test.to_csv(os.path.join(test_output_path, "y_test.csv"), index=False)

    # Create a real model object (not a MagicMock) for pickling
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model_path = os.path.join(test_model_path, "final_model.pkl")

    # Actually save the model to disk
    os.makedirs(test_model_path, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return test_output_path, test_model_path, model_path, X_test, y_test


def test_scorer_success(setup_scorer_data):
    """Test successful model scoring."""
    test_output_path, test_model_path, model_path, X_test, y_test = setup_scorer_data

    # Since we've created a real model file, we don't need to mock pickle.load
    # just call the scorer function directly
    rmse = scorer(
        output_path=test_output_path,
        output_path_model=test_model_path,
        log_level="DEBUG",
        console_log=True,
        log_path=None,
    )

    # Verify RMSE is a float
    assert isinstance(rmse, float)
    assert rmse > 0  # RMSE should be positive


# Integration test
@patch("urllib.request.urlretrieve")
@patch("tarfile.open")
@patch("pandas.read_csv")
@patch("sklearn.model_selection.StratifiedShuffleSplit")
def test_simplified_integration(
    mock_stratified,
    mock_read_csv,
    mock_tarfile,
    mock_urlretrieve,
    setup_test_dirs,
    mock_housing_data,
):
    """A simplified end-to-end test of the entire workflow."""
    test_output_path, test_model_path = setup_test_dirs

    # Configure mocks
    mock_tarfile_instance = MagicMock()
    mock_tarfile.return_value = mock_tarfile_instance
    mock_read_csv.return_value = mock_housing_data

    # Mock the stratified split to avoid the "too few members" error
    mock_split = MagicMock()
    mock_indices = [(np.array([0, 1]), np.array([2, 3]))]  # train, test indices
    mock_split.__iter__.return_value = mock_indices
    mock_stratified.return_value = mock_split

    # 1. Load and preprocess data
    load_data(
        output_path=test_output_path,
        log_level="DEBUG",
        console_log=True,
        log_path=None,
    )

    # Patch model training components just for this part
    with (
        patch("sklearn.linear_model.LinearRegression") as MockLinearRegression,
        patch("sklearn.tree.DecisionTreeRegressor") as MockDecisionTreeRegressor,
        patch("sklearn.model_selection.RandomizedSearchCV") as MockRandomizedSearchCV,
        patch("sklearn.model_selection.GridSearchCV") as MockGridSearchCV,
    ):

        # Configure mocks
        mock_lin_reg = MagicMock()
        mock_tree_reg = MagicMock()
        mock_rnd_search = MagicMock()
        mock_grid_search = MagicMock()

        MockLinearRegression.return_value = mock_lin_reg
        MockDecisionTreeRegressor.return_value = mock_tree_reg
        MockRandomizedSearchCV.return_value = mock_rnd_search
        MockGridSearchCV.return_value = mock_grid_search

        # Create a real model object that can be properly pickled
        real_model = RandomForestRegressor(n_estimators=10, random_state=42)
        mock_grid_search.best_estimator_ = real_model
        mock_grid_search.cv_results_ = {
            "mean_test_score": np.array([-10000]),
            "params": [{"n_estimators": 10, "max_features": 2}],
        }

        # 2. Train models
        training(
            output_path=test_output_path,
            output_path_model=test_model_path,
            log_level="DEBUG",
            console_log=True,
            log_path=None,
        )

    # 3. Score the model
    rmse = scorer(
        output_path=test_output_path,
        output_path_model=test_model_path,
        log_level="DEBUG",
        console_log=True,
        log_path=None,
    )

    # Verify that we got a valid RMSE score
    assert isinstance(rmse, float)
    assert rmse > 0  # RMSE should be positive
