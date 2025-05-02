import os
import shutil
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, mock_open, patch
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
    # Create a simple housing dataframe
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
            "ocean_proximity": ["NEAR BAY", "<1H OCEAN", "<1H OCEAN", "NEAR BAY"],
        }
    )
    return housing_data


# Tests for load_data function
class TestLoadData:

    @patch("urllib.request.urlretrieve")
    @patch("tarfile.open")
    @patch("pandas.read_csv")
    def test_load_data_success(
        self,
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
        mock_tarfile_instance.close.assert_called_once()

        # Check that the output files were created
        assert os.path.exists(os.path.join(test_output_path, "housing_prepared.csv"))
        assert os.path.exists(os.path.join(test_output_path, "housing_labels.csv"))
        assert os.path.exists(os.path.join(test_output_path, "X_test_prepared.csv"))
        assert os.path.exists(os.path.join(test_output_path, "y_test.csv"))

    @patch("urllib.request.urlretrieve")
    @patch("pandas.read_csv")
    def test_load_data_empty_dataframe(
        self, mock_read_csv, mock_urlretrieve, setup_test_dirs
    ):
        """Test error handling when housing.csv is empty."""
        test_output_path, _ = setup_test_dirs

        # Configure mock to return empty dataframe
        mock_read_csv.return_value = pd.DataFrame()

        # Call the function with error logging redirected
        with patch("logging.Logger.error") as mock_logger_error:
            result = load_data(
                output_path=test_output_path,
                log_level="DEBUG",
                console_log=True,
                log_path=None,
            )

            # Verify error was logged
            mock_logger_error.assert_called_with("Unable to read housing.csv")

    @patch("urllib.request.urlretrieve", side_effect=Exception("Download failed"))
    def test_load_data_download_failure(self, mock_urlretrieve, setup_test_dirs):
        """Test handling of download failure."""
        test_output_path, _ = setup_test_dirs

        # Call the function and expect an exception
        with pytest.raises(Exception, match="Download failed"):
            load_data(
                output_path=test_output_path,
                log_level="DEBUG",
                console_log=True,
                log_path=None,
            )


# Tests for training function
class TestTraining:

    @pytest.fixture
    def setup_training_data(self, setup_test_dirs, mock_housing_data):
        """Fixture to create training data files."""
        test_output_path, _ = setup_test_dirs

        # Create dummy training files
        housing_prepared = mock_housing_data.drop("median_house_value", axis=1)
        housing_labels = mock_housing_data["median_house_value"]

        housing_prepared.to_csv(
            os.path.join(test_output_path, "housing_prepared.csv"), index=False
        )
        housing_labels.to_csv(
            os.path.join(test_output_path, "housing_labels.csv"), index=False
        )

        return housing_prepared, housing_labels

    @patch("sklearn.linear_model.LinearRegression.fit")
    @patch("sklearn.tree.DecisionTreeRegressor.fit")
    @patch("sklearn.model_selection.RandomizedSearchCV.fit")
    @patch("sklearn.model_selection.GridSearchCV.fit")
    @patch("pickle.dump")
    def test_training_success(
        self,
        mock_pickle_dump,
        mock_grid_search_fit,
        mock_rnd_search_fit,
        mock_tree_fit,
        mock_lin_fit,
        setup_test_dirs,
        setup_training_data,
    ):
        """Test successful model training and pickle creation."""
        test_output_path, test_model_path = setup_test_dirs

        # Configure GridSearchCV mock to return a model with feature importances
        mock_best_estimator = MagicMock()
        mock_best_estimator.feature_importances_ = np.array([0.1, 0.2, 0.3, 0.4])
        mock_grid_search = MagicMock()
        mock_grid_search.best_estimator_ = mock_best_estimator
        mock_grid_search.cv_results_ = {
            "mean_test_score": [np.array([-10000])],
            "params": [{"n_estimators": 10, "max_features": 2}],
        }
        mock_grid_search_fit.return_value = mock_grid_search

        # Call the function
        training(
            output_path=test_output_path,
            output_path_model=test_model_path,
            log_level="DEBUG",
            console_log=True,
            log_path=None,
        )

        # Verify the function called the expected methods
        mock_lin_fit.assert_called_once()
        mock_tree_fit.assert_called_once()
        mock_rnd_search_fit.assert_called_once()
        mock_grid_search_fit.assert_called_once()

        # Check that the model pickle was created
        assert mock_pickle_dump.called
        assert os.path.exists(test_model_path)

    def test_training_missing_data(self, setup_test_dirs):
        """Test handling of missing training files."""
        test_output_path, test_model_path = setup_test_dirs

        # Call the function with error logging redirected
        with patch("logging.Logger.error") as mock_logger_error:
            training(
                output_path=test_output_path,
                output_path_model=test_model_path,
                log_level="DEBUG",
                console_log=True,
                log_path=None,
            )

            # Verify error was logged for missing files
            mock_logger_error.assert_any_call("Unable to read Xtrain")
            mock_logger_error.assert_any_call("Unable to read ytrain")

    @patch("pandas.read_csv")
    @patch(
        "sklearn.linear_model.LinearRegression.fit",
        side_effect=Exception("Model training failed"),
    )
    def test_training_model_failure(
        self, mock_lin_fit, mock_read_csv, setup_test_dirs, mock_housing_data
    ):
        """Test handling of model training failure."""
        test_output_path, test_model_path = setup_test_dirs

        # Configure mock to return valid dataframes
        housing_prepared = mock_housing_data.drop("median_house_value", axis=1)
        housing_labels = mock_housing_data["median_house_value"]
        mock_read_csv.side_effect = [housing_prepared, housing_labels]

        # Call the function and expect an exception
        with pytest.raises(Exception, match="Model training failed"):
            training(
                output_path=test_output_path,
                output_path_model=test_model_path,
                log_level="DEBUG",
                console_log=True,
                log_path=None,
            )


# Tests for scorer function
class TestScorer:

    @pytest.fixture
    def setup_scorer_data(self, setup_test_dirs, mock_housing_data):
        """Fixture to create test data and model for scorer."""
        test_output_path, test_model_path = setup_test_dirs

        # Create dummy test files
        X_test = mock_housing_data.drop("median_house_value", axis=1)
        y_test = mock_housing_data["median_house_value"]

        X_test.to_csv(
            os.path.join(test_output_path, "X_test_prepared.csv"), index=False
        )
        y_test.to_csv(os.path.join(test_output_path, "y_test.csv"), index=False)

        # Create a dummy model pickle
        os.makedirs(test_model_path, exist_ok=True)
        model_path = os.path.join(test_model_path, "final_model.pkl")

        # Return paths and data for tests
        return test_output_path, test_model_path, model_path, X_test, y_test

    @patch("pickle.load")
    def test_scorer_success(self, mock_pickle_load, setup_scorer_data):
        """Test successful model scoring."""
        test_output_path, test_model_path, model_path, X_test, y_test = (
            setup_scorer_data
        )

        # Create a mock model that returns fixed predictions
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([400000, 350000, 350000, 340000])
        mock_pickle_load.return_value = mock_model

        # Call the function
        rmse = scorer(
            output_path=test_output_path,
            output_path_model=test_model_path,
            log_level="DEBUG",
            console_log=True,
            log_path=None,
        )

        # Verify RMSE is a float
        assert isinstance(rmse, float)

        # Verify the function called the expected methods
        mock_pickle_load.assert_called_once()
        mock_model.predict.assert_called_once()

    @patch("pickle.load", return_value=None)
    def test_scorer_model_loading_failure(self, mock_pickle_load, setup_scorer_data):
        """Test handling of model loading failure."""
        test_output_path, test_model_path, _, _, _ = setup_scorer_data

        # Call the function with error logging redirected
        with patch("logging.Logger.error") as mock_logger_error:
            scorer(
                output_path=test_output_path,
                output_path_model=test_model_path,
                log_level="DEBUG",
                console_log=True,
                log_path=None,
            )

            # Verify error was logged
            mock_logger_error.assert_called_with("Unpickling Failed")

    def test_scorer_missing_test_files(self, setup_test_dirs):
        """Test handling of missing test files."""
        test_output_path, test_model_path = setup_test_dirs

        # Create empty model directory but no test files
        os.makedirs(test_model_path, exist_ok=True)

        # Expect an error when files don't exist
        with pytest.raises(FileNotFoundError):
            scorer(
                output_path=test_output_path,
                output_path_model=test_model_path,
                log_level="DEBUG",
                console_log=True,
                log_path=None,
            )

    @patch("pickle.load")
    @patch("pandas.read_csv")
    def test_scorer_prediction_error(
        self, mock_read_csv, mock_pickle_load, setup_scorer_data
    ):
        """Test handling of prediction errors."""
        test_output_path, test_model_path, _, X_test, y_test = setup_scorer_data

        # Configure mocks to return valid data but make model.predict fail
        mock_read_csv.side_effect = [X_test, y_test]
        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("Prediction failed")
        mock_pickle_load.return_value = mock_model

        # Call the function and expect an exception
        with pytest.raises(Exception, match="Prediction failed"):
            scorer(
                output_path=test_output_path,
                output_path_model=test_model_path,
                log_level="DEBUG",
                console_log=True,
                log_path=None,
            )


# Tests for logging functionality across all functions
class TestLogging:

    @patch("logging.Logger.setLevel")
    @patch("logging.Logger.addHandler")
    @patch("logging.FileHandler")
    def test_file_logging(
        self, mock_file_handler, mock_add_handler, mock_set_level, setup_test_dirs
    ):
        """Test that file logging is set up correctly when log_path is provided."""
        test_output_path, _ = setup_test_dirs
        test_log_path = os.path.join(test_output_path, "test.log")

        # Mock additional calls to avoid actual execution
        with (
            patch("urllib.request.urlretrieve"),
            patch("tarfile.open"),
            patch("pandas.read_csv", return_value=pd.DataFrame()),
        ):

            # Call the function with a log file path
            load_data(
                output_path=test_output_path,
                log_level="DEBUG",
                console_log=True,
                log_path=test_log_path,
            )

            # Verify that file handler was created with the correct path
            mock_file_handler.assert_called_with(test_log_path)

            # Verify log handlers were added
            assert mock_add_handler.call_count >= 2  # At least file and stream handlers

    @patch("logging.Logger.disable")
    def test_console_log_disabled(self, mock_disable, setup_test_dirs):
        """Test that console logging is disabled when console_log is False."""
        test_output_path, _ = setup_test_dirs

        # Mock additional calls to avoid actual execution
        with (
            patch("urllib.request.urlretrieve"),
            patch("tarfile.open"),
            patch("pandas.read_csv", return_value=pd.DataFrame()),
        ):

            # Call the function with console_log=False
            load_data(
                output_path=test_output_path,
                log_level="DEBUG",
                console_log=False,
                log_path=None,
            )

            # Verify that logging was disabled
            mock_disable.assert_called_once()


# Integration test for all three functions
def test_end_to_end_integration(setup_test_dirs, mock_housing_data):
    """Test the entire workflow with mocked data."""
    test_output_path, test_model_path = setup_test_dirs

    # Create necessary mocks to avoid actual HTTP requests and file operations
    with (
        patch("urllib.request.urlretrieve"),
        patch("tarfile.open") as mock_tarfile,
        patch("pandas.read_csv", return_value=mock_housing_data),
        patch("pickle.dump") as mock_pickle_dump,
        patch("pickle.load") as mock_pickle_load,
    ):

        # Configure tarfile mock
        mock_tarfile_instance = MagicMock()
        mock_tarfile.return_value = mock_tarfile_instance

        # 1. Load and preprocess data
        housing_prepared, housing_labels, X_test_prepared, y_test = load_data(
            output_path=test_output_path,
            log_level="DEBUG",
            console_log=True,
            log_path=None,
        )

        # Verify that the first function completed successfully
        assert isinstance(housing_prepared, pd.DataFrame)
        assert isinstance(housing_labels, pd.Series) or isinstance(
            housing_labels, pd.DataFrame
        )

        # Create a mock model for the final estimator
        mock_model = RandomForestRegressor(random_state=42)
        mock_best_estimator = MagicMock()
        mock_best_estimator.feature_importances_ = np.array(
            [0.1] * len(housing_prepared.columns)
        )
        mock_best_estimator.predict.return_value = np.array(
            [400000, 350000, 350000, 340000]
        )

        # 2. Train models
        with patch("sklearn.model_selection.GridSearchCV.fit") as mock_grid_search_fit:
            # Configure GridSearchCV mock to return our mock model
            mock_grid_search = MagicMock()
            mock_grid_search.best_estimator_ = mock_best_estimator
            mock_grid_search.cv_results_ = {
                "mean_test_score": [np.array([-10000])],
                "params": [{"n_estimators": 10, "max_features": 2}],
            }
            mock_grid_search_fit.return_value = mock_grid_search

            # Run training
            training(
                output_path=test_output_path,
                output_path_model=test_model_path,
                log_level="DEBUG",
                console_log=True,
                log_path=None,
            )

        # Verify that model pickle was "created"
        assert mock_pickle_dump.called

        # 3. Score the model
        # Configure pickle load to return our mock model
        mock_pickle_load.return_value = mock_best_estimator

        # Run scorer
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
