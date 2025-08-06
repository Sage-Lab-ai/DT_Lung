from typing import Callable, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV

# set pandas display options
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 800)
pd.set_option("display.max_rows", 100)


# the default evaluation metrics
default_evaluation_metrics = {
    "MAE": mean_absolute_error,
    "MAPE": mean_absolute_percentage_error,
    "MSE": mean_squared_error,
}

# the default hyperparameters to search for each model
default_xgb_hyperparameters_to_search = {
    'n_estimators': [20, 50, 100, 150, 200, 300],
    'learning_rate': [0.002, 0.01, 0.1],
    'max_depth': [3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.5, 1],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.5, 0.7, 0.9, 1],
}

default_rf_hyperparameters_to_search = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 5, 7]
}

# the grid search settings
default_grid_search_settings = {
    "scoring": "neg_mean_absolute_error",
    "cv": 10,
    "n_jobs": -1
}


def evaluate_prediction(y_true: np.ndarray, y_pred: np.ndarray, metrics: dict[str, Callable]) -> dict[str, float]:
    """
    Evaluate the prediction results using various metrics.

    :param y_true: true target values
    :param y_pred: predicted target values

    :return: dictionary of evaluation results
    """

    evaluation = {metric_name: metric_func(y_true, y_pred) for metric_name, metric_func in metrics.items()}

    return evaluation


def grid_search_builder(model: BaseEstimator,
                        hyperparameters_to_search: dict[str, list[Any]],
                        grid_search_settings: dict[str, Any]) -> Callable[[], GridSearchCV]:
    def get_grid_search():
        return GridSearchCV(estimator=model, param_grid=hyperparameters_to_search, **grid_search_settings)

    return get_grid_search