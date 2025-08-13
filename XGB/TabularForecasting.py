from .Dataset import *
from .utils import *
import pandas as pd
import numpy as np
from typing import Union, Optional, Callable, Iterable
from sklearn.model_selection import GridSearchCV, LeaveOneOut, KFold
from sklearn.base import BaseEstimator, clone
from xgboost import XGBRegressor, XGBClassifier
import os
import json
import pickle
import time
from tqdm import tqdm


class Estimator:
    X_FILE = "X.parquet"
    Y_FILE = "y.parquet"
    MODEL_FILE = "best_model.pkl"
    GRID_SEARCH_FILE = "grid_search.pkl"
    GRID_SEARCH_RESULTS_FILE = "grid_search_results.csv"
    PREDICTION_FILE = "prediction.csv"

    def __init__(self):
        self.X = pd.DataFrame()
        self.y = pd.Series()
        self.grid_search = None
        self.model = None
        self.prediction = None
        self.evaluation = None

    def fit(self, X: pd.DataFrame, y: pd.Series, model: Union[BaseEstimator, GridSearchCV]) -> 'Estimator':
        self.X = X
        self.y = y
        self.grid_search = model if isinstance(model, GridSearchCV) else None
        self.model = model
        self.model.fit(X, y)
        if self.grid_search:
            self.model = self.grid_search.best_estimator_
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X = X[self.X.columns]
        return pd.Series(self.model.predict(X), index=X.index, name=self.y.name)

    def predict_without(self, X: pd.DataFrame) -> pd.Series:
        X = X[self.X.columns]
        intersection = self.X.index.intersection(X.index)
        if intersection.size == 0:
            return self.predict(X)
        else:
            estimator = clone(self.model)
            estimator.fit(self.X.drop(intersection), self.y.drop(intersection))
            return pd.Series(estimator.predict(X), index=X.index, name=self.y.name)

    def predict_kfolds(self, X: pd.DataFrame, kfolds: int = 10, verbose: bool = False) -> pd.Series:
        prediction = pd.Series(index=X.index, name=self.y.name)
        if kfolds == 1:
            return self.predict_without(X)
        elif kfolds > 1 and kfolds < X.shape[0]:
            folds = KFold(n_splits=kfolds).split(X)
        else:
            folds = LeaveOneOut().split(X)
            kfolds = X.shape[0]
        if verbose:
            folds = tqdm(folds, desc="Cross Validation", unit="fold", total=kfolds)
        for train_index, val_index in folds:
            X_val = X.iloc[val_index]
            prediction.loc[X_val.index] = self.predict_without(X_val).values
        return prediction

    def validate(self, kfolds: int = 10, verbose: bool = False) -> pd.DataFrame:
        self.prediction = pd.DataFrame(index=self.X.index, columns=["True", "Train", "Validation"])
        self.prediction["True"] = self.y
        self.prediction["Train"] = self.predict(self.X)
        self.prediction["Validation"] = self.predict_kfolds(self.X, kfolds, verbose)
        return self.prediction

    def evaluate(self, metrics: Optional[dict[str, Callable]] = None) -> pd.DataFrame:
        metrics = default_evaluation_metrics if metrics is None else metrics
        self.evaluation = pd.DataFrame()
        self.evaluation["Train"] = evaluate_prediction(self.y.values, self.prediction["Train"].values, metrics)
        self.evaluation["Validation"] = evaluate_prediction(self.y.values, self.prediction["Validation"].values, metrics)
        self.evaluation["Difference"] = self.evaluation["Validation"] - self.evaluation["Train"]
        self.evaluation["Percentage Increase"] = self.evaluation["Difference"] / self.evaluation["Train"]
        return self.evaluation

    def get_grid_search_results(self) -> pd.DataFrame:
        if self.grid_search:
            results = pd.DataFrame(self.grid_search.cv_results_)
            results = results[["rank_test_score", "params", "mean_test_score", "std_test_score", "mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time"]]
            results["mean_test_score"] = -results["mean_test_score"]
            results.sort_values("rank_test_score", inplace=True)
            return results
        else:
            return pd.DataFrame()

    def save(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        self.X.to_parquet(os.path.join(folder, self.X_FILE))
        self.y.to_frame().to_parquet(os.path.join(folder, self.Y_FILE))
        if self.prediction is not None:
            self.prediction.to_csv(os.path.join(folder, self.PREDICTION_FILE))
        with open(os.path.join(folder, self.MODEL_FILE), "wb") as file:
            pickle.dump(self.model, file)
        if self.grid_search:
            with open(os.path.join(folder, self.GRID_SEARCH_FILE), "wb") as file:
                pickle.dump(self.grid_search, file)
            self.get_grid_search_results().to_csv(os.path.join(folder, self.GRID_SEARCH_RESULTS_FILE), index=False)

    def load(self, folder: str):
        self.X = pd.read_parquet(os.path.join(folder, self.X_FILE))
        self.y = pd.read_parquet(os.path.join(folder, self.Y_FILE)).squeeze()
        if os.path.exists(os.path.join(folder, self.PREDICTION_FILE)):
            self.prediction = pd.read_csv(os.path.join(folder, self.PREDICTION_FILE), index_col=0)
        with open(os.path.join(folder, self.MODEL_FILE), "rb") as file:
            self.model = pickle.load(file)
            # enable categorical features for XGBoost models
            if isinstance(self.model, XGBRegressor) or isinstance(self.model, XGBClassifier):
                self.model.enable_categorical = True
        if os.path.exists(os.path.join(folder, self.GRID_SEARCH_FILE)):
            with open(os.path.join(folder, self.GRID_SEARCH_FILE), "rb") as file:
                self.grid_search = pickle.load(file)


class Evaluator:

    def __init__(self, metrics: Optional[dict[str, Callable]] = None):

        self.metrics = metrics or default_evaluation_metrics

        self.predictions: dict[str, pd.DataFrame] = {}
        self.comparison: dict[str, pd.DataFrame] = {metrics_name: pd.DataFrame(columns=["Train", "Validation"]) for metrics_name in self.metrics.keys()}

    def add_estimator(self, estimator: Estimator, name: Optional[str] = None, kfolds: int = 10, verbose: bool = False):
        name = estimator.model.__class__.__name__ if name is None else name
        if estimator.prediction is None:
            estimator.validate(kfolds, verbose)
        self.predictions[name] = estimator.prediction
        estimator.evaluate()
        for metric_name, metric_comparisson in self.comparison.items():
            metric_comparisson.loc[name, 'Train'] = estimator.evaluation.loc[metric_name, 'Train']
            metric_comparisson.loc[name, 'Validation'] = estimator.evaluation.loc[metric_name, 'Validation']

    def save_comparison(self, savepath: str):
        # combine all metrics into a single dataframe, add the model names as prefix
        comparison = pd.concat(self.comparison.values(), keys=self.comparison.keys(), axis=1)
        comparison.to_csv(savepath)


class TabularForecasting:

    def __init__(self, training_set: Dataset):
        self.training_set = training_set
        self.parameter_estimators: dict[str, Estimator] = {}
        self.grid_search_summary: pd.DataFrame = pd.DataFrame()
        self.performance: pd.DataFrame = pd.DataFrame()
        self.predicted_Y: pd.DataFrame = pd.DataFrame(index=self.training_set.df.index, columns=self.training_set.y_columns)

    def apply_model(self, model_builder: Callable[[], BaseEstimator], parameters: Optional[Iterable[str]] = None, sleep: int = 0):
        parameters = self.training_set.y_columns if parameters is None else parameters
        for parameter in tqdm(parameters, desc="Parameters", unit="parameter"):
            estimator = Estimator()
            self.parameter_estimators[parameter] = estimator
            X, y = self.training_set.get_X_y(parameter)
            estimator.fit(X, y, model_builder())
            estimator.validate()
            if estimator.grid_search:
                self.grid_search_summary.loc[parameter, "Best Params"] = str(estimator.grid_search.best_params_)
                self.grid_search_summary.loc[parameter, "Best Score"] = estimator.grid_search.best_score_
            self.performance[parameter] = estimator.evaluate().stack()
            predicted_y = estimator.prediction["Validation"]
            self.predicted_Y.loc[predicted_y.index, parameter] = predicted_y
            remaining_indices = self.training_set.df.index.difference(predicted_y.index)
            if remaining_indices.size > 0:
                X_remaining = self.training_set.df.loc[remaining_indices, estimator.X.columns]
                self.predicted_Y.loc[remaining_indices, parameter] = estimator.predict_without(X_remaining)
            if sleep > 0:
                time.sleep(sleep)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        predict_Y = pd.DataFrame(index=X.index, columns=self.training_set.y_columns)
        for parameter, estimator in self.parameter_estimators.items():
            predict_Y[parameter] = estimator.predict(X)
        return predict_Y

    def predict_without(self, X: pd.DataFrame):
        predict_Y = pd.DataFrame(index=X.index, columns=self.training_set.y_columns)
        for parameter, estimator in self.parameter_estimators.items():
            predict_Y[parameter] = estimator.predict_without(X)
        return predict_Y

    def predict_kfolds(self, X: pd.DataFrame, kfolds: int = 10, verbose: bool = False) -> pd.DataFrame:
        predict_Y = pd.DataFrame(index=X.index, columns=self.training_set.y_columns)
        param_est = self.parameter_estimators.items()
        if verbose:
            param_est = tqdm(param_est, desc="Parameters", unit="parameter")
        for parameter, estimator in param_est:
            predict_Y[parameter] = estimator.predict_kfolds(X, kfolds)
        return predict_Y

    def save(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        self.grid_search_summary.to_csv(os.path.join(folder, "grid_search_summary.csv"))
        self.performance.to_csv(os.path.join(folder, "performance.csv"))
        self.predicted_Y.to_csv(os.path.join(folder, "predicted_Y.csv"))
        for parameter, estimator in self.parameter_estimators.items():
            estimator.save(os.path.join(folder, parameter))

    def load(self, folder: str):
        if os.path.exists(os.path.join(folder, "grid_search_summary.csv")):
            self.grid_search_summary = pd.read_csv(os.path.join(folder, "grid_search_summary.csv"), index_col=0)
        if os.path.exists(os.path.join(folder, "performance.csv")):
            self.performance = pd.read_csv(os.path.join(folder, "performance.csv"), index_col=[0, 1])
        if not os.path.exists(os.path.join(folder, "predicted_Y.csv")):
            self.predicted_Y = pd.read_csv(os.path.join(folder, "predicted_Y.csv"), index_col=0)
        for parameter in os.listdir(folder):
            if not os.path.isdir(os.path.join(folder, parameter)):
                continue
            estimator = Estimator()
            estimator.load(os.path.join(folder, parameter))
            self.parameter_estimators[parameter] = estimator

