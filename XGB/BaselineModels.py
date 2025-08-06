from .TemporalOrder import *
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


baseline_models = {}

def register_baseline_model(cls):
    baseline_models[cls.__name__] = cls
    return cls


@register_baseline_model
class MedianTargetBaseline(BaseEstimator, RegressorMixin):
    """A simple baseline model that predicts the median of training target values."""

    def __init__(self):
        self.target_ = None
        self.median_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.target_ = y.name
        self.median_ = y.median()

    def predict(self, X: pd.DataFrame):
        return pd.Series(index=X.index, name=self.target_, data=self.median_)


@register_baseline_model
class MeanTargetBaseline(BaseEstimator, RegressorMixin):
    """A simple baseline model that predicts the mean of training target values."""

    def __init__(self, whis_iqr: float = 3):
        self.whis_iqr = whis_iqr
        self.target_ = None
        self.mean_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.target_ = y.name
        iq1 = np.percentile(y, 25)
        iq3 = np.percentile(y, 75)
        iqr = iq3 - iq1
        lower_bound = iq1 - self.whis_iqr * iqr
        upper_bound = iq3 + self.whis_iqr * iqr
        self.mean_ = np.mean(y[(y >= lower_bound) & (y <= upper_bound)])

    def predict(self, X: pd.DataFrame):
        return pd.Series(index=X.index, name=self.target_, data=self.mean_)


@register_baseline_model
class LastRecordBaseline(BaseEstimator, RegressorMixin):
    """A simple baseline model that predicts the last record of the target values."""

    def __init__(self, temp_order: TemporalOrder):
        self.target_ = None
        self.median_ = None
        self.temp_order = temp_order

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.target_ = y.name
        self.median_ = y.median()

    def predict(self, X: pd.DataFrame):
        pre_features = reversed(self.temp_order.pre(self.target_))
        prediction = pd.Series(index=X.index, name=self.target_, data=np.nan)
        for feature in pre_features:
            if feature in X.columns:
                prediction.fillna(X[feature], inplace=True)
        prediction.fillna(self.median_, inplace=True)
        return prediction


@register_baseline_model
class AdjustedLastRecordBaseline(BaseEstimator, RegressorMixin):
    """A simple baseline model that predicts the last record of the target values plus a constant."""

    def __init__(self, temp_order: TemporalOrder):
        self.target_ = None
        self.median_ = None
        self.adjust_ = None
        self.temp_order = temp_order

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.target_ = y.name
        self.median_ = y.median()
        pre_features = reversed(self.temp_order.pre(self.target_))
        self.adjust_ = {}
        for feature in pre_features:
            if feature in X.columns:
                self.adjust_[feature] = self.median_ - X[feature].median()

    def predict(self, X: pd.DataFrame):
        pre_features = reversed(self.temp_order.pre(self.target_))
        prediction = pd.Series(index=X.index, name=self.target_, data=np.nan)
        for feature in pre_features:
            if feature in X.columns:
                prediction.fillna(X[feature] + self.adjust_[feature], inplace=True)
        prediction.fillna(self.median_, inplace=True)
        return prediction


if __name__ == "__main__":
    from itertools import chain

    orders = [
        ["H1_A", "H2_A", "H3_A", "A_y"],
        ["H1_B", "H2_B", "H3_B", "B_y"],
    ]
    temp_order = TemporalOrder(orders)

    columns = list(chain(*orders))

    data = {
        "H1_A": [1, 2, 3],
        "H2_A": [2, 3, 4],
        "H3_A": [3, 4, 5],
        "A_y": [4, 5, 6],
        "H1_B": [1, 1, 1],
        "H2_B": [2, 3, 4],
        "H3_B": [4, 9, 16],
        "B_y": [8, 27, 64],
    }
    targets = ["A_y", "B_y"]

    data = pd.DataFrame(data)
    X = data.drop(columns=targets)

    for name, model in baseline_models.items():
        print(name)
        prediction = pd.DataFrame(columns=targets)
        for target in targets:
            y = data[target]
            try:
                model_instance = model(temp_order=temp_order)
            except TypeError:
                model_instance = model()
            model_instance.fit(X, y)
            prediction[target] = model_instance.predict(X)
        print(prediction)
        print()
