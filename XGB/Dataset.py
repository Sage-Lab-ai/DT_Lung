import pandas as pd
import numpy as np
from typing import Iterable
import os


class Dataset:
    X_FILE = "X.csv"
    Y_FILE = "y.csv"

    def __init__(self):
        self.df = pd.DataFrame()
        self.x_columns = pd.Index([])
        self.y_columns = pd.Index([])

    def create_from_df(self,
                       data: pd.DataFrame,
                       target_columns: Iterable[str],
                       drop_x_na: bool = True,
                       verbose: bool = False) -> 'Dataset':
        """
        Initialize the Dataset object by loading a DataFrame.

        :param data: input DataFrame
        :param target_columns: target variables
        :param drop_x_na: drop rows with missing input features
        :param verbose: print additional information
        """
        self.df = data

        if verbose:
            print(f"Shape of the original dataframe: {self.df.shape}\n")

        self.y_columns = self.df.columns.intersection(target_columns)
        self.x_columns = self.df.columns.difference(self.y_columns)

        if drop_x_na:
            self.df.dropna(subset=self.x_columns, inplace=True)
            if verbose:
                print(f"Shape of the dataframe after dropping rows with missing input features: {self.df.shape}\n")

        if self.df.shape[0] == 0:
            print("[WARNING] No data available after filtering. Please check the filtering criteria.\n")

        non_numeric_columns = self.df.select_dtypes(exclude=np.number).columns
        if non_numeric_columns.size > 0:
            self.df[non_numeric_columns] = self.df[non_numeric_columns].astype("category")
            if verbose:
                print(f"Categorized {non_numeric_columns.size} non-numeric columns: {non_numeric_columns.tolist()}\n")

        return self

    def get_X(self) -> pd.DataFrame:
        """
        Get the features of the dataset.
        :return: features of the dataset as a DataFrame
        """
        return self.df[self.x_columns]

    def get_Y(self) -> pd.DataFrame:
        """
        Get the target variables of the dataset.
        :return: target variables of the dataset as a DataFrame
        """
        return self.df[self.y_columns]

    def get_X_y(self, target_column: str, drop_missing_y: bool = True) -> tuple[pd.DataFrame, pd.Series]:
        """
        Split the dataset into features and a single target variable.

        :param target_column: name of the target variable
        :param drop_missing_y: If True, drop rows with missing target variable

        :return: tuple of features and target variable
        """
        X = self.df[(self.x_columns.difference([target_column]))]
        y = self.df[target_column]

        if drop_missing_y:
            valid_indices = y.notna()
            X = X.loc[valid_indices, :]
            y = y.loc[valid_indices]

        return X, y

    def save(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        self.df[self.x_columns].to_csv(os.path.join(folder, self.X_FILE))
        self.df[self.y_columns].to_csv(os.path.join(folder, self.Y_FILE))

    def load(self, folder: str):
        x_df = pd.read_csv(os.path.join(folder, self.X_FILE), index_col=0)
        y_df = pd.read_csv(os.path.join(folder, self.Y_FILE), index_col=0)
        self.df = pd.concat([x_df, y_df], axis=1)
        self.x_columns = x_df.columns
        self.y_columns = y_df.columns
        return self