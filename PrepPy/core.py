import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder


class Preprocessor:
    def __init__(self, data: pd.DataFrame, verbose: bool = True):
        """
        Initialize the Preprocessor class.

        Parameters:
            data (pd.DataFrame): The dataset to preprocess.
            verbose (bool): Whether to print debug information (default: True).
        """
        self.data = data
        self.verbose = verbose
        self.scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
        }
        self.encoder = OneHotEncoder()

    def scale(self, column: str, method: str = "standard") -> pd.DataFrame:
        """
        Scale a numeric column using StandardScaler or MinMaxScaler.

        Parameters:
            column (str): The column to scale.
            method (str): Scaling method ('standard' or 'minmax').

        Returns:
            pd.DataFrame: Updated DataFrame with scaled column.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in data.")
        if method not in self.scalers:
            raise ValueError(f"Scaling method '{method}' not supported.")

        scaler = self.scalers[method]
        self.data[column] = scaler.fit_transform(self.data[[column]])
        if self.verbose:
            print(f"Scaled '{column}' using {method} scaling.")
        return self.data

    def encode(self, column: str) -> pd.DataFrame:
        """
        One-hot encode a categorical column.

        Parameters:
            column (str): The column to encode.

        Returns:
            pd.DataFrame: Updated DataFrame with encoded columns.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in data.")

        encoded = self.encoder.fit_transform(self.data[[column]])
        encoded_df = pd.DataFrame(
            encoded.toarray(),
            columns=self.encoder.get_feature_names_out([column]),
        )
        self.data = self.data.drop(columns=[column])
        self.data = pd.concat([self.data, encoded_df], axis=1)
        if self.verbose:
            print(f"Encoded '{column}' using OneHotEncoder.")
        return self.data

    def handle_missing(self, method: str = "mean") -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Parameters:
            method (str): Method to handle missing values ('mean', 'median', 'mode').

        Returns:
            pd.DataFrame: Updated DataFrame with missing values handled.
        """
        if method == "mean":
            for column in self.data.select_dtypes(include=["number"]).columns:
                self.data[column] = self.data[column].fillna(self.data[column].mean())
        elif method == "median":
            for column in self.data.select_dtypes(include=["number"]).columns:
                self.data[column] = self.data[column].fillna(self.data[column].median())
        elif method == "mode":
            for column in self.data.columns:
                self.data[column] = self.data[column].fillna(self.data[column].mode()[0])
        else:
            raise ValueError(f"Method '{method}' not supported for missing values.")
        if self.verbose:
            print(f"Handled missing values using {method} method.")
        return self.data

    def identify_outliers(self, column: str, method: str = "zscore", threshold: float = 3) -> list:
        """
        Identify outliers in a numeric column using Z-score or IQR.

        Parameters:
            column (str): The column to check for outliers.
            method (str): Method to use ('zscore' or 'iqr').
            threshold (float): Threshold for identifying outliers.

        Returns:
            list: Indices of rows that are outliers.
        """
        if column not in self.data.columns or self.data[column].dtype not in ["int64", "float64"]:
            raise ValueError(f"Column '{column}' must be numeric and exist in the dataset.")

        if method == "zscore":
            mean = self.data[column].mean()
            std = self.data[column].std()
            z_scores = (self.data[column] - mean) / std
            if self.verbose:
                print(f"Z-scores for column '{column}':\n{z_scores}")
            return self.data[np.abs(z_scores) > threshold].index.tolist()
        elif method == "iqr":
            Q1 = self.data[column].quantile(0.25)
            Q3 = self.data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            if self.verbose:
                print(f"IQR bounds for column '{column}': Lower={lower_bound}, Upper={upper_bound}")
            return self.data[(self.data[column] < lower_bound) | (self.data[column] > upper_bound)].index.tolist()
        else:
            raise ValueError("Invalid method. Choose 'zscore' or 'iqr'.")

    def remove_outliers(self, column: str, method: str = "zscore", threshold: float = 3) -> pd.DataFrame:
        """
        Remove outliers from a column using Z-score or IQR.

        Parameters:
            column (str): The column to clean.
            method (str): Method to use ('zscore' or 'iqr').
            threshold (float): Threshold for identifying outliers.

        Returns:
            pd.DataFrame: DataFrame with outliers removed.
        """
        outlier_indices = self.identify_outliers(column, method, threshold)
        if self.verbose:
            print(f"Removing outliers at indices: {outlier_indices}")
        self.data = self.data.drop(index=outlier_indices).reset_index(drop=True)
        return self.data

    def remove_outliers_iterative(self, column: str, method: str = "zscore", threshold: float = 1.5) -> pd.DataFrame:
        """
        Iteratively remove outliers from a column using Z-score or IQR until no more outliers are detected.

        Parameters:
            column (str): The column to clean.
            method (str): Method to use ('zscore' or 'iqr').
            threshold (float): Threshold for identifying outliers.

        Returns:
            pd.DataFrame: DataFrame with all outliers removed.
        """
        while True:
            outliers = self.identify_outliers(column, method, threshold)
            if not outliers:
                break
            if self.verbose:
                print(f"Removing outliers at indices: {outliers}")
            self.data = self.data.drop(index=outliers).reset_index(drop=True)
        return self.data
