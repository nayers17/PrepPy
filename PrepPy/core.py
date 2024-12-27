import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

class Preprocessor:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
        }
        self.encoder = OneHotEncoder()

    def scale(self, column: str, method: str = "standard") -> pd.DataFrame:
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in data.")
        if method not in self.scalers:
            raise ValueError(f"Scaling method '{method}' not supported.")

        scaler = self.scalers[method]
        self.data[column] = scaler.fit_transform(self.data[[column]])
        return self.data

    def encode(self, column: str) -> pd.DataFrame:
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in data.")

        encoded = self.encoder.fit_transform(self.data[[column]])
        encoded_df = pd.DataFrame(
            encoded.toarray(),
            columns=self.encoder.get_feature_names_out([column]),
        )
        self.data = self.data.drop(columns=[column])
        self.data = pd.concat([self.data, encoded_df], axis=1)
        return self.data

    def handle_missing(self, method: str = "mean") -> pd.DataFrame:
        if method == "mean":
            # Fill missing values for numeric columns
            for column in self.data.select_dtypes(include=["number"]).columns:
                self.data[column] = self.data[column].fillna(self.data[column].mean())
        elif method == "median":
            # Fill missing values for numeric columns
            for column in self.data.select_dtypes(include=["number"]).columns:
                self.data[column] = self.data[column].fillna(self.data[column].median())
        elif method == "mode":
            # Fill missing values for all columns with their mode
            for column in self.data.columns:
                self.data[column] = self.data[column].fillna(self.data[column].mode()[0])
        else:
            raise ValueError(f"Method '{method}' not supported for missing values.")
        return self.data


