import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


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

    def inspect_dataset(self, verbose=True):
        """
        Inspect the dataset for key statistics, missing values, and outliers.

        Parameters:
            verbose (bool): If True, display intermediate calculations.

        Returns:
            pd.DataFrame: Summary report of the dataset inspection.
        """
        report = []
        for column in self.data.columns:
            col_type = self.data[column].dtype
            missing_values = self.data[column].isnull().sum()
            unique_values = (
                self.data[column].nunique() if col_type == "object" else "Not Applicable"
            )
            if col_type in ["int64", "float64"]:
                outliers_zscore = len(self.identify_outliers(column, method="zscore"))
                outliers_iqr = len(self.identify_outliers(column, method="iqr"))
                stats = {
                    "Mean": self.data[column].mean(),
                    "Median": self.data[column].median(),
                    "Min": self.data[column].min(),
                    "Max": self.data[column].max(),
                }
            else:
                outliers_zscore = outliers_iqr = stats = "Not Applicable"

            report.append({
                "Column": column,
                "Type": col_type,
                "Missing Values": missing_values,
                "Outliers (Z-Score)": outliers_zscore,
                "Outliers (IQR)": outliers_iqr,
                "Unique Values": unique_values,
                "Stats": stats,
            })

        report_df = pd.DataFrame(report)
        print("\nDataset Inspection Report:")
        print(report_df)
        return report_df

    
    def correlation_matrix(self, method: str = "pearson", visualize: bool = False):
        """
        Compute and display the correlation matrix for numeric columns.

        Parameters:
            method (str): Correlation method ('pearson', 'spearman', 'kendall').
            visualize (bool): If True, plot the correlation matrix as a heatmap (requires seaborn).

        Returns:
            pd.DataFrame: Correlation matrix.
        """
        numeric_data = self.data.select_dtypes(include=["float64", "int64"])
        corr_matrix = numeric_data.corr(method=method)

        if visualize:
            try:
                import seaborn as sns
                import matplotlib.pyplot as plt
                sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
                plt.title(f"Correlation Matrix ({method.capitalize()})")
                plt.show()
            except ImportError:
                print("Seaborn and Matplotlib are required for visualization.")
        
        if self.verbose:
            print(f"Correlation Matrix ({method.capitalize()}):\n{corr_matrix}")
        return corr_matrix
    
    def scale_columns(self, columns: list, method: str = "standard") -> pd.DataFrame:
        """
        Scale multiple numeric columns using StandardScaler or MinMaxScaler.

        Parameters:
            columns (list): List of columns to scale.
            method (str): Scaling method ('standard' or 'minmax').

        Returns:
            pd.DataFrame: Updated DataFrame with scaled columns.
        """
        for column in columns:
            self.scale(column, method)
        return self.data

    def split_data(self, target_column: str, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
        """
        Split the dataset into train, validation, and test sets.

        Parameters:
            target_column (str): The name of the target column for stratification.
            test_size (float): Proportion of the dataset to include in the test split.
            val_size (float): Proportion of the training dataset to include in the validation split.
            random_state (int): Random seed for reproducibility.

        Returns:
            tuple: Train, validation, and test datasets as pandas DataFrames.
        """
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")
        
        # Separate features and target
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]

        # Validate test size
        num_classes = len(y.unique())
        min_samples = max(1, num_classes)
        if len(y) * test_size < min_samples:
            raise ValueError(
                f"test_size is too small for stratification with {num_classes} classes. "
                f"Increase test_size to at least {min_samples / len(y):.2f}."
            )
        if len(y) * val_size < min_samples:
            raise ValueError(
                f"val_size is too small for stratification with {num_classes} classes. "
                f"Increase val_size to at least {min_samples / len(y):.2f}."
            )

        # Split data into train + validation and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        # Split train + validation set into train and validation sets
        val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size relative to remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, stratify=y_train_val, random_state=random_state
        )

        # Combine features and target back into DataFrames
        train = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
        val = pd.concat([X_val, y_val.reset_index(drop=True)], axis=1)
        test = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)

        return train, val, test