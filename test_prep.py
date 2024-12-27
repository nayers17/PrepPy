from PrepPy import Preprocessor
import pandas as pd

# Create a dataset with intentional outliers
data = pd.DataFrame({
    "age": [25, 30, 35, 40, 120],  # 120 is an outlier
    "income": [50000, 60000, 70000, 80000, 1000000],  # 1000000 is an outlier
    "city": ["NY", "SF", "NY", "LA", "SF"],
})

print("Original Data:")
print(data)

# Initialize Preprocessor
preprocessor = Preprocessor(data)

# --- Test Outlier Detection ---
print("\nTesting with Z-score threshold of 1.5:")
outliers_age = preprocessor.identify_outliers(column="age", method="zscore", threshold=1.5)
outliers_income = preprocessor.identify_outliers(column="income", method="zscore", threshold=1.5)
print("Outliers detected in 'age' (Z-score):", outliers_age)
print("Outliers detected in 'income' (Z-score):", outliers_income)

print("\nTesting with IQR method:")
outliers_age_iqr = preprocessor.identify_outliers(column="age", method="iqr", threshold=1.5)
outliers_income_iqr = preprocessor.identify_outliers(column="income", method="iqr", threshold=1.5)
print("Outliers detected in 'age' (IQR):", outliers_age_iqr)
print("Outliers detected in 'income' (IQR):", outliers_income_iqr)

# --- Test Outlier Removal ---
print("\nRemoving outliers from 'age' using Z-score with threshold 1.5:")
data_cleaned = preprocessor.remove_outliers(column="age", method="zscore", threshold=1.5)
print("Data after removing outliers from 'age' using Z-score:")
print(data_cleaned)

print("\nRemoving outliers from 'income' using Z-score with threshold 1.5:")
data_cleaned = preprocessor.remove_outliers(column="income", method="zscore", threshold=1.5)
print("Data after removing outliers from 'income' using Z-score:")
print(data_cleaned)

print("\nRemoving outliers from 'income' using IQR:")
data_cleaned = preprocessor.remove_outliers(column="income", method="iqr", threshold=1.5)
print("Data after removing outliers from 'income' using IQR:")
print(data_cleaned)
