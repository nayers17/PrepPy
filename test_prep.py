import pandas as pd
from PrepPy import Preprocessor

# Create a larger dataset with some outliers
data = pd.DataFrame({
    "age": [25, 30, 35, 40, 120, 45, 50, 55, 60, 65, 70, 75, 80, 85, 200],  # 120 and 200 are outliers
    "income": [50000, 60000, 70000, 80000, 1000000, 90000, 100000, 110000, 120000, 
               130000, 140000, 150000, 160000, 170000, 2000000],  # 1,000,000 and 2,000,000 are outliers
    "city": ["NY", "SF", "NY", "LA", "SF", "LA", "SF", "NY", "LA", "SF", "NY", "LA", "SF", "NY", "SF"]
})

print("Original Data:")
print(data)

# Initialize Preprocessor
preprocessor = Preprocessor(data)

# Test outlier detection
print("\nTesting with Z-score threshold of 2:")
outliers_age = preprocessor.identify_outliers(column="age", method="zscore", threshold=2)
outliers_income = preprocessor.identify_outliers(column="income", method="zscore", threshold=2)
print("Outliers detected in 'age' (Z-score):", outliers_age)
print("Outliers detected in 'income' (Z-score):", outliers_income)

print("\nTesting with IQR method:")
outliers_age_iqr = preprocessor.identify_outliers(column="age", method="iqr", threshold=1.5)
outliers_income_iqr = preprocessor.identify_outliers(column="income", method="iqr", threshold=1.5)
print("Outliers detected in 'age' (IQR):", outliers_age_iqr)
print("Outliers detected in 'income' (IQR):", outliers_income_iqr)

# Test outlier removal
print("\nRemoving outliers from 'age' using Z-score:")
data_cleaned = preprocessor.remove_outliers(column="age", method="zscore", threshold=2)
print("Data after removing outliers from 'age' using Z-score:")
print(data_cleaned)

print("\nRemoving outliers from 'income' using Z-score:")
data_cleaned = preprocessor.remove_outliers(column="income", method="zscore", threshold=2)
print("Data after removing outliers from 'income' using Z-score:")
print(data_cleaned)

print("\nRemoving outliers from 'income' using IQR:")
data_cleaned = preprocessor.remove_outliers(column="income", method="iqr", threshold=1.5)
print("Data after removing outliers from 'income' using IQR:")
print(data_cleaned)
