import pandas as pd
import numpy as np
from PrepPy import Preprocessor

# Sample Dataset with Various Cases
data = pd.DataFrame({
    "age": [25, 30, None, 35, 120, 50, 45],  # Contains missing values and an outlier
    "income": [50000, 60000, 70000, 80000, 1000000, 120000, None],  # Contains missing values and an outlier
    "city": ["NY", "SF", "NY", "LA", "SF", None, "LA"],  # Contains missing values
    "target": [1, 0, 1, 0, 1, 0, 1]  # Binary target for stratified split testing
})

print("Original Data:")
print(data)

# Initialize Preprocessor
preprocessor = Preprocessor(data)

# Test Missing Value Handling
print("\nHandling Missing Values:")
data_handled = preprocessor.handle_missing(method="mean")
print(data_handled)

# Test Scaling
print("\nScaling 'age' and 'income':")
data_scaled = preprocessor.scale("age", method="minmax")
data_scaled = preprocessor.scale("income", method="standard")
print(data_scaled)

# Test Encoding
print("\nEncoding 'city':")
data_encoded = preprocessor.encode("city")
print(data_encoded)

# Test Outlier Detection
print("\nOutlier Detection:")
outliers_age_zscore = preprocessor.identify_outliers("age", method="zscore", threshold=2)
outliers_income_iqr = preprocessor.identify_outliers("income", method="iqr", threshold=1.5)
print(f"Outliers in 'age' (Z-score): {outliers_age_zscore}")
print(f"Outliers in 'income' (IQR): {outliers_income_iqr}")

# Test Outlier Removal
print("\nRemoving Outliers from 'age':")
data_outliers_removed = preprocessor.remove_outliers("age", method="zscore", threshold=2)
print(data_outliers_removed)

# Test Dataset Inspection
print("\nDataset Inspection:")
inspection_report = preprocessor.inspect_dataset()
print(inspection_report)

# Test Correlation Matrix
print("\nCorrelation Matrix:")
correlation_matrix = preprocessor.correlation_matrix(method="pearson", visualize=False)
print(correlation_matrix)

# Duplicate the dataset to increase size
data = pd.concat([data] * 3, ignore_index=True)
print(f"New dataset size: {data.shape[0]} rows")

# Increase dataset size by duplicating rows multiple times
data = pd.concat([data] * 5, ignore_index=True)
print(f"New dataset size: {data.shape[0]} rows")

# Reinitialize the Preprocessor with the larger dataset
preprocessor = Preprocessor(data)

# Re-run the split
train, val, test = preprocessor.split_data(target_column="target", test_size=0.4, val_size=0.2)

print(f"Train set size: {train.shape[0]}")
print(f"Validation set size: {val.shape[0]}")
print(f"Test set size: {test.shape[0]}")

print(train['target'].value_counts())
print(val['target'].value_counts())
print(test['target'].value_counts())
