PrepPy

PrepPy is a Python library designed to simplify data preprocessing for machine learning and data science projects. It automates common preprocessing tasks, suggests best practices based on your dataset, and provides easy-to-use methods for scaling, encoding, and handling missing data.

Features

Automated Suggestions: Analyze your dataset and get suggestions for preprocessing steps.

Scaling: Support for StandardScaler and MinMaxScaler.

Encoding: One-hot encoding for categorical variables.

Handling Missing Values: Options to fill missing data with mean, median, or mode.

Customizable and Extensible: Add your own preprocessing steps as needed.

Installation

Install PrepPy using pip:

pip install PrepPy

Quick Start

Here’s a quick example of how to use PrepPy:

Example

import pandas as pd
from PrepPy import Preprocessor, suggest_steps

# Sample data
data = pd.DataFrame({
    "age": [25, 30, None, 35],
    "income": [50000, 60000, 70000, 80000],
    "city": ["NY", "SF", "NY", None],
})

# Step 1: Get preprocessing suggestions
print("Suggested steps:")
for step in suggest_steps(data):
    print(step)

# Step 2: Preprocess data
preprocessor = Preprocessor(data)
data = preprocessor.handle_missing()
data = preprocessor.scale("income", "minmax")
data = preprocessor.encode("city")

print("\nProcessed Data:")
print(data)

API Reference

Class: Preprocessor

Preprocessor(data: pd.DataFrame)

Initializes the Preprocessor with a pandas DataFrame.

Methods

handle_missing(method: str = "mean"): Handles missing values using the specified method (mean, median, or mode).

scale(column: str, method: str = "standard"): Scales the specified column using the chosen method (standard or minmax).

encode(column: str): Performs one-hot encoding on the specified column.

Function: suggest_steps(data: pd.DataFrame)

Analyzes the dataset and suggests preprocessing steps.

Testing

Run the test suite using pytest:

pytest tests/

Contributing

We welcome contributions! If you’d like to contribute, please:

Fork the repository.

Create a new branch for your feature or bugfix.

Submit a pull request with a detailed description of the changes.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments

PrepPy leverages the power of pandas and scikit-learn for efficient data processing. Special thanks to the open-source community for their inspiration and support.

