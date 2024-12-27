import pandas as pd
from PrepPy import Preprocessor, suggest_steps

# Create a sample dataset
data = pd.DataFrame({
    "age": [25, 30, None, 35],
    "income": [50000, 60000, 70000, 80000],
    "city": ["NY", "SF", "NY", None],
})

# Display original data
print("Original Data:")
print(data)

# Step 1: Suggest preprocessing steps
print("\nSuggested Steps:")
steps = suggest_steps(data)
for step in steps:
    print(f"- {step}")

# Step 2: Preprocess data
preprocessor = Preprocessor(data)

# Handle missing values
data = preprocessor.handle_missing()
print("\nData After Handling Missing Values:")
print(data)

# Scale the 'income' column
data = preprocessor.scale("income", method="minmax")
print("\nData After Scaling 'income':")
print(data)

# Encode the 'city' column
data = preprocessor.encode("city")
print("\nData After Encoding 'city':")
print(data)
