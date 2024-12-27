import pandas as pd
from PrepPy import Preprocessor, suggest_steps

# Sample dataset
data = pd.DataFrame({
    "age": [25, None, 30],
    "income": [50000, 60000, 70000],
    "city": ["NY", "SF", None],
})

# Step 1: Get suggestions
print("Suggested Steps:")
steps = suggest_steps(data)
for step in steps:
    print(step)

# Step 2: Preprocess
preprocessor = Preprocessor(data)
data = preprocessor.handle_missing()
data = preprocessor.scale("income", method="minmax")
data = preprocessor.encode("city")

print("\nProcessed Data:")
print(data)
