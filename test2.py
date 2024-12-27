from PrepPy import Preprocessor, Pipeline
import pandas as pd

# Initialize data and preprocessor
data = pd.DataFrame({
    "age": [25, 30, None, 35, 120, 50, 45],
    "income": [50000, 60000, 70000, 80000, 1000000, 120000, None],
    "city": ["NY", "SF", "NY", "LA", "SF", None, "LA"],
    "target": [1, 0, 1, 0, 1, 0, 1]
})

preprocessor = Preprocessor(data)
pipeline = Pipeline()

# Add steps to pipeline
pipeline.add_step("handle_missing", method="mean")
pipeline.add_step("scale", column="age", method="minmax")
pipeline.add_step("scale", column="income", method="standard")
pipeline.add_step("encode", column="city")


# Run pipeline
processed_data = pipeline.run(data, preprocessor)
print(processed_data)

# Save pipeline
pipeline.save_pipeline("my_pipeline.txt")

# Load pipeline
new_pipeline = Pipeline()
new_pipeline.load_pipeline("my_pipeline.txt")
new_processed_data = new_pipeline.run(data, preprocessor)
print(new_processed_data)
