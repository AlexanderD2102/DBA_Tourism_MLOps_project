#Use the project folder name, then the folder where model building code will be stored.

# Import pandas for loading and validating the raw tourism dataset
import pandas as pd

# Define the project-relative path to the raw dataset
# Using a relative path is important because the same script will later run
# automatically inside the GitHub Actions workflow
RAW_PATH = "tourism_project/data/tourism.csv"

# Load the raw tourism dataset from the project data directory
df = pd.read_csv(RAW_PATH)

# Define all columns that are required for the subsequent ML pipeline
# The dataset is only considered successfully registered if all required
# business and target variables are available
expected_columns = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]

# Identify any required columns that are missing from the raw dataset
missing = [column for column in expected_columns if column not in df.columns]

# Stop the pipeline immediately if required columns are missing
# This prevents later preprocessing or model-training steps from using
# an incomplete or incorrectly structured dataset
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

# Report successful registration and basic dataset information
print("✓ Dataset registered successfully.")
print(f"✓ Rows: {df.shape[0]:,}")
print(f"✓ Columns: {df.shape[1]}")

# Display the target distribution to document the class balance
# ProdTaken = 0 means no purchase; ProdTaken = 1 means purchase
print("\nProdTaken class distribution:")
print(df["ProdTaken"].value_counts().sort_index())
