
# Import pandas for loading and validating the raw tourism dataset
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Define the registered dataset location
# ---------------------------------------------------------------------------

# Define the project-relative path to the raw tourism dataset
# Using a relative path is essential because this standalone script will later
# be executed automatically inside the GitHub Actions workflow
RAW_PATH = "tourism_project/data/tourism.csv"


# ---------------------------------------------------------------------------
# 2. Load the raw dataset
# ---------------------------------------------------------------------------

# Read the raw tourism dataset from the project data directory
df = pd.read_csv(RAW_PATH)


# ---------------------------------------------------------------------------
# 3. Validate the expected dataset schema
# ---------------------------------------------------------------------------

# Define all business, interaction, and target variables expected in the
# original tourism dataset before any data preparation is performed
expected_columns = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]

# Identify any required columns that are missing from the raw dataset
missing_columns = [
    column
    for column in expected_columns
    if column not in df.columns
]

# Stop the pipeline immediately if the expected schema is incomplete
# This prevents subsequent preparation or training steps from operating on
# incorrectly structured input data
if missing_columns:
    raise ValueError(
        f"Dataset is missing expected columns: {missing_columns}"
    )


# ---------------------------------------------------------------------------
# 4. Report successful data registration
# ---------------------------------------------------------------------------

print("✓ Dataset registered successfully.")
print(f"✓ Rows: {df.shape[0]:,}")
print(f"✓ Columns: {df.shape[1]}")

# Display the target distribution to document the original class imbalance
# ProdTaken = 0 means no purchase
# ProdTaken = 1 means purchase
print("\nProdTaken class distribution:")
print(
    df["ProdTaken"]
    .value_counts()
    .sort_index()
)
