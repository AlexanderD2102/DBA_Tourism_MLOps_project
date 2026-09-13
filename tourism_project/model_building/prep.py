
# Import operating system utilities for creating output directories
import os

# Import pandas for loading, cleaning, and saving tabular data
import pandas as pd

# Import train_test_split for creating reproducible training and test datasets
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# 1. Define input and output locations
# ---------------------------------------------------------------------------

# Define the project-relative path to the registered raw tourism dataset
# A project-relative path allows the same script to run later in GitHub Actions
RAW_PATH = "tourism_project/data/tourism.csv"

# Store all prepared datasets in a dedicated processed-data directory
PROCESSED_DIR = "tourism_project/data/processed"

# Create the processed-data directory if it does not already exist
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 2. Load the registered raw dataset
# ---------------------------------------------------------------------------

# Read the raw tourism dataset created during the Data Registration step
df = pd.read_csv(RAW_PATH)

print(f"Raw dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")


# ---------------------------------------------------------------------------
# 3. Clean technical and inconsistent data
# ---------------------------------------------------------------------------

# Remove automatically generated CSV index columns such as 'Unnamed: 0'
# These columns only represent former row numbers and contain no predictive value
unnamed_columns = [
    column for column in df.columns
    if column.startswith("Unnamed")
]

if unnamed_columns:
    df = df.drop(columns=unnamed_columns)
    print(f"Removed technical index column(s): {unnamed_columns}")

# Standardize the inconsistent 'Fe Male' category to the intended 'Female' value
# This prevents the same gender category from being treated as two separate classes
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# Report missing values and duplicate rows as basic data-quality checks
# The supplied dataset is expected to contain neither missing values nor duplicates
missing_values = int(df.isnull().sum().sum())
duplicate_rows = int(df.duplicated().sum())

print(f"Missing values: {missing_values}")
print(f"Duplicate rows: {duplicate_rows}")


# ---------------------------------------------------------------------------
# 4. Remove identifier and post-contact leakage features
# ---------------------------------------------------------------------------

# CustomerID uniquely identifies customers but does not describe purchasing behavior
identifier_columns = ["CustomerID"]

# The business objective requires predicting purchase propensity BEFORE contacting
# a customer. These variables are generated during or after the sales interaction
# and therefore would not be available at the intended prediction time.
# Excluding them prevents unrealistic offline performance caused by data leakage.
post_contact_columns = [
    "DurationOfPitch",
    "NumberOfFollowups",
    "ProductPitched",
    "PitchSatisfactionScore",
]

# Remove identifier and post-contact variables from the modeling dataset
columns_to_drop = identifier_columns + post_contact_columns
df = df.drop(columns=columns_to_drop)

print(f"Removed non-production features: {columns_to_drop}")


# ---------------------------------------------------------------------------
# 5. Separate predictors and target variable
# ---------------------------------------------------------------------------

# ProdTaken is the binary target:
# 0 = customer did not purchase the tourism package
# 1 = customer purchased the tourism package
target = "ProdTaken"

# X contains all features available to the prediction model
X = df.drop(columns=[target])

# y contains the target variable that the model will learn to predict
y = df[target]


# ---------------------------------------------------------------------------
# 6. Create stratified training and test datasets
# ---------------------------------------------------------------------------

# Use 80% of the data for training and 20% for final testing
# random_state=42 ensures that the split is reproducible
# stratify=y preserves the approximately 81/19 class distribution in both sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------------------------------
# 7. Save prepared datasets for later pipeline stages
# ---------------------------------------------------------------------------

# Save feature and target datasets separately so that the model-training job
# can consume them directly in the automated GitHub Actions workflow
Xtrain.to_csv(f"{PROCESSED_DIR}/Xtrain.csv", index=False)
Xtest.to_csv(f"{PROCESSED_DIR}/Xtest.csv", index=False)
ytrain.to_csv(f"{PROCESSED_DIR}/ytrain.csv", index=False)
ytest.to_csv(f"{PROCESSED_DIR}/ytest.csv", index=False)


# ---------------------------------------------------------------------------
# 8. Report a compact preparation summary
# ---------------------------------------------------------------------------

print("\n✓ Data preparation completed successfully.")
print(f"✓ Training features: {Xtrain.shape}")
print(f"✓ Test features: {Xtest.shape}")
print(f"✓ Number of predictors: {Xtrain.shape[1]}")
print(f"✓ Training purchase rate: {ytrain.mean():.2%}")
print(f"✓ Test purchase rate: {ytest.mean():.2%}")
