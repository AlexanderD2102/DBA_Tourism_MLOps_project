
# ---------------------------------------------------------------------------
# Import required libraries
# ---------------------------------------------------------------------------

# Import operating system utilities for paths, folders, and environment variables
import os

# Import JSON for storing model metadata in a human-readable format
import json

# Import pandas for loading the prepared training and test datasets
import pandas as pd

# Import joblib for serializing the complete trained ML pipeline
import joblib

# Import MLflow for experiment tracking in the production workflow
import mlflow

# Import XGBoost for binary classification
import xgboost as xgb

# Import preprocessing and pipeline utilities
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Import GridSearchCV for reproducible hyperparameter optimization
from sklearn.model_selection import GridSearchCV

# Import evaluation metrics suitable for the imbalanced binary target
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ---------------------------------------------------------------------------
# 1. Define project paths
# ---------------------------------------------------------------------------

# Directory containing the production-safe train/test datasets
# created during the Data Preparation pipeline step
PROCESSED_DIR = "tourism_project/data/processed"

# Directory used by the deployed Streamlit application
DEPLOYMENT_DIR = "tourism_project/deployment"

# Define the location where the final trained pipeline will be stored
MODEL_PATH = f"{DEPLOYMENT_DIR}/best_model.joblib"

# Store model metrics and configuration separately for transparency
METRICS_PATH = f"{DEPLOYMENT_DIR}/model_metrics.json"

# Create the deployment directory if it does not already exist
os.makedirs(DEPLOYMENT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 2. Configure MLflow for the production environment
# ---------------------------------------------------------------------------

# In GitHub Actions, MLflow runs locally on port 5000.
# The environment variable allows the URI to be changed without modifying code.
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

# Use the same experiment name as in the development environment
# so that the project remains conceptually consistent
EXPERIMENT_NAME = "Tourism_Package_Purchase_Prediction"

# Configure MLflow to use the local tracking server
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Create or reuse the project experiment
mlflow.set_experiment(EXPERIMENT_NAME)


# ---------------------------------------------------------------------------
# 3. Load the prepared production-safe datasets
# ---------------------------------------------------------------------------

# Load training and test predictor datasets
Xtrain = pd.read_csv(f"{PROCESSED_DIR}/Xtrain.csv")
Xtest = pd.read_csv(f"{PROCESSED_DIR}/Xtest.csv")

# Load the corresponding binary target datasets and convert them to Series
ytrain = pd.read_csv(f"{PROCESSED_DIR}/ytrain.csv").squeeze("columns")
ytest = pd.read_csv(f"{PROCESSED_DIR}/ytest.csv").squeeze("columns")

# Report the loaded dataset dimensions for pipeline traceability
print("✓ Production-safe datasets loaded successfully.")
print(f"✓ Xtrain: {Xtrain.shape}")
print(f"✓ Xtest:  {Xtest.shape}")
print(f"✓ ytrain: {ytrain.shape}")
print(f"✓ ytest:  {ytest.shape}")


# ---------------------------------------------------------------------------
# 4. Identify numerical and categorical predictors
# ---------------------------------------------------------------------------

# Automatically determine numerical predictors from pandas data types
numeric_features = Xtrain.select_dtypes(
    include=["number"]
).columns.tolist()

# Automatically determine categorical predictors stored as strings/objects
categorical_features = Xtrain.select_dtypes(
    include=["object"]
).columns.tolist()

print(f"✓ Numerical predictors: {len(numeric_features)}")
print(f"✓ Categorical predictors: {len(categorical_features)}")
print(f"✓ Total predictors: {Xtrain.shape[1]}")


# ---------------------------------------------------------------------------
# 5. Calculate class weight
# ---------------------------------------------------------------------------

# Count negative cases: customers who did not purchase the package
negative_cases = int((ytrain == 0).sum())

# Count positive cases: customers who purchased the package
positive_cases = int((ytrain == 1).sum())

# Calculate scale_pos_weight to compensate for the class imbalance
class_weight = negative_cases / positive_cases

print(f"✓ scale_pos_weight: {class_weight:.3f}")


# ---------------------------------------------------------------------------
# 6. Build the preprocessing pipeline
# ---------------------------------------------------------------------------

# Standardize numerical features and one-hot encode categorical variables.
# handle_unknown="ignore" ensures that inference does not fail if a valid
# categorical value appears that was not represented in the training subset.
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)


# ---------------------------------------------------------------------------
# 7. Define the XGBoost classifier
# ---------------------------------------------------------------------------

# Configure the base classifier with class weighting and reproducible settings
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=class_weight,
    random_state=42,
    tree_method="hist",
    n_jobs=1
)


# ---------------------------------------------------------------------------
# 8. Define the hyperparameter search space
# ---------------------------------------------------------------------------

# Use the same focused search space validated in the development environment
param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__colsample_bytree": [0.8, 1.0],
    "xgbclassifier__colsample_bylevel": [0.8, 1.0],
    "xgbclassifier__learning_rate": [0.05, 0.10],
    "xgbclassifier__reg_lambda": [1.0, 5.0],
}


# ---------------------------------------------------------------------------
# 9. Combine preprocessing and classification into one pipeline
# ---------------------------------------------------------------------------

# Saving preprocessing and model together ensures that Streamlit applies
# exactly the same transformations that were used during model training
model_pipeline = make_pipeline(
    preprocessor,
    xgb_model
)


# ---------------------------------------------------------------------------
# 10. Run hyperparameter tuning and MLflow tracking
# ---------------------------------------------------------------------------

# Start a parent MLflow run representing the complete production training job
with mlflow.start_run(run_name="Production_XGBoost_GridSearch"):

    # Record experiment-level configuration
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("optimization_metric", "roc_auc")
    mlflow.log_param("scale_pos_weight", float(class_weight))
    mlflow.log_param("production_safe_features", True)
    mlflow.log_param("number_of_predictors", Xtrain.shape[1])

    # Configure hyperparameter tuning
    # ROC-AUC is used because the target is substantially imbalanced
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        refit=True,
        verbose=1,
        return_train_score=False
    )

    # Train all candidate configurations using the training dataset only
    grid_search.fit(Xtrain, ytrain)

    # Retrieve cross-validation results for every tested parameter combination
    results = grid_search.cv_results_

    # Log every hyperparameter combination as a nested MLflow run
    # so that individual experiments can be compared later
    for i in range(len(results["params"])):

        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]
        std_score = results["std_test_score"][i]

        with mlflow.start_run(
            run_name=f"Production_Grid_Combination_{i + 1:02d}",
            nested=True
        ):
            # Log the exact hyperparameter combination
            mlflow.log_params(param_set)

            # Log mean and variation of the cross-validation ROC-AUC
            mlflow.log_metric(
                "mean_cv_roc_auc",
                float(mean_score)
            )

            mlflow.log_metric(
                "std_cv_roc_auc",
                float(std_score)
            )


    # -----------------------------------------------------------------------
    # 11. Select and evaluate the best model
    # -----------------------------------------------------------------------

    # Retrieve the automatically refitted model associated with
    # the highest cross-validation ROC-AUC
    best_model = grid_search.best_estimator_

    # Log the selected hyperparameters to the parent run
    mlflow.log_params(grid_search.best_params_)

    # Log the best cross-validation performance
    mlflow.log_metric(
        "best_cv_roc_auc",
        float(grid_search.best_score_)
    )

    # Keep the transparent baseline threshold validated during development
    classification_threshold = 0.50

    mlflow.log_param(
        "classification_threshold",
        classification_threshold
    )

    # Generate purchase probabilities for training and untouched test data
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]

    # Convert probabilities into binary classifications
    y_pred_train = (
        y_pred_train_proba >= classification_threshold
    ).astype(int)

    y_pred_test = (
        y_pred_test_proba >= classification_threshold
    ).astype(int)


    # -----------------------------------------------------------------------
    # 12. Calculate final model metrics
    # -----------------------------------------------------------------------

    # Evaluate model performance on the training dataset
    train_metrics = {
        "accuracy": float(
            accuracy_score(ytrain, y_pred_train)
        ),
        "precision": float(
            precision_score(ytrain, y_pred_train, zero_division=0)
        ),
        "recall": float(
            recall_score(ytrain, y_pred_train, zero_division=0)
        ),
        "f1": float(
            f1_score(ytrain, y_pred_train, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(ytrain, y_pred_train_proba)
        ),
    }

    # Evaluate generalization performance on the untouched test dataset
    test_metrics = {
        "accuracy": float(
            accuracy_score(ytest, y_pred_test)
        ),
        "precision": float(
            precision_score(ytest, y_pred_test, zero_division=0)
        ),
        "recall": float(
            recall_score(ytest, y_pred_test, zero_division=0)
        ),
        "f1": float(
            f1_score(ytest, y_pred_test, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(ytest, y_pred_test_proba)
        ),
    }

    # Log all training metrics to MLflow
    for metric_name, metric_value in train_metrics.items():
        mlflow.log_metric(
            f"train_{metric_name}",
            metric_value
        )

    # Log all test metrics to MLflow
    for metric_name, metric_value in test_metrics.items():
        mlflow.log_metric(
            f"test_{metric_name}",
            metric_value
        )


    # -----------------------------------------------------------------------
    # 13. Save the deployment artifacts
    # -----------------------------------------------------------------------

    # Serialize the COMPLETE pipeline, including preprocessing and XGBoost.
    # Streamlit can therefore send raw user inputs directly to this object.
    joblib.dump(
        best_model,
        MODEL_PATH
    )

    # Store additional metadata required for transparent deployment
    model_metadata = {
        "model_type": "XGBoost",
        "classification_threshold": classification_threshold,
        "best_cv_roc_auc": float(grid_search.best_score_),
        "best_parameters": grid_search.best_params_,
        "training_metrics": train_metrics,
        "test_metrics": test_metrics,
        "predictor_columns": Xtrain.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "excluded_features": [
            "CustomerID",
            "DurationOfPitch",
            "NumberOfFollowups",
            "ProductPitched",
            "PitchSatisfactionScore"
        ]
    }

    # Save metadata next to the trained model
    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as metadata_file:
        json.dump(
            model_metadata,
            metadata_file,
            indent=4
        )

    # Log both deployment artifacts to the production MLflow run
    mlflow.log_artifact(
        MODEL_PATH,
        artifact_path="deployment"
    )

    mlflow.log_artifact(
        METRICS_PATH,
        artifact_path="deployment"
    )


# ---------------------------------------------------------------------------
# 14. Report the production-training result
# ---------------------------------------------------------------------------

print("\n✓ Production model training completed successfully.")
print(f"✓ Best CV ROC-AUC: {grid_search.best_score_:.4f}")
print(f"✓ Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
print(f"✓ Test Recall: {test_metrics['recall']:.4f}")
print(f"✓ Test Precision: {test_metrics['precision']:.4f}")
print(f"✓ Test F1-score: {test_metrics['f1']:.4f}")
print(f"✓ Model saved to: {MODEL_PATH}")
print(f"✓ Metadata saved to: {METRICS_PATH}")
