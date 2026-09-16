
# ---------------------------------------------------------------------------
# Step 4.1 - Import required libraries
# ---------------------------------------------------------------------------

# Import operating system utilities for constructing file paths
import os

# Import JSON to load the model metadata generated during production training
import json

# Import pandas to create the single-row input DataFrame expected by the model
import pandas as pd

# Import joblib to load the serialized preprocessing-and-model pipeline
import joblib

# Import Streamlit to build the interactive web application
import streamlit as st


# ---------------------------------------------------------------------------
# Step 4.2 - Configure the Streamlit page
# ---------------------------------------------------------------------------

# Configure the browser tab and use a wide layout for a clearer input form
st.set_page_config(
    page_title="Tourism Package Purchase Prediction",
    page_icon="✈️",
    layout="wide"
)


# ---------------------------------------------------------------------------
# Step 4.3 - Define model and metadata locations
# ---------------------------------------------------------------------------

# app.py, best_model.joblib, and model_metrics.json are stored in the same
# deployment directory. Using __file__ makes the paths independent of the
# working directory used by Streamlit Community Cloud or Docker.
APP_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    APP_DIR,
    "best_model.joblib"
)

METADATA_PATH = os.path.join(
    APP_DIR,
    "model_metrics.json"
)


# ---------------------------------------------------------------------------
# Step 4.4 - Load the trained production artifacts
# ---------------------------------------------------------------------------

# Cache the model so that Streamlit does not reload it from disk
# every time the user changes an input value.
@st.cache_resource
def load_model():
    """Load the complete preprocessing-and-XGBoost production pipeline."""

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Production model not found at: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def load_metadata():
    """Load metadata generated together with the production model."""

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"Model metadata not found at: {METADATA_PATH}"
        )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as metadata_file:
        return json.load(metadata_file)


# Attempt to load both deployment artifacts
try:
    model = load_model()
    metadata = load_metadata()

except Exception as error:
    st.error(
        "The production model could not be loaded. "
        "Please verify that the deployment artifacts are available."
    )
    st.exception(error)
    st.stop()


# ---------------------------------------------------------------------------
# Step 4.5 - Retrieve deployment configuration
# ---------------------------------------------------------------------------

# Load the classification threshold selected during production training
classification_threshold = metadata.get(
    "classification_threshold",
    0.50
)

# Retrieve the exact predictor order used during model training
predictor_columns = metadata.get(
    "predictor_columns",
    []
)


# ---------------------------------------------------------------------------
# Step 4.6 - Application header and business explanation
# ---------------------------------------------------------------------------

st.title(
    "✈️ Tourism Package Purchase Prediction"
)

st.write(
    """
    This application estimates the likelihood that a customer will purchase
    the **Wellness Tourism Package before being contacted**.

    The prediction uses only customer information that is realistically
    available before the sales interaction. Variables generated during or
    after a sales pitch are intentionally excluded from the production model.
    """
)


# Provide additional model information without cluttering the main interface
with st.expander(
    "About the production model"
):

    st.write(
        """
        The application uses an **XGBoost classification model** combined with
        the same preprocessing pipeline used during training. Numerical
        variables are processed automatically and categorical variables are
        one-hot encoded inside the saved model pipeline.
        """
    )

    st.write(
        f"**Classification threshold:** "
        f"{classification_threshold:.2f}"
    )

    # Display the stored test ROC-AUC when available
    test_roc_auc = (
        metadata
        .get("test_metrics", {})
        .get("roc_auc")
    )

    if test_roc_auc is not None:
        st.write(
            f"**Test ROC-AUC:** "
            f"{test_roc_auc:.4f}"
        )


# ---------------------------------------------------------------------------
# Step 4.7 - Collect customer information
# ---------------------------------------------------------------------------

st.subheader(
    "Customer Information"
)

st.caption(
    "Enter the customer information available before the sales contact."
)


# Use a form so that the model only runs after the user explicitly
# presses the prediction button
with st.form(
    "prediction_form"
):

    # Divide the inputs into two columns to keep the interface compact
    left_column, right_column = st.columns(2)


    # -----------------------------------------------------------------------
    # Step 4.7.1 - Left-column predictors
    # -----------------------------------------------------------------------

    with left_column:

        Age = st.slider(
            "Age",
            min_value=18,
            max_value=61,
            value=36
        )

        TypeofContact = st.selectbox(
            "Type of Contact",
            [
                "Self Enquiry",
                "Company Invited"
            ]
        )

        CityTier = st.selectbox(
            "City Tier",
            [1, 2, 3]
        )

        Occupation = st.selectbox(
            "Occupation",
            [
                "Salaried",
                "Small Business",
                "Large Business",
                "Free Lancer"
            ]
        )

        # The inconsistent raw category 'Fe Male' was normalized
        # to 'Female' during data preparation.
        Gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female"
            ]
        )

        NumberOfPersonVisiting = st.slider(
            "Number of Persons Visiting",
            min_value=1,
            max_value=5,
            value=3
        )

        PreferredPropertyStar = st.selectbox(
            "Preferred Property Star Rating",
            [3, 4, 5]
        )


    # -----------------------------------------------------------------------
    # Step 4.7.2 - Right-column predictors
    # -----------------------------------------------------------------------

    with right_column:

        MaritalStatus = st.selectbox(
            "Marital Status",
            [
                "Married",
                "Single",
                "Divorced",
                "Unmarried"
            ]
        )

        NumberOfTrips = st.number_input(
            "Number of Trips per Year",
            min_value=1,
            max_value=22,
            value=3,
            step=1
        )

        Passport_label = st.selectbox(
            "Has Passport?",
            [
                "No",
                "Yes"
            ]
        )

        Passport = (
            1
            if Passport_label == "Yes"
            else 0
        )

        OwnCar_label = st.selectbox(
            "Owns a Car?",
            [
                "No",
                "Yes"
            ]
        )

        OwnCar = (
            1
            if OwnCar_label == "Yes"
            else 0
        )

        NumberOfChildrenVisiting = st.slider(
            "Number of Children Visiting",
            min_value=0,
            max_value=3,
            value=1
        )

        Designation = st.selectbox(
            "Designation",
            [
                "Executive",
                "Manager",
                "Senior Manager",
                "AVP",
                "VP"
            ]
        )

        MonthlyIncome = st.number_input(
            "Monthly Income",
            min_value=1000.0,
            max_value=100000.0,
            value=22418.0,
            step=500.0
        )


    # Create the prediction button at the bottom of the form
    predict_button = st.form_submit_button(
        "Predict Purchase Likelihood",
        use_container_width=True
    )


# ---------------------------------------------------------------------------
# Step 4.8 - Prepare model input
# ---------------------------------------------------------------------------

if predict_button:

    # Create a single-row DataFrame containing exactly the 14
    # production-safe predictors used during model training.
    #
    # The excluded post-contact variables are deliberately absent:
    # - DurationOfPitch
    # - NumberOfFollowups
    # - ProductPitched
    # - PitchSatisfactionScore
    input_data = pd.DataFrame(
        [{
            "Age": Age,
            "TypeofContact": TypeofContact,
            "CityTier": CityTier,
            "Occupation": Occupation,
            "Gender": Gender,
            "NumberOfPersonVisiting": NumberOfPersonVisiting,
            "PreferredPropertyStar": PreferredPropertyStar,
            "MaritalStatus": MaritalStatus,
            "NumberOfTrips": NumberOfTrips,
            "Passport": Passport,
            "OwnCar": OwnCar,
            "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
            "Designation": Designation,
            "MonthlyIncome": MonthlyIncome
        }]
    )


    # -----------------------------------------------------------------------
    # Step 4.9 - Validate the production schema
    # -----------------------------------------------------------------------

    # Reorder the input columns to exactly match the schema saved during
    # production training.
    if predictor_columns:

        missing_predictors = [
            column
            for column in predictor_columns
            if column not in input_data.columns
        ]

        if missing_predictors:

            st.error(
                "Prediction cannot be generated because required "
                f"features are missing: {missing_predictors}"
            )

            st.stop()

        # Match the exact feature order used during production training
        input_data = input_data[
            predictor_columns
        ]


    # -----------------------------------------------------------------------
    # Step 4.10 - Generate purchase probability
    # -----------------------------------------------------------------------

    try:

        # ProdTaken = 1 represents purchase of the tourism package
        purchase_probability = float(
            model.predict_proba(
                input_data
            )[0, 1]
        )

        # Apply exactly the same classification threshold
        # stored during production model training
        prediction = int(
            purchase_probability
            >= classification_threshold
        )


        # -------------------------------------------------------------------
        # Step 4.11 - Display the prediction result
        # -------------------------------------------------------------------

        st.divider()

        st.subheader(
            "Prediction Result"
        )

        # Show probability prominently because it is more informative
        # for prioritization than the binary prediction alone
        st.metric(
            "Estimated Purchase Probability",
            f"{purchase_probability:.1%}"
        )

        # Visual probability indicator
        st.progress(
            min(
                max(
                    purchase_probability,
                    0.0
                ),
                1.0
            )
        )

        # Display classification in business-oriented language
        if prediction == 1:

            st.success(
                "This customer is classified as a potential buyer "
                "and may be prioritized for marketing contact."
            )

        else:

            st.info(
                "This customer is currently classified as less likely "
                "to purchase the tourism package."
            )

        # Explain how the classification was derived
        st.caption(
            f"Classification threshold: "
            f"{classification_threshold:.0%}"
        )


    # Display inference errors in the interface instead of crashing the app
    except Exception as error:

        st.error(
            "An error occurred while generating the prediction."
        )

        st.exception(error)


# ---------------------------------------------------------------------------
# Step 4.12 - Add model-use disclaimer
# ---------------------------------------------------------------------------

st.divider()

st.caption(
    "This prediction is intended to support customer prioritization and "
    "marketing decisions. It should be used as decision support rather than "
    "as the sole basis for customer treatment."
)
