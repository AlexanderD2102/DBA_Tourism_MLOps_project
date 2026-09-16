
# ===========================================================================
# STEP 2 - DEFINE THE CONTAINERIZED STREAMLIT ENVIRONMENT
# ===========================================================================

# ---------------------------------------------------------------------------
# Step 2.1 - Use a lightweight Python base image
# ---------------------------------------------------------------------------

# Python 3.12 is aligned with the deployment environment and supports
# the package versions used by the validated production model.
FROM python:3.12-slim


# ---------------------------------------------------------------------------
# Step 2.2 - Define the initial container working directory
# ---------------------------------------------------------------------------

WORKDIR /app


# ---------------------------------------------------------------------------
# Step 2.3 - Copy deployment files into the container
# ---------------------------------------------------------------------------

# The Docker build context is expected to be:
# tourism_project/deployment/
#
# This copies app.py, best_model.joblib, model_metrics.json,
# requirements.txt, and the other deployment files into the container.
COPY . .


# ---------------------------------------------------------------------------
# Step 2.4 - Install the application dependencies
# ---------------------------------------------------------------------------

RUN pip install --no-cache-dir -r requirements.txt


# ---------------------------------------------------------------------------
# Step 2.5 - Run the application as a non-root user
# ---------------------------------------------------------------------------

# Create a dedicated user for safer container execution
RUN useradd -m -u 1000 user

USER user

# Configure the user home directory and executable path
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH


# ---------------------------------------------------------------------------
# Step 2.6 - Move the application into the non-root user's workspace
# ---------------------------------------------------------------------------

WORKDIR $HOME/app

COPY --chown=user . $HOME/app


# ---------------------------------------------------------------------------
# Step 2.7 - Expose the Streamlit application port
# ---------------------------------------------------------------------------

EXPOSE 8501


# ---------------------------------------------------------------------------
# Step 2.8 - Define the container startup command
# ---------------------------------------------------------------------------

# Start the Streamlit application and expose it outside the container
CMD [
    "streamlit",
    "run",
    "app.py",
    "--server.port=8501",
    "--server.address=0.0.0.0",
    "--server.enableXsrfProtection=false"
]
