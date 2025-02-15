import streamlit as st
import arviz as az
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load Bayesian logistic regression models
MODEL_PATH = "models/bayesian_model_SDMT.nc"

@st.cache_resource
def load_model():
    """Load the Bayesian logistic regression model."""
    return az.from_netcdf(MODEL_PATH)

trace = load_model()

# Function to compute the score threshold
def get_score_threshold(trace, measure, age, education_year, gender, target_prob):
    """Compute the score threshold corresponding to a given probability threshold.

    Args:
        trace (InferenceData): The Bayesian inference trace.
        measure (str): The cognitive measure to evaluate.
        age (int): Age of the subject.
        education_year (int): Years of education.
        gender (str): Gender ('M' for male, 'F' for female).
        target_prob (float): The probability threshold.

    Returns:
        float: The computed score threshold.
    """
    scaler = StandardScaler()
    scaler.fit(pd.DataFrame({"age": [40], "education_year": [12]}))  # Use a reference dataset

    intercept = trace.posterior["intercept"].mean().item()
    β_age = trace.posterior["β_age"].mean().item()
    β_edu = trace.posterior["β_edu"].mean().item()
    β_gender = trace.posterior["β_gender"].mean().item()
    β_measure = trace.posterior["β_measure"].mean().item()

    age_scaled = (age - 40) / 11  # Approximate scaling
    edu_scaled = (education_year - 12) / 2
    gender_binary = 1 if gender == "M" else 0  

    A_threshold_scaled = (np.log(target_prob / (1 - target_prob)) - 
                          (intercept + β_age * age_scaled + β_edu * edu_scaled + β_gender * gender_binary)) / β_measure
    return A_threshold_scaled * 10 + 50  # Adjust scaling as needed

# Streamlit UI
st.title("BRB-N Cognitive Classification Threshold Estimator")
st.markdown("Enter demographic information to compute the BRB-N classification threshold.")

# User inputs
age = st.number_input("Age", min_value=18, max_value=90, value=40)
education_year = st.number_input("Years of Education", min_value=6, max_value=20, value=12)
gender = st.radio("Gender", ("M", "F"))

# Target probability (Youden Index threshold)
target_prob = st.slider("Target Probability Threshold", min_value=0.1, max_value=0.9, value=0.5)

# Compute threshold
if st.button("Compute Threshold"):
    threshold = get_score_threshold(trace, "SDMT", age, education_year, gender, target_prob)
    st.write(f"Computed BRB-N Classification Threshold: {threshold:.2f}")

