import arviz as az
import joblib
import numpy as np
import pandas as pd
import pymc as pm
import streamlit as st
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler

# Define y (based on the provided information)
y_labels = np.array([0] * 43 + [1] * 22)

# Define cognitive measures
cognitive_measures = [
    "SRT-LTS", "SRT-cLTS", "SPART-correct", "SPART-incorrect", "SDMT",
    "PASAT 3", "PASAT 2", "SRT delayed recall", "SPART delayed correct response",
    "SPART delayed incorrect response", "WLG-correct", "WLG-repeat", "WLG-incorrect"
]

# Load Bayesian logistic regression models
@st.cache_resource
def load_models():
    """Load all Bayesian logistic regression models from the models/ directory."""
    return {measure: joblib.load(f"models/bayesian_model_{measure}.pkl") for measure in cognitive_measures}

models = load_models()

# Load StandardScalers
@st.cache_resource
def load_scalers():
    """Load trained StandardScalers for full dataset and healthy subjects."""
    scalers = {measure: joblib.load(f"models/scaler_{measure}.pkl") for measure in cognitive_measures}
    scalers_hs = {measure: joblib.load(f"models/scaler_hs_{measure}.pkl") for measure in cognitive_measures}
    return scalers, scalers_hs

scalers, scalers_hs = load_scalers()

def get_optimal_threshold(trace, y_labels):
    """Determine the optimal probability threshold using the Youden Index.

    Args:
        trace (InferenceData): The Bayesian inference trace.
        y_labels (array-like): The true labels (0 for healthy, 1 for patient).

    Returns:
        float: The optimal probability threshold for the given measure.
    """
    import streamlit as st  # Streamlit の UI にデバッグ情報を表示

    # **Debug: Display available keys in posterior_predictive**
    if "posterior_predictive" not in trace.groups():
        st.error("Error: `posterior_predictive` is missing in trace. Ensure the model was saved correctly.")
        raise KeyError("posterior_predictive does not exist in trace.")

    available_keys = list(trace.posterior_predictive.data_vars)
    st.write("Available posterior_predictive keys:", available_keys)

    if "y" not in available_keys:
        st.error("Error: `y` is missing in `posterior_predictive`. Please check your saved model.")
        raise KeyError("posterior_predictive['y'] is missing.")

    # Extract predicted probabilities safely
    posterior_pred = np.asarray(trace.posterior_predictive["y"]).mean(axis=(0, 1))

    # Compute ROC curve using actual y_labels
    fpr, tpr, thresholds = roc_curve(y_labels, posterior_pred)
    youden_index = tpr - fpr
    best_threshold = thresholds[np.argmax(youden_index)]
    
    return best_threshold


def get_score_threshold(trace, measure, age, education_year, gender, target_prob):
    """Compute the score threshold using healthy subject data for standardization.

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
    intercept = trace.posterior["intercept"].mean().item()
    β_age = trace.posterior["β_age"].mean().item()
    β_edu = trace.posterior["β_edu"].mean().item()
    β_gender = trace.posterior["β_gender"].mean().item()
    β_measure = trace.posterior["β_measure"].mean().item()

    # Standardize using healthy subject dataset scaler
    input_data = pd.DataFrame({"age": [age], "education_year": [education_year]})
    scaled_input = scalers_hs[measure].transform(input_data)  # Use measure-specific scaler

    age_scaled, edu_scaled = scaled_input[0]
    gender_binary = 1 if gender == "M" else 0  

    # Compute threshold in standardized scale
    A_threshold_scaled = (np.log(target_prob / (1 - target_prob)) - 
                          (intercept + β_age * age_scaled + β_edu * edu_scaled + β_gender * gender_binary)) / β_measure

    # Reverse standardization using measure-specific scaler for healthy subjects
    A_threshold_original = scalers_hs[measure].inverse_transform([[A_threshold_scaled]])[0, 0]

    return A_threshold_original


# Streamlit UI
st.title("BRB-N Cognitive Classification Threshold Estimator")
st.markdown("Enter demographic information to compute classification thresholds for all BRB-N cognitive measures.")

# User inputs
age = st.number_input("Age", min_value=18, max_value=90, value=40)
education_year = st.number_input("Years of Education", min_value=6, max_value=20, value=12)
gender = st.radio("Gender", ("M", "F"))

# Compute thresholds automatically using Youden Index
if st.button("Compute All Thresholds"):
    threshold_results = {}

    for measure in cognitive_measures:
        trace = models[measure]

        # Compute optimal probability threshold (Youden Index) per measure
        optimal_threshold = get_optimal_threshold(trace)

        # Compute corresponding score threshold
        threshold = get_score_threshold(trace, measure, age, education_year, gender, optimal_threshold)
        threshold_results[measure] = threshold

    # Display results
    st.write("### Computed Classification Thresholds")
    threshold_df = pd.DataFrame(list(threshold_results.items()), columns=["Cognitive Measure", "Threshold"])
    st.dataframe(threshold_df)
