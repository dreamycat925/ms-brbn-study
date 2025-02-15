import streamlit as st
import arviz as az
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve

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
    return {measure: az.from_netcdf(f"models/bayesian_model_{measure}.nc") for measure in cognitive_measures}

models = load_models()

def get_optimal_threshold(trace, X_train, y, age, education_year, gender):
    """Determine the optimal probability threshold using the Youden Index.

    Args:
        trace (InferenceData): The Bayesian inference trace.
        X_train (DataFrame): Training data for feature scaling.
        y (array): Target variable.
        age (int): Age of the subject.
        education_year (int): Years of education.
        gender (str): Gender ('M' for male, 'F' for female).

    Returns:
        float: The optimal probability threshold for the given measure.
    """
    input_data = pd.DataFrame({
        "age": [age],
        "education_year": [education_year],
        "gender_M": [1 if gender == "M" else 0]
    })

    scaler = StandardScaler()
    scaler.fit(X_train[X_train.select_dtypes(include=[np.number]).columns])
    input_data[X_train.select_dtypes(include=[np.number]).columns] = scaler.transform(input_data[X_train.select_dtypes(include=[np.number]).columns])

    with pm.Model():
        posterior_pred = pm.sample_posterior_predictive(trace, var_names=["y"])

    pred_prob = posterior_pred.posterior_predictive["y"].mean(dim=["chain", "draw"]).values  
    fpr, tpr, thresholds = roc_curve(y, pred_prob)
    youden_index = tpr - fpr
    best_threshold = thresholds[np.argmax(youden_index)]
    
    return best_threshold

def get_score_threshold(trace, measure, age, education_year, gender, target_prob):
    """Compute the score threshold corresponding to the optimal probability threshold.

    Args:
        trace (InferenceData): The Bayesian inference trace.
        measure (str): The cognitive measure to evaluate.
        age (int): Age of the subject.
        education_year (int): Years of education.
        gender (str): Gender ('M' for male, 'F' for female).
        target_prob (float): The probability threshold determined by the Youden Index.

    Returns:
        float: The computed score threshold.
    """
    scaler = StandardScaler()
    scaler.fit(pd.DataFrame({"age": [40], "education_year": [12]}))  # Reference standardization

    intercept = trace.posterior["intercept"].mean().item()
    β_age = trace.posterior["β_age"].mean().item()
    β_edu = trace.posterior["β_edu"].mean().item()
    β_gender = trace.posterior["β_gender"].mean().item()
    β_measure = trace.posterior["β_measure"].mean().item()

    age_scaled = (age - 40) / 11  
    edu_scaled = (education_year - 12) / 2  
    gender_binary = 1 if gender == "M" else 0  

    A_threshold_scaled = (np.log(target_prob / (1 - target_prob)) - 
                          (intercept + β_age * age_scaled + β_edu * edu_scaled + β_gender * gender_binary)) / β_measure
    return A_threshold_scaled * 10 + 50  # Adjust scaling as needed

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

        # Prepare X and y for each cognitive measure
        X = df[['age', 'gender', 'education_year', measure]].copy()
        X = pd.get_dummies(X, columns=['gender'], drop_first=True, dtype=int)  
        y = (df['group'] == 'ci').astype(int).values  

        scaler = StandardScaler()
        num_cols = ['age', 'education_year', measure]
        X[num_cols] = scaler.fit_transform(X[num_cols])

        # Compute optimal probability threshold (Youden Index) per measure
        optimal_threshold = get_optimal_threshold(trace, X, y, age, education_year, gender)

        # Compute corresponding score threshold
        threshold = get_score_threshold(trace, measure, age, education_year, gender, optimal_threshold)
        threshold_results[measure] = threshold

    # Display results
    st.write("### Computed Classification Thresholds")
    threshold_df = pd.DataFrame(list(threshold_results.items()), columns=["Cognitive Measure", "Threshold"])
    st.dataframe(threshold_df)
