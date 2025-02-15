import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

# Extract columns related to cognitive test scores
A_cols = df_hs.columns[df_hs.columns.str.match('A_')].to_list()
B_cols = df_hs.columns[df_hs.columns.str.match('B_')].to_list()
formal_cols = [
    'SRT-LTS', 'SRT-cLTS', 'SPART-correct', 'SPART-incorrect', 'SDMT',
    'PASAT 3', 'PASAT 2', 'SRT delayed recall', 'SPART delayed correct response',
    'SPART delayed incorrect response', 'WLG-correct', 'WLG-repeat', 'WLG-incorrect'
]

# Perform Bayesian logistic regression for each cognitive measure
for A_col, f_col in zip(A_cols, formal_cols):
    print(f'\n============================================{f_col}============================================')
    
    # Define explanatory variables
    X = df[['age', 'gender', 'education_year', A_col]].copy()
    
    # Encode categorical variables
    X = pd.get_dummies(X, columns=['gender'], drop_first=True, dtype=int) 
    
    # Define target variable
    y = (df['group'] == 'ci').astype(int).values  # Convert to numpy array
    
    # Standardize numerical variables
    scaler = StandardScaler()
    num_cols = ['age', 'education_year', A_col]
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    # Perform Bayesian logistic regression using PyMC
    with pm.Model() as bayesian_logit_model:
        β_age = pm.Normal("β_age", mu=0, sigma=1)
        β_edu = pm.Normal("β_edu", mu=0, sigma=1)
        β_gender = pm.Normal("β_gender", mu=0, sigma=1)
        β_measure = pm.Normal("β_measure", mu=0, sigma=1)
        intercept = pm.Normal("intercept", mu=0, sigma=2)
        
        # Linear combination of predictors
        μ = (
            intercept
            + β_age * X["age"].values
            + β_edu * X["education_year"].values
            + β_gender * X["gender_M"].values
            + β_measure * X[A_col].values
        )
        
        # Convert to probability using sigmoid function
        p = pm.math.sigmoid(μ)
        
        # Define likelihood using Bernoulli distribution
        likelihood = pm.Bernoulli("y", p=p, observed=y)
        
        # Sampling from posterior distribution
        trace_bayes_logit = pm.sample(2000, tune=1000, target_accept=0.98, return_inferencedata=True)
    
    # Summarize posterior distribution
    summary = az.summary(trace_bayes_logit)
    print(summary)
    
    # Compute odds ratios
    odds_ratios = np.exp(summary['mean'])
    print("\nOdds Ratios:")
    print(odds_ratios)

    odds_ratios_lower = np.exp(summary['hdi_3%'])
    print("\nLower Bound of Odds Ratios:")
    print(odds_ratios_lower)

    odds_ratios_upper = np.exp(summary['hdi_97%'])
    print("\nUpper Bound of Odds Ratios:")
    print(odds_ratios_upper)
    
def compute_posterior_probabilities(trace, param):
    """Compute posterior probabilities for a given parameter.

    Args:
        trace (InferenceData): The Bayesian inference trace.
        param (str): The name of the parameter.

    Returns:
        tuple: Probability that the parameter is positive and negative.
    """
    samples = trace.posterior[param].values.flatten()
    prob_positive = (samples > 0).mean()
    prob_negative = (samples < 0).mean()
    return prob_positive, prob_negative

print("\nPosterior Probabilities:")
for param in ["β_age", "β_edu", "β_gender", "β_measure"]:
    p_pos, p_neg = compute_posterior_probabilities(trace_bayes_logit, param)
    print(f"{param}: P(β > 0) = {p_pos:.3f}, P(β < 0) = {p_neg:.3f}")

def get_optimal_threshold_for_specific_condition(trace, X_train, y, age, education_year, gender):
    """Determine the optimal probability threshold using the Youden Index.

    Args:
        trace (InferenceData): The Bayesian inference trace.
        X_train (DataFrame): Training data for feature scaling.
        y (array): Target variable.
        age (int): Age of the subject.
        education_year (int): Years of education.
        gender (str): Gender ('M' for male, 'F' for female).

    Returns:
        float: The optimal probability threshold.
    """
    input_data = pd.DataFrame({
        "age": [age],
        "education_year": [education_year],
        "gender_M": [1 if gender == "M" else 0]
    })
    
    scaler = StandardScaler()
    scaler.fit(X_train[X_train.select_dtypes(include=[np.number]).columns])
    input_data[X_train.select_dtypes(include=[np.number]).columns] = scaler.transform(input_data[X_train.select_dtypes(include=[np.number]).columns])
    
    with bayesian_logit_model:
        posterior_pred = pm.sample_posterior_predictive(trace, var_names=["y"])
    
    pred_prob = posterior_pred.posterior_predictive["y"].mean(dim=["chain", "draw"]).values  
    fpr, tpr, thresholds = roc_curve(y, pred_prob)
    youden_index = tpr - fpr
    best_threshold = thresholds[np.argmax(youden_index)]
    
    return best_threshold

def get_score_threshold_for_prob(trace, df_hs, measure, age, education_year, gender, target_prob):
    """Compute the score threshold corresponding to a given probability threshold.

    Args:
        trace (InferenceData): The Bayesian inference trace.
        df_hs (DataFrame): Data containing cognitive measures.
        measure (str): The cognitive measure to evaluate.
        age (int): Age of the subject.
        education_year (int): Years of education.
        gender (str): Gender ('M' for male, 'F' for female).
        target_prob (float): The probability threshold.

    Returns:
        float: The computed score threshold.
    """
    X = df_hs[['age', 'gender', 'education_year', measure]]

    scaler = StandardScaler()
    scaler.fit(X[['age', 'education_year']])
    
    intercept = trace.posterior["intercept"].mean().item()
    β_age = trace.posterior["β_age"].mean().item()
    β_edu = trace.posterior["β_edu"].mean().item()
    β_gender = trace.posterior["β_gender"].mean().item()
    β_measure = trace.posterior["β_measure"].mean().item()

    age_scaled = (age - scaler.mean_[0]) / scaler.scale_[0]
    edu_scaled = (education_year - scaler.mean_[1]) / scaler.scale_[1]
    gender_binary = 1 if gender == "M" else 0  

    A_threshold_scaled = (np.log(target_prob / (1 - target_prob)) - (intercept + β_age * age_scaled + β_edu * edu_scaled + β_gender * gender_binary)) / β_measure
    A_threshold = A_threshold_scaled * df_hs[measure].std() + df_hs[measure].mean()  

    return A_threshold

print_threshold_equation(trace_bayes_logit, optimal_threshold_condition)
