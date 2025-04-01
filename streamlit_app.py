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

# best thresholds
best_thresholds = {'SRT-LTS': 0.4375, 'SRT-cLTS': 0.4876666666666667, 'SPART-correct': 0.2753333333333333, 'SPART-incorrect': 0.38816666666666666, 'SDMT': 0.622, 'PASAT 3': 0.528, 'PASAT 2': 0.43516666666666665, 'SRT delayed recall': 0.4701666666666667, 'SPART delayed correct response': 0.37733333333333335, 'SPART delayed incorrect response': 0.31183333333333335, 'WLG-correct': 0.3975, 'WLG-repeat': 0.3915, 'WLG-incorrect': 0.385}

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


# Streamlit UI
st.title("BRB-N Cognitive Classification Threshold Estimator")
st.markdown("Enter demographic information to compute classification thresholds for all BRB-N cognitive measures.")

# User inputs
age = st.number_input("Age", min_value=18, max_value=90, value=40)
education_year = st.number_input("Years of Education", min_value=6, max_value=20, value=12)
gender = st.radio("Gender", ("M", "F"))

def get_optimal_score(trace, scaler, target_prob, age, education_year, gender):
    """ 
    年齢・教育歴・性別を入力し、ベイズモデルの事後分布を用いて
    認知検査スコアの最適閾値を求める関数
    """
    
    # 事後分布の平均値（MAP推定）
    intercept = trace.posterior["intercept"].mean().item()
    β_age = trace.posterior["β_age"].mean().item()
    β_edu = trace.posterior["β_edu"].mean().item()
    β_gender = trace.posterior["β_gender"].mean().item()
    β_measure = trace.posterior["β_measure"].mean().item()

    # 標準化 (scaler から取得)
    age_scaled = (age - scaler.mean_[0]) / scaler.scale_[0]
    edu_scaled = (education_year - scaler.mean_[1]) / scaler.scale_[1]
    gender_binary = 1 if gender == "M" else 0  # 性別を数値化

    # `A_col` のスコアの閾値を計算（p = sigmoid(intercept + ... + β_measure * A) の逆算）
    A_threshold_scaled = (np.log(target_prob / (1 - target_prob)) - 
                          (intercept + β_age * age_scaled + β_edu * edu_scaled + β_gender * gender_binary)) / β_measure
    
    # 逆標準化（元のスコアに戻す）
    A_threshold = A_threshold_scaled * scaler.scale_[2] + scaler.mean_[2]  

    return A_threshold

# results
optimal_thresholds = {}

# Compute thresholds automatically using Youden Index
if st.button("Compute All Thresholds"):
    for measure in cognitive_measures:
        trace = models[measure]
        scaler_hs = scalers_hs[measure]
        best_threshold = best_thresholds[measure]

        # Compute optimal probability threshold (Youden Index) per measure
        optimal_thresholds[measure] = get_optimal_score(
            trace, scaler_hs, best_threshold, age, education_year, gender
        )

    # 表示形式を調整して表示
    st.write("### Computed Classification Thresholds")

    # 異常値が「高い」場合（→ 正常は以下）
    abnormal_high_measures = {
        "SPART-incorrect", "SDMT", "SPART delayed incorrect response",
        "WLG-repeat", "WLG-incorrect"
    }

    display_thresholds = {}

    for measure, value in optimal_thresholds.items():
        if measure in abnormal_high_measures:
            value_display = int(np.floor(value))  # 以下 → 切り捨て
            display = f"{value_display} or less"
        else:
            value_display = int(np.ceil(value))   # 以上 → 切り上げ
            display = f"{value_display} or more"
        display_thresholds[measure] = display

    # DataFrame で表示
    display_df = pd.DataFrame(display_thresholds.values(), index=display_thresholds.keys(), columns=["Normal Range"])
    st.dataframe(display_df)
