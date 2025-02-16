# Bayesian Logistic Regression for Cognitive Impairment in MS

## Overview

This repository contains the Bayesian logistic regression analysis and trained models used in our study on cognitive impairment in multiple sclerosis (MS). The analysis was performed using PyMC to estimate odds ratios and posterior probabilities for cognitive measures.

## Repository Contents

- `bayesian_logistic.py` - Python script for Bayesian logistic regression, odds ratio estimation, and posterior probability calculation
- `streamlit_app.py` - Streamlit app for interactive threshold estimation
- `requirements.txt` - List of required Python libraries
- `models/` - Directory containing trained Bayesian logistic regression models for each cognitive measure

## Getting Started

### Environment Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Model Location and Loading

The trained Bayesian logistic regression models for each cognitive measure are stored in the `models/` directory. To load a specific model, use the following command:

```python
import joblib

# Example: Load the SDMT model
trace = joblib.load("models/bayesian_model_SDMT.pkl")
```

### Running the Analysis
Instead of executing `bayesian_logistic.py` directly, users should load the pretrained model and perform the desired computations. Below is an example workflow:

```python
import joblib
from bayesian_logistic import get_optimal_score
from sklearn.preprocessing import StandardScaler

# Load a trained model
trace = joblib.load("models/bayesian_model_SDMT.pkl")

# Define demographic information
age = 40
education_year = 12
gender = "M"
target_prob = 0.7  # Example threshold probability

# Use the correct scaler (healthy control group)
scaler_hs = StandardScaler()
scaler_hs.fit(df_hs[['age', 'education_year', "A_col"]])

# Compute the optimal cognitive measure threshold
score_threshold = get_optimal_score(
    trace, scaler_hs, target_prob=target_prob, age=age, education_year=education_year, gender=gender
)

print(f"Score threshold: {score_threshold:.3f}")
```

The script will output:

- Odds Ratios (ORs) and their credible intervals
- Posterior probabilities for each predictor
- Classification thresholds for cognitive measures

## Model Details

### Convergence and Performance

- Model convergence was assessed using R-hat and effective sample size (ESS)
- Confirmed stability with R-hat ≤1.01 and ESS ≥400
- Convergence diagnostics (trace plots, rank plots, and autocorrelation plots) are provided in the Supplementary Figures of our manuscript

### Classification Threshold Estimation

The repository includes functions for estimating classification thresholds:

- `get_optimal_score()`: Determines the optimal probability threshold for classification using the Youden Index. Computes the corresponding cognitive score threshold for classification

To estimate classification thresholds for a specific demographic profile:

```python
# Define demographic information
age = 40
education_year = 12
gender = "M"
target_prob = 0.7  # Example threshold probability

# Use the correct scaler (healthy control group)
scaler_hs = StandardScaler()
scaler_hs.fit(df_hs[['age', 'education_year', "A_col"]])

# Compute the optimal cognitive measure threshold
score_threshold = get_optimal_score(
    trace_bayes_logit, 
    scaler_hs,  # Ensure the scaler is based on healthy control data
    target_prob=target_prob, 
    age=age, 
    education_year=education_year, 
    gender=gender
)

print(f"Score threshold: {score_threshold:.3f}")
```

### Streamlit App Usage

A Streamlit app (`streamlit_app.py`) has been added for interactive threshold estimation.  
You can access the live application here:

👉 **[Try the Web App](https://ms-brbn-study-dfj5etjwjf78trz4u7d9jk.streamlit.app/)** 👈

Alternatively, you can run it locally:

```bash
streamlit run streamlit_app.py
```
### Features:
- Compute optimal classification thresholds interactively
- Display results in a **scroll-free, horizontally aligned table**
- Toggle between different cognitive measures

## Data Presentation in Streamlit
To improve visibility, the classification thresholds are presented in a horizontal table format using the following command:

```python
st.dataframe(df.T.style.set_properties(**{'text-align': 'center'}))
```

## Reproducibility

For full reproducibility:
- Exact versions of dependencies are listed in `requirements.txt`
- Statistical methods and results are detailed in our published manuscript

## Citation

If you use this code or model in your research, please cite our paper:

```text
[Our Paper Title]
[Our Authors]
[Journal Name, Year]
[DOI or URL]
```
*Citation details will be added after the publication of our paper.*

## License

This code is released under the MIT License.