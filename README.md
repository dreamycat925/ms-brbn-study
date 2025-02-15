# Bayesian Logistic Regression for Cognitive Impairment in MS

## Overview

This repository contains the Bayesian logistic regression analysis and trained models used in our study on cognitive impairment in multiple sclerosis (MS). The analysis was performed using PyMC to estimate odds ratios and posterior probabilities for cognitive measures.

## Repository Contents

- `bayesian_logistic.py` - Python script for Bayesian logistic regression, odds ratio estimation, and posterior probability calculation
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
from bayesian_logistic import get_optimal_threshold, get_score_threshold

# Load a trained model
trace = joblib.load("models/bayesian_model_SDMT.pkl")

# Define demographic information
age = 40
education_year = 12
gender = "M"

# Compute classification thresholds
optimal_threshold = get_optimal_threshold(trace, X, y, age=age, education_year=education_year, gender=gender)
score_threshold = get_score_threshold(trace, df_hs, "A_col", target_prob=optimal_threshold, age=age, education_year=education_year, gender=gender)

print(f"Optimal probability threshold: {optimal_threshold:.3f}")
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

- `get_optimal_threshold()`: Determines the optimal probability threshold for classification using the Youden Index
- `get_score_threshold()`: Computes the corresponding cognitive score threshold for classification

To estimate classification thresholds for a specific demographic profile:

```python
optimal_threshold = get_optimal_threshold(
    trace_bayes_logit, 
    X, 
    y, 
    age=40, 
    education_year=12, 
    gender="M"
)

score_threshold = get_score_threshold(
    trace_bayes_logit, 
    df_hs, 
    "A_col", 
    target_prob=optimal_threshold, 
    age=40, 
    education_year=12, 
    gender="M"
)
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