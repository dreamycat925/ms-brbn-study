# Bayesian Logistic Regression for Cognitive Impairment in MS

## Overview

This repository contains the Bayesian logistic regression analysis and trained models used in our study on cognitive impairment in multiple sclerosis (MS). The analysis was performed using PyMC to estimate odds ratios and posterior probabilities for cognitive measures.

## Repository Contents

- `bayesian_logistic.py` - Python script for Bayesian logistic regression, odds ratio estimation, and posterior probability calculation
- `requirements.txt` - List of required Python libraries
- `model.pkl` - Trained Bayesian logistic regression model (if applicable)

## Getting Started

### Environment Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Analysis

Execute the Bayesian logistic regression model to obtain odds ratios and posterior probabilities:

```bash
python bayesian_logistic.py
```

The script will output:
- Odds Ratios (ORs) and their credible intervals
- Posterior probabilities for each predictor

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
[Your Paper Title]
[Your Authors]
[Journal Name, Year]
[DOI or URL]
```

## License

This code is released under the MIT License.