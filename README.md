# Bayesian Logistic Regression for BRB-N Cognitive Classification in MS

## Overview

This repository contains trained Bayesian logistic regression models and a Streamlit application for estimating demographic-adjusted classification thresholds for Brief Repeatable Battery of Neuropsychological Tests (BRB-N) cognitive measures in multiple sclerosis (MS).

The models estimate the probability of cognitive impairment from age, years of education, gender, and each cognitive measure. The Streamlit app uses the pretrained models, healthy-control scalers, and measure-specific probability thresholds to compute score cutoffs for a given demographic profile.

## Repository Contents

- `streamlit_app.py` - Streamlit application for interactive threshold estimation
- `bayesian_logistic.py` - Analysis script used to fit Bayesian logistic regression models and derive model summaries
- `requirements.txt` - Python package requirements
- `models/` - Pretrained Bayesian models and fitted scalers for each BRB-N measure

## Available Cognitive Measures

The app and pretrained model files cover the following measures:

- `SRT-LTS`
- `SRT-cLTS`
- `SPART-correct`
- `SPART-incorrect`
- `SDMT`
- `PASAT 3`
- `PASAT 2`
- `SRT delayed recall`
- `SPART delayed correct response`
- `SPART delayed incorrect response`
- `WLG-correct`
- `WLG-repeat`
- `WLG-incorrect`

## Getting Started

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit App Locally

```bash
streamlit run streamlit_app.py
```

The app asks for age, years of education, and gender, then computes classification thresholds for all available BRB-N measures.

You can also access the deployed app here:

[Try the Web App](https://ms-brbn-study-dfj5etjwjf78trz4u7d9jk.streamlit.app/)

## How Thresholds Are Computed

For each cognitive measure, the app:

1. Loads the pretrained Bayesian logistic regression model from `models/bayesian_model_<measure>.pkl`.
2. Loads the healthy-control scaler from `models/scaler_hs_<measure>.pkl`.
3. Uses the measure-specific probability threshold selected by the Youden index.
4. Solves the logistic regression equation for the cognitive score threshold.
5. Converts the threshold back to the original score scale.

The displayed result is rounded into an interpretable normal-range rule, such as `N or more` or `N or less`, depending on the direction of abnormality for the measure.

## Loading a Model Programmatically

Pretrained models are stored as pickle files in `models/` and can be loaded with `joblib`:

```python
import joblib

trace = joblib.load("models/bayesian_model_SDMT.pkl")
scaler_hs = joblib.load("models/scaler_hs_SDMT.pkl")
```

The loaded Bayesian model is an ArviZ `InferenceData` object containing posterior samples for:

- `intercept`
- `β_age`
- `β_edu`
- `β_gender`
- `β_measure`

## Analysis Script

`bayesian_logistic.py` records the model-fitting workflow used for the study, including:

- Bayesian logistic regression with PyMC
- posterior summaries with ArviZ
- odds ratio calculation
- posterior probability calculation
- ROC analysis and Youden-index threshold selection
- demographic-adjusted score threshold calculation

The script assumes that the study data frames and column lists (`df`, `df_hs`, `A_cols`, and `formal_cols`) already exist in the analysis environment. It is therefore documentation of the fitting workflow rather than a standalone command-line entry point.

## Model Diagnostics

Model convergence was assessed with standard Bayesian diagnostics, including R-hat and effective sample size (ESS). Additional diagnostic plots and statistical details are provided in the manuscript and supplementary materials.

## Reproducibility

- Python dependencies are listed in `requirements.txt`.
- Pretrained model and scaler files are included in `models/`.
- The original fitting workflow is provided in `bayesian_logistic.py`.

The raw study dataset is not included in this repository.

## Citation

If you use this code or the pretrained models in research, please cite:

Akaike S, Mifune N, Yabe I, Otsuki M. The Japanese version of the Brief Repeatable Battery of Neuropsychological Tests in multiple sclerosis: A preliminary validation. Cogn Behav Neurol. 2026 Apr 14. Available from: http://dx.doi.org/10.1097/WNN.0000000000000422

## License

This code is released under the MIT License.
