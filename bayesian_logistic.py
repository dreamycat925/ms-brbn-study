import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

# カラムの抽出
A_cols = df_hs.columns[df_hs.columns.str.match('A_')].to_list()
B_cols = df_hs.columns[df_hs.columns.str.match('B_')].to_list()
formal_cols = ['SRT-LTS', 'SRT-cLTS', 'SPART-correct', 'SPART-incorrect', 'SDMT', 'PASAT 3', 'PASAT 2', 'SRT delayed recall', 'SPART delayed correct response', 'SPART delayed incorrect response', 'WLG-correct', 'WLG-repeat', 'WLG-incorrect']

# 各成績でベイズロジスティック回帰分析
for A_col, f_col in zip(A_cols, formal_cols):
    print('')
    print(f'============================================{f_col}============================================')
    
    # 説明変数
    X = df[['age', 'gender', 'education_year', A_col]].copy()
    
    # カテゴリ変数のエンコーディング
    X = pd.get_dummies(X, columns=['gender'], drop_first=True, dtype=int) 
    
    # 目的変数
    y = (df['group'] == 'ci').astype(int).values  # numpy 配列に変換
    
    # 標準化
    scaler = StandardScaler()
    num_cols = ['age', 'education_year', A_col]  # 標準化対象の数値カラム
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    # PyMCによるベイズロジスティック回帰
    with pm.Model() as bayesian_logit_model:
        β_age = pm.Normal("β_age", mu=0, sigma=1)
        β_edu = pm.Normal("β_edu", mu=0, sigma=1)
        β_gender = pm.Normal("β_gender", mu=0, sigma=1)
        β_measure = pm.Normal("β_measure", mu=0, sigma=1)
        intercept = pm.Normal("intercept", mu=0, sigma=2)
        
        # 線形結合
        μ = (
            intercept
            + β_age * X["age"].values
            + β_edu * X["education_year"].values
            + β_gender * X["gender_M"].values
            + β_measure * X[A_col].values
        )
        
        # シグモイド関数で確率を得る
        p = pm.math.sigmoid(μ)
        
        # ベルヌーイ分布の尤度
        likelihood = pm.Bernoulli("y", p=p, observed=y)
        
        # サンプリング
        trace_bayes_logit = pm.sample(2000, tune=1000, target_accept=0.98, return_inferencedata=True)
    
    # 事後分布の要約
    summary = az.summary(trace_bayes_logit)
    print(summary)
    
    # オッズ比を計算
    odds_ratios = np.exp(summary['mean'])
    print("\nオッズ比:")
    print(odds_ratios)

    odds_ratios_lower = np.exp(summary['hdi_3%'])
    print("\nオッズ比下限:")
    print(odds_ratios_lower)

    odds_ratios_upper = np.exp(summary['hdi_97%'])
    print("\nオッズ比上限:")
    print(odds_ratios_upper)
    
    # 事後確率を計算
    def compute_posterior_probabilities(trace, param):
        """事後確率を求める"""
        samples = trace.posterior[param].values.flatten()
        prob_positive = (samples > 0).mean()
        prob_negative = (samples < 0).mean()
        return prob_positive, prob_negative
    
    print("\n事後確率:")
    for param in ["β_age", "β_edu", "β_gender", "β_measure"]:
        p_pos, p_neg = compute_posterior_probabilities(trace_bayes_logit, param)
        print(f"{param}: P(β > 0) = {p_pos:.3f}, P(β < 0) = {p_neg:.3f}")
    
    # 最適な predict_proba の閾値を求める関数
    def get_optimal_threshold_for_specific_condition(trace, X_train, y, age, education_year, gender):
        input_data = pd.DataFrame({
            "age": [age],
            "education_year": [education_year],
            "gender_M": [1 if gender == "M" else 0]
        })
        
        # すべてのカラムを0で初期化
        input_data[X_train.columns] = 0  
        input_data["age"] = age
        input_data["education_year"] = education_year
        input_data["gender_M"] = 1 if gender == "M" else 0
    
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
    
    optimal_threshold_condition = get_optimal_threshold_for_specific_condition(trace_bayes_logit, X, y, 40, 12, "M")
    print(f"40歳・教育12年・男性の最適閾値（Youden Index 最大）: {optimal_threshold_condition:.3f}")
    
    
    # `A_col` のスコア閾値を求める関数
    def get_score_threshold_for_prob(trace, df_hs, measure, age, education_year, gender, target_prob):
        """特定の条件に対して、predict_proba の閾値に対応するスコアの閾値を求める"""
    
        # X
        X = df_hs[['age', 'gender', 'education_year', measure]]
    
        # scaler
        scaler = StandardScaler()
        scaler.fit(X[['age', 'education_year']])
        
        # 事後分布の平均値（MAP推定）
        intercept = trace.posterior["intercept"].mean().item()
        β_age = trace.posterior["β_age"].mean().item()
        β_edu = trace.posterior["β_edu"].mean().item()
        β_gender = trace.posterior["β_gender"].mean().item()
        β_measure = trace.posterior["β_measure"].mean().item()
        
        # 変数の標準化（`X` から `age`, `education_year` のスケールを取得）
        age_scaled = (age - scaler.mean_[0]) / scaler.scale_[0]
        edu_scaled = (education_year - scaler.mean_[1]) / scaler.scale_[1]
        gender_binary = 1 if gender == "M" else 0  # 性別を数値化
    
        # `A_col` のスコアの閾値を計算（p = sigmoid(intercept + ... + β_measure * A) の逆算）
        A_threshold_scaled = (np.log(target_prob / (1 - target_prob)) - (intercept + β_age * age_scaled + β_edu * edu_scaled + β_gender * gender_binary)) / β_measure
        
        # 逆標準化（元のスコアに戻す）
        A_threshold = A_threshold_scaled * df_hs[measure].std() + df_hs[measure].mean()  
    
        return A_threshold
    
    
    score_threshold = get_score_threshold_for_prob(
        trace_bayes_logit, df_hs, A_col, 40, 12, "M", target_prob=optimal_threshold_condition
    )
    print(f"40歳・教育12年・男性の最適スコア閾値: {score_threshold:.3f}")
    
    # 閾値の計算式を出力
    def print_threshold_equation(trace, p_star):
        """学習済みのパラメータを使い、閾値の計算式を生成"""
        β_0 = trace.posterior["intercept"].mean().item()
        β_age = trace.posterior["β_age"].mean().item()
        β_edu = trace.posterior["β_edu"].mean().item()
        β_gender = trace.posterior["β_gender"].mean().item()
        β_measure = trace.posterior["β_measure"].mean().item()
        
        print("\n閾値計算式:")
        print(f"score = (log({p_star:.3f} / (1 - {p_star:.3f})) - ({β_0:.3f} + β_age * ((age - 40.462) / 11.232) + β_edu * ((education - 13.877) / 1.799) + β_gender * gender)) / {β_measure:.3f}")
        print('genderはM:1, F:0')
    
    print_threshold_equation(trace_bayes_logit, optimal_threshold_condition)