import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import pymc as pm
from sklearn.metrics import roc_curve

# ページ設定
st.set_page_config(page_title="認知機能検査閾値計算ツール", layout="wide")
st.title("認知機能検査閾値計算ツール")

# 目的変数の定義
Y_LABELS = np.array([0] * 43 + [1] * 22)

# テスト名と表示名のマッピング
TEST_NAMES = {
    'PASAT 2': '注意機能検査（2秒）',
    'PASAT 3': '注意機能検査（3秒）',
    'SDMT': '情報処理速度検査',
    'SPART delayed correct response': '視空間記憶検査（遅延再生・正答数）',
    'SPART delayed incorrect response': '視空間記憶検査（遅延再生・誤答数）',
    'SPART-correct': '視空間記憶検査（即時再生・正答数）',
    'SPART-incorrect': '視空間記憶検査（即時再生・誤答数）',
    'SRT delayed recall': '言語記憶検査（遅延再生）',
    'SRT-cLTS': '言語記憶検査（一貫性のある長期記憶）',
    'SRT-LTS': '言語記憶検査（長期記憶）',
    'WLG-correct': '言語流暢性検査（正答数）',
    'WLG-incorrect': '言語流暢性検査（誤答数）',
    'WLG-repeat': '言語流暢性検査（反復数）'
}

@st.cache_resource
def load_models():
    """モデルとスケーラーを読み込む"""
    models = {}
    scalers = {}
    optimal_thresholds = {}
    
    model_files = Path('models').glob('bayesian_model_*.pkl')
    for model_file in model_files:
        test_name = model_file.stem.replace('bayesian_model_', '')
        
        # モデルの読み込み
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
            models[test_name] = model
            
            # 最適な確率閾値を計算
            optimal_thresholds[test_name] = calculate_optimal_threshold(model)
        
        # 対応するスケーラーの読み込み
        scaler_file = f'models/scaler_hs_{test_name}.pkl'
        if Path(scaler_file).exists():
            with open(scaler_file, 'rb') as f:
                scalers[test_name] = pickle.load(f)
    
    return models, scalers, optimal_thresholds

def calculate_optimal_threshold(model):
    """Youden indexが最大となる閾値を計算"""
    with model:
        # 事後予測の計算
        posterior_pred = pm.sample_posterior_predictive(model, return_inferencedata=True)
        pred_probs = posterior_pred.posterior_predictive["y"].mean(dim=["chain", "draw"]).values
    
    # ROC曲線とYouden indexの計算
    fpr, tpr, thresholds = roc_curve(Y_LABELS, pred_probs)
    youden_index = tpr - fpr
    optimal_threshold = thresholds[np.argmax(youden_index)]
    
    return optimal_threshold

def get_score_threshold(model, scaler, age, education_year, gender, optimal_prob):
    """スコアの閾値を計算する"""
    # 入力データの準備と標準化
    input_data = np.array([[age, education_year]])
    input_scaled = scaler.transform(input_data)
    
    # モデルのパラメータを取得
    intercept = model.posterior["intercept"].mean().item()
    β_age = model.posterior["β_age"].mean().item()
    β_edu = model.posterior["β_edu"].mean().item()
    β_gender = model.posterior["β_gender"].mean().item()
    β_measure = model.posterior["β_measure"].mean().item()
    
    # 閾値の計算
    age_scaled = input_scaled[0, 0]
    edu_scaled = input_scaled[0, 1]
    gender_binary = 1 if gender == "M" else 0
    
    # ロジスティック回帰の逆関数を使用して閾値を計算
    score_scaled = (np.log(optimal_prob / (1 - optimal_prob)) - 
                   (intercept + β_age * age_scaled + 
                    β_edu * edu_scaled + β_gender * gender_binary)) / β_measure
    
    # スケール変換の逆変換
    score = score_scaled * scaler.scale_[-1] + scaler.mean_[-1]
    
    return score

# サイドバーで入力を受け付ける
st.sidebar.header("患者情報入力")
with st.sidebar:
    age = st.slider("年齢", 20, 60, 40)
    gender = st.selectbox("性別", ["M", "F"])
    education_year = st.number_input("教育年数", min_value=6, max_value=20, value=12)

# メインコンテンツ
try:
    # モデルとスケーラーの読み込み
    models, scalers, optimal_thresholds = load_models()
    
    if st.sidebar.button("閾値を計算", type="primary"):
        st.header("検査閾値結果")
        
        # 3列のレイアウトを作成
        cols = st.columns(3)
        
        # 各テストの結果を表示
        for i, (test_name, model) in enumerate(models.items()):
            # スコアの閾値を計算
            threshold = get_score_threshold(
                model,
                scalers[test_name],
                age,
                education_year,
                gender,
                optimal_thresholds[test_name]
            )
            
            # 日本語の表示名を取得
            display_name = TEST_NAMES.get(test_name, test_name)
            
            # 結果を3列に分けて表示
            with cols[i % 3]:
                st.metric(
                    label=display_name,
                    value=f"{threshold:.2f}"
                )
        
        # 注意書きの追加
        st.info("※ これらの閾値は参考値です。臨床判断の際は、他の検査結果や臨床所見も併せてご判断ください。")

except Exception as e:
    st.error(f"エラーが発生しました: {str(e)}")
    st.error("必要なモデルファイルが見つからないか、読み込めない可能性があります。")