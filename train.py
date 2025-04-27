"""
詐騙檢測模型訓練與評估模組，用於處理交易數據、生成特徵、訓練XGBoost模型並進行性能評估。
"""
import os
import warnings
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    precision_recall_curve
)
import xgboost as xgb
import joblib
from generator.full_auto_feature_generator import FraudDetectionFeatureGenerator
from service.utils import ensure_directory_exists

# 載入環境變數
load_dotenv()

# 常量設定
TRAIN_DATA = os.environ.get('TRAIN_DATA_PATH')

warnings.filterwarnings('ignore')

def load_data(file_path):
    """載入數據並進行基本處理"""
    print("載入數據...")
    try:
        # 嘗試不同的編碼方式
        encodings = ['utf-8', 'big5', 'cp950']
        df = None

        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"成功使用 {encoding} 編碼讀取數據")
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            raise ValueError("無法使用任何編碼方式讀取文件")

        # 打印列名資訊
        print("\n數據欄位資訊:")
        print("列名:", df.columns.tolist())
        print(f"原始數據維度: {df.shape}")

        # 檢查列名中的空格和特殊字符
        for col in df.columns:
            if col.strip() != col:
                print(f"警告: 列名 '{col}' 包含前導或尾隨空格")
                # 清理列名
                df.rename(columns={col: col.strip()}, inplace=True)

        return df

    except Exception as e:
        print(f"讀取數據時發生錯誤: {str(e)}")
        print("文件路徑:", file_path)
        raise

def prepare_features(df):
    """特徵準備和預處理"""
    print("\n準備特徵...")

    # 確保必要欄位存在
    required_columns = {
        'ACCT_NBR', 'TX_AMT', 'TX_DATE', 'TX_TIME', 'CHANNEL_CODE',
        'OWN_TRANS_ACCT', 'PB_BAL', 'SAME_NUMBER_IP', 'SAME_NUMBER_UUID',
        'CANCEL_NO_CONTACT', 'IS_DIGITAL', 'IS_FRAUD'
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"缺少必要欄位: {missing_columns}")

    # 分離特徵和目標變量
    y = df['IS_FRAUD']
    features = df.drop('IS_FRAUD', axis=1)

    # 生成特徵
    generator = FraudDetectionFeatureGenerator()
    features = generator.generate_features(features)

    # 確保X和y的長度匹配
    if len(features) != len(y):
        print(f"警告：特徵矩陣和目標變量長度不匹配 (features: {len(features)}, y: {len(y)})")
        # 使用索引對齊
        common_index = features.index.intersection(y.index)
        features = features.loc[common_index]
        y = y.loc[common_index]
        print(f"對齊後的維度: features: {features.shape}, y: {len(y)}")

    # 保存特徵生成器
    print("\n保存特徵生成器...")
    model_path = 'models/feature_generator.joblib'
    ensure_directory_exists(model_path)
    joblib.dump(generator, model_path)
    print(f"特徵生成器已保存為 '{model_path}'")

    return features, y

def find_optimal_threshold(model, x_val, y_val):
    """尋找最佳分類閾值"""
    print("\n尋找最佳分類閾值...")

    # 獲取預測概率
    y_pred_proba = model.predict_proba(x_val)[:, 1]

    # 計算精確率、召回率和閾值
    precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)

    # 計算F1分數
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)  # 避免除以零

    # 找到F1分數最大的閾值
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = (thresholds[best_threshold_idx]
                     if best_threshold_idx < len(thresholds)
                     else thresholds[-1])

    # 計算最佳閾值下的性能指標
    y_pred_optimal = (y_pred_proba > best_threshold).astype(int)
    optimal_precision = precision_score(y_val, y_pred_optimal)
    optimal_recall = recall_score(y_val, y_pred_optimal)
    optimal_f1 = f1_score(y_val, y_pred_optimal)

    print(f"最佳閾值: {best_threshold:.4f}")
    print(f"最佳閾值下的精確率: {optimal_precision:.4f}")
    print(f"最佳閾值下的召回率: {optimal_recall:.4f}")
    print(f"最佳閾值下的F1分數: {optimal_f1:.4f}")

    return best_threshold

def train_model(features, y):
    """訓練XGBoost模型"""
    print("\n開始訓練XGBoost模型...")

    # 檢查類別分布
    print("\n目標變量分布:")
    print(y.value_counts(normalize=True))

    # 移除ACCT_NBR欄位（如果存在）
    if 'ACCT_NBR' in features.columns:
        print("移除ACCT_NBR欄位用於訓練...")
        features = features.drop('ACCT_NBR', axis=1)

    # 檢查特徵相關性
    print("\n檢查特徵與目標變量的相關性...")
    correlations = pd.DataFrame()
    for col in features.columns:
        if features[col].dtype in ['int64', 'float64']:
            corr = features[col].corr(y)
            correlations = pd.concat([
                correlations,
                pd.DataFrame({'feature': [col], 'correlation': [abs(corr)]})
            ])

    correlations = correlations.sort_values('correlation', ascending=False)
    print("\n與目標變量相關性最高的前5個特徵:")
    print(correlations.head())

    # 檢查是否有極高相關性的特徵（可能表示數據洩漏）
    high_corr_features = correlations[correlations['correlation'] > 0.9]
    if not high_corr_features.empty:
        print("\n警告：發現與目標變量高度相關的特徵（相關性 > 0.9）:")
        print(high_corr_features)

    # 分割訓練集和測試集
    x_train, x_test, y_train, y_test = train_test_split(
        features, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 進一步分割訓練集得到驗證集
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

    # 設置XGBoost參數
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 7,
        'learning_rate': 0.03,
        'n_estimators': 1500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': 10,
        'early_stopping_rounds': 50,
        'random_state': 42
    }

    # 訓練模型
    print("\n開始訓練...")
    model = xgb.XGBClassifier(**params)
    eval_set = [(x_val, y_val)]
    model.fit(
        x_train, y_train,
        eval_set=eval_set,
        verbose=100
    )
    print("訓練完成！")

    # 特徵重要性分析
    feature_importance = pd.DataFrame({
        'feature': x_train.columns,
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    print("\n前10個最重要的特徵:")
    print(feature_importance.head(10))

    # 尋找最佳閾值
    best_threshold = find_optimal_threshold(model, x_val, y_val)

    # 保存模型
    print("\n保存模型...")
    model_path = 'models/fraud_detection_model.joblib'
    ensure_directory_exists(model_path)
    joblib.dump(model, model_path)
    print(f"模型已保存為 '{model_path}'")

    # 保存最佳閾值
    threshold_path = 'models/optimal_threshold.txt'
    ensure_directory_exists(threshold_path)
    with open(threshold_path, 'w', encoding='utf-8') as f:
        f.write(str(best_threshold))
    print(f"最佳閾值已保存為 '{threshold_path}'")

    return model, x_test, y_test, best_threshold

def evaluate_model(model, x_test, y_test, threshold=0.5):
    """評估模型性能"""
    print("\n評估模型性能...")

    # 預測
    y_pred = model.predict_proba(x_test)[:, 1]
    y_pred_binary = (y_pred > threshold).astype(int)

    # 混淆矩陣
    cm = confusion_matrix(y_test, y_pred_binary)
    print("\n混淆矩陣:")
    print(cm)
    print("\n各類別的樣本數:")
    print(f"True Negatives: {cm[0,0]}")
    print(f"False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}")
    print(f"True Positives: {cm[1,1]}")

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred_binary),
        'Precision': precision_score(y_test, y_pred_binary),
        'Recall': recall_score(y_test, y_pred_binary),
        'F1-score': f1_score(y_test, y_pred_binary),
        'AUC-ROC': roc_auc_score(y_test, y_pred)
    }

    print("\n模型評估指標:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # 檢查預測概率分布
    print("\n預測概率分布:")
    print(pd.Series(y_pred).describe())

def main():
    """主函數"""
    # 載入數據
    df = load_data(TRAIN_DATA)

    # 準備特徵
    features, y = prepare_features(df)

    # 訓練模型
    model, x_test, y_test, best_threshold = train_model(features, y)

    # 評估模型
    evaluate_model(model, x_test, y_test, threshold=best_threshold)

if __name__ == "__main__":
    main()
