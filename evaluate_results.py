"""
模型評估模組：評估詐騙偵測模型效能，計算準確率、召回率和F1分數等指標。
"""
import json
import os
import warnings
from dotenv import load_dotenv
import pandas as pd
from service.utils import ensure_directory_exists, load_optimal_threshold

# 載入環境變數
load_dotenv()

# 常量設定
ECCUS_DATA = os.environ.get('ECCUS_DATA_PATH')

warnings.filterwarnings('ignore')

def load_actual_fraud_accounts():
    """載入實際詐騙帳號清單"""
    print("載入實際詐騙帳號...")
    fraud_df = pd.read_csv(ECCUS_DATA)
    fraud_accounts = set(fraud_df['ACCT_NBR'].unique())
    print(f"實際詐騙帳號數量: {len(fraud_accounts)}")
    return fraud_accounts

def load_predicted_results():
    """載入預測結果"""
    print("\n載入預測結果...")
    pred_df = pd.read_csv('dataset/Output/account_risk_assessment.csv')

    # 載入最佳閾值
    threshold = load_optimal_threshold()

    # 將index列名改為ACCT_NBR（如果需要）
    if 'Unnamed: 0' in pred_df.columns:
        pred_df = pred_df.rename(columns={'Unnamed: 0': 'ACCT_NBR'})

    # 根據閾值標記預測的詐騙帳號
    predicted_fraud = pred_df[
        (pred_df['fraud_prob'] > threshold)
    ]

    predicted_accounts = set(predicted_fraud['ACCT_NBR'])
    print(f"使用閾值 {threshold:.4f} 預測為詐騙的帳號數量: {len(predicted_accounts)}")
    return pred_df, predicted_accounts

def evaluate_predictions(actual_fraud, predicted_fraud, all_predictions_df):
    """評估預測結果"""
    print("\n評估預測結果...")

    # 計算各種指標
    true_positives = len(actual_fraud & predicted_fraud)
    false_positives = len(predicted_fraud - actual_fraud)
    false_negatives = len(actual_fraud - predicted_fraud)

    # 獲取所有預測過的帳號
    all_accounts = set(all_predictions_df['ACCT_NBR'])
    true_negatives = len(all_accounts - actual_fraud - predicted_fraud)

    # 打印詳細結果
    print("\n預測結果分析:")
    print(f"正確預測的詐騙帳號 (True Positives): {true_positives}")
    print(f"錯誤預測為詐騙 (False Positives): {false_positives}")
    print(f"未被發現的詐騙帳號 (False Negatives): {false_negatives}")
    print(f"正確預測的正常帳號 (True Negatives): {true_negatives}")

    # 計算準確率、召回率等指標
    precision = (true_positives / (true_positives + false_positives)
                if (true_positives + false_positives) > 0 else 0)
    recall = (true_positives / (true_positives + false_negatives)
              if (true_positives + false_negatives) > 0 else 0)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n模型效能指標:")
    print(f"準確率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分數: {f1_score:.4f}")

    # 顯示未被發現的詐騙帳號
    missed_fraud = actual_fraud - predicted_fraud
    if missed_fraud:
        print("\n未被發現的詐騙帳號:")
        print(list(missed_fraud)[:10])  # 只顯示前10個
        if len(missed_fraud) > 10:
            print(f"... 以及其他 {len(missed_fraud)-10} 個帳號")

    # 顯示誤判為詐騙的帳號
    false_alarms = predicted_fraud - actual_fraud
    if false_alarms:
        print("\n誤判為詐騙的帳號:")
        print(list(false_alarms)[:10])  # 只顯示前10個
        if len(false_alarms) > 10:
            print(f"... 以及其他 {len(false_alarms)-10} 個帳號")

    # 分析預測分數分布
    print("\n預測分數分析:")
    fraud_scores = all_predictions_df[all_predictions_df['ACCT_NBR'].isin(actual_fraud)]
    non_fraud_scores = all_predictions_df[~all_predictions_df['ACCT_NBR'].isin(actual_fraud)]

    print("\n實際詐騙帳號的預測分數:")
    print(fraud_scores['fraud_prob'].describe())

    print("\n正常帳號的預測分數:")
    print(non_fraud_scores['fraud_prob'].describe())

    # 構建評估結果數據
    result_data = {
        "confusion_matrix": {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives
        },
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        },
        "dataset_stats": {
            "total_accounts": len(all_accounts),
            "actual_fraud_count": len(actual_fraud),
            "predicted_fraud_count": len(predicted_fraud)
        },
        "fraud_scores_stats": {
            "actual_fraud": json.loads(
                fraud_scores[['fraud_prob']].describe().to_json()
            ),
            "non_fraud": json.loads(
                non_fraud_scores[['fraud_prob']].describe().to_json()
            )
        },
        "missed_fraud_sample": list(missed_fraud)[:10] if missed_fraud else [],
        "false_alarms_sample": list(false_alarms)[:10] if false_alarms else []
    }

    return result_data

def main():
    """主函數：載入實際詐騙資料、預測結果並進行評估"""
    # 載入實際詐騙帳號
    actual_fraud = load_actual_fraud_accounts()

    # 載入預測結果
    pred_df, predicted_fraud = load_predicted_results()

    # 評估預測結果
    evaluation_results = evaluate_predictions(actual_fraud, predicted_fraud, pred_df)

    # 保存評估結果為JSON文件，以便其他模組使用
    output_file = 'dataset/Output/evaluation_results.json'
    ensure_directory_exists(output_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
    return evaluation_results

if __name__ == "__main__":
    main()
