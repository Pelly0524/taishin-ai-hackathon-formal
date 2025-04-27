"""
測試數據預測模組：載入測試數據，生成特徵，並使用訓練好的模型進行預測與評估。
"""
import os
import warnings
from dotenv import load_dotenv
import pandas as pd
import joblib
import boto3
from generator.full_auto_feature_generator import FraudDetectionFeatureGenerator
from service.utils import load_optimal_threshold

# 載入環境變數
load_dotenv()

# 常量設定
TEST_DATA = os.environ.get('TEST_DATA_PATH')
TEST_ANS_FILE = 'dataset/Test/(Test)ANS_參賽者_202501.csv'
MODEL_PATH = 'models/fraud_detection_model.joblib'
OUTPUT_PATH = os.environ.get('OUTPUT_PATH', 'dataset/Output')
S3_BUCKET = os.environ.get('S3_BUCKET', 'your-bucket-name')
S3_PREFIX = os.environ.get('S3_PREFIX', 'predictions/')

warnings.filterwarnings('ignore')

def load_test_data(file_path):
    """載入測試數據"""
    print("載入測試數據...")
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

        print(f"測試數據維度: {df.shape}")
        return df

    except Exception as e:
        print(f"讀取測試數據時發生錯誤: {str(e)}")
        raise

def prepare_test_features(df):
    """準備測試數據特徵"""
    print("\n準備測試數據特徵...")

    # 確保必要欄位存在
    required_columns = {
        'ACCT_NBR', 'TX_AMT', 'TX_DATE', 'TX_TIME', 'CHANNEL_CODE',
        'OWN_TRANS_ACCT', 'PB_BAL', 'SAME_NUMBER_IP', 'SAME_NUMBER_UUID',
        'CANCEL_NO_CONTACT', 'IS_DIGITAL'
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"缺少必要欄位: {missing_columns}")

    # 載入特徵生成器
    print("\n載入特徵生成器...")
    try:
        generator = joblib.load('models/feature_generator.joblib')
        print("成功載入特徵生成器")
    except FileNotFoundError:
        print("警告：找不到保存的特徵生成器，將創建新的生成器")
        generator = FraudDetectionFeatureGenerator()

    # 生成特徵
    features = generator.generate_features(df)

    # 保存ACCT_NBR
    account_numbers = None
    if 'ACCT_NBR' in features.columns:
        account_numbers = features['ACCT_NBR'].copy()
        features = features.drop('ACCT_NBR', axis=1)

    # 檢查並轉換數據類型
    for col in features.columns:
        if features[col].dtype == 'object':
            print(f"警告：列 {col} 是對象類型，將轉換為類別編碼")
            features[col] = pd.Categorical(features[col]).codes

    return features, account_numbers

def predict_test_fraud(model_path, features, account_numbers):
    """使用模型進行預測"""
    print("\n開始預測...")

    # 載入最佳閾值
    threshold = load_optimal_threshold()

    # 載入模型
    model = joblib.load(model_path)

    # 預測欺詐概率
    fraud_prob = model.predict_proba(features)[:, 1]
    print(f"完成 {len(fraud_prob)} 筆交易的風險預測")

    # 創建結果DataFrame，只包含帳號和風險值
    results = pd.DataFrame({
        'ACCT_NBR': account_numbers,
        'fraud_prob': fraud_prob,
        'is_fraud_predicted': (fraud_prob > threshold).astype(int)
    })

    # 提早聚合帳號的風險值，只保留每個帳號的第一筆交易
    # 先按帳號排序，然後只取每個帳號的第一筆交易
    results_sorted = results.sort_values('ACCT_NBR')
    account_risk = results_sorted.groupby('ACCT_NBR').first().round(4)

    # 根據風險程度排序
    account_risk = account_risk.sort_values('fraud_prob', ascending=False)

    # 標記高風險帳戶（欺詐概率超過閾值）
    high_risk = account_risk[
        account_risk['fraud_prob'] > threshold
    ]

    print(f"\n使用閾值 {threshold:.4f} 識別出 {len(high_risk)} 個高風險帳戶")
    print("\n前10個最高風險帳戶:")
    print(high_risk.head(10))

    return account_risk, high_risk

def upload_to_s3(file_path, s3_key):
    """上傳檔案到 S3"""
    try:
        s3_client = boto3.client('s3')
        s3_client.upload_file(file_path, S3_BUCKET, s3_key)
        print(f"成功上傳檔案到 S3: s3://{S3_BUCKET}/{s3_key}")
        return True
    except Exception as e:
        print(f"上傳檔案到 S3 時發生錯誤: {str(e)}")
        return False

def update_test_ans_file(test_ans_file, high_risk_accounts, account_risk):
    """更新測試答案文件"""
    print(f"\n更新測試答案文件: {test_ans_file}")
    
    try:
        # 讀取測試答案文件
        ans_df = pd.read_csv(test_ans_file)
        
        # 將高風險帳戶標記為 Y
        ans_df['Y'] = ans_df['ACCT_NBR'].apply(
            lambda x: 'Y' if x in high_risk_accounts.index else ''
        )
        
        # 使用環境變數中設定的輸出路徑
        output_dir = os.path.dirname(OUTPUT_PATH)
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成新檔案名稱
        file_name = os.path.basename(test_ans_file)
        name, ext = os.path.splitext(file_name)
        new_file_path = os.path.join(output_dir, f"{name}_predicted{ext}")
        
        # 保存為新文件
        ans_df.to_csv(new_file_path, index=False)
        print(f"成功將預測結果保存至新檔案：{new_file_path}")
        print(f"已標記 {len(high_risk_accounts)} 個高風險帳戶")
        
        # 上傳到 S3
        s3_key = f"{S3_PREFIX}{name}_predicted{ext}"
        upload_to_s3(new_file_path, s3_key)
        
        # 新增一個包含所有帳號風險值的輸出檔案
        risk_df = pd.DataFrame({
            'ACCT_NBR': account_risk.index,
            'fraud_prob': account_risk['fraud_prob'],
            'Y': account_risk.index.map(lambda x: 'Y' if x in high_risk_accounts.index else '')
        })
        
        # 重新排序欄位
        risk_df = risk_df[['ACCT_NBR', 'fraud_prob', 'Y']]
        
        # 按照風險值由高到低排序
        risk_df = risk_df.sort_values('fraud_prob', ascending=False)
        
        # 生成風險值檔案名稱
        risk_file_path = os.path.join(output_dir, f"{name}_risk{ext}")
        
        # 保存風險值檔案
        risk_df.to_csv(risk_file_path, index=False)
        print(f"成功將風險值結果保存至新檔案：{risk_file_path}")
        
        # 上傳到 S3
        s3_key = f"{S3_PREFIX}{name}_risk{ext}"
        upload_to_s3(risk_file_path, s3_key)
        
    except Exception as e:
        print(f"更新測試答案文件時發生錯誤: {str(e)}")
        raise

def main():
    """主函數"""
    # 載入測試數據
    test_df = load_test_data(TEST_DATA)

    # 準備特徵
    features, account_numbers = prepare_test_features(test_df)

    # 預測並分析結果
    account_risk, high_risk = predict_test_fraud(MODEL_PATH, features, account_numbers)

    # 更新測試答案文件
    update_test_ans_file(TEST_ANS_FILE, high_risk, account_risk)

if __name__ == "__main__":
    main()
