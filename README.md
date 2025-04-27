# 台新金控黑客松競賽資料結構說明

本次資料集分為 **TRAIN** 與 **TEST**，用於識別潛在詐騙帳戶。資料表結構如下：

## 💻 環境需求

- Python 3.12.1
- 相關套件需求請參考 `requirements.txt` 檔案

## 📂 資料集設置

本專案在 GitHub 上不包含完整資料集。克隆專案後，請依照以下步驟設置資料集：

1. 在專案根目錄中，確保存在 `dataset` 資料夾並包含以下子目錄結構：
   ```
   dataset/
   ├── Output/     # 輸出結果目錄
   ├── Prepare/    # 預處理資料目錄
   ├── Test/       # 測試資料集目錄
   └── Train/      # 訓練資料集目錄
   ```

2. 將原始資料集檔案放置於對應目錄：
   - 將訓練資料（`SAV_TXN_Data_202412.csv`、`ID_Data_202412.csv`、`ACCTS_Data_202412.csv`、`ECCUS_Data_202412.csv`）放入 `dataset/Train/` 目錄
   - 將測試資料（`SAV_TXN_Data_202501.csv`、`ID_Data_202501.csv`、`ACCTS_Data_202501.csv`）放入 `dataset/Test/` 目錄

3. 使用資料合併工具（如Data Wrangler）手動合併資料：
   - 將上述各訓練資料合併後，手動將合併結果儲存為 `dataset/Prepare/train_data.csv`
   - 將上述各測試資料合併後，手動將合併結果儲存為 `dataset/Prepare/test_data.csv`
   - 注意：本專案不使用自動化腳本進行資料合併，需手動完成此步驟

4. 確認 `.env` 檔案中的資料路徑設定正確：
   ```
   # 訓練相關路徑
   TRAIN_DATA_PATH=dataset/Prepare/train_data.csv
   PREDICT_DATA_PATH=dataset/Prepare/predict_data.csv
   ECCUS_DATA_PATH=dataset/Train/(Train)ECCUS_Data_202412.csv

   # 測試相關路徑
   TEST_DATA_PATH=dataset/Test/test_data.csv
   TEST_ANS_FILE=dataset/Test/(Test)ANS_參賽者_202501.csv
   
   # 輸出路徑
   OUTPUT_PATH=dataset/Output/account_risk_assessment.csv
   
   # AWS配置（若需要使用AI分析功能）
   AWS_ACCESS_KEY_ID=您的AWS存取金鑰ID
   AWS_SECRET_ACCESS_KEY=您的AWS私密存取金鑰
   AWS_DEFAULT_REGION=us-west-2
   AWS_BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
   ```

如未包含原始資料集，可向主辦單位索取或參考比賽規則獲取資料集來源。

## 📁 資料表清單

| 類別 | 筆數 | 時間 | 檔名 | 說明 |
|------|------|------|------|------|
| TRAIN | 206,333 | 2024年12月 | `SAV_TXN_Data_202412` | 交易明細表 |
|       |        |             | `ID_Data_202412`      | 客戶資料表 |
|       |        |             | `ACCTS_Data_202412`   | 帳戶列表 |
|       |        |             | `ECCUS_Data_202412`   | 警示戶列表 |
| TEST  | 52,122 | 2025年1月   | `SAV_TXN_Data_202501` | 交易明細表 |
|       |        |             | `ID_Data_202501`      | 客戶資料表 |
|       |        |             | `ACCTS_Data_202501`   | 帳戶列表 |

---

## 📄 交易明細表 SAV_TXN_Data

| 欄位 | 說明 | 範例 |
|------|------|------|
| ACCT_NBR | 帳號 | ACCT6068 |
| CUST_ID | 客戶 ID | ID5684 |
| TX_DATE | 交易日期 | 18264 |
| TX_TIME | 交易時間 (0-23 小時) | 7 |
| DRCR | 傳入/傳出 (1:入, 0:出) | 1 |
| TX_AMT | 交易金額 | 250 |
| PB_BAL | 交易後餘額 | 100 |
| OWN_TRANS_ACCT | 對手帳號 | ACCT31429 |
| OWN_TRANS_ID | 對手 ID | ID99999 |
| CHANNEL_CODE | 交易通路代碼 | 10 |
| TRN_CODE | 交易類型代碼 | 6 |
| BRANCH_NO | 分行代號 | B3 |
| EMP_NO | 櫃員代號 | E2590 |
| mb_check | 行動銀行風險評分 | 9 |
| eb_check | 網路銀行風險評分 | 5 |
| SAME_NUMBER_IP | 是否 IP 共用 | 0 |
| SAME_NUMBER_UUID | 是否 UUID 共用 | 1 |
| DAY_OF_WEEK | 星期幾 | Monday |

---

## 👤 客戶資料表 ID_Data

| 欄位 | 說明 | 範例 |
|------|------|------|
| CUST_ID | 客戶 ID | ID5684 |
| AUM_AMT | 客戶資產總額 | 25,034,921 |
| DATE_OF_BIRTH | 年齡 | 30 |
| YEARLYINCOMELEVEL | 年收入等級碼 | 125 |
| CNTY_CD | 地區代碼 | 12 |

---

## ⚠️ 警示戶資料表 ECCUS_Data

| 欄位 | 說明 | 範例 |
|------|------|------|
| CUST_ID | 客戶 ID | ID27476 |
| ACCT_NBR | 帳號 | ACCT28942 |
| DATA_DT | 通報日期 | 18265 |

---

## 🏦 帳戶列表 ACCTS_Data

| 欄位 | 說明 | 範例 |
|------|------|------|
| ACCT_NBR | 帳號 | ACCT6068 |
| CUST_ID | 客戶 ID | ID5684 |
| CANCEL_NO_CONTACT | 是否久未往來後解除 | 1 |
| IS_DIGITAL | 是否為數位帳戶 | 0 |
| ACCT_OPEN_DT | 開戶日期 | 8400 |

---

## 🧠 比賽說明摘要（台新金控命題）

- **主題**：金融詐欺 AI 神探出擊
- **目標**：利用生成式 AI 技術識別詐騙帳戶
- **可使用技術**：LLM、RAG、Agent、傳統 ML 模型等
- **評分面向**：
  - 技術架構合理性（25%）
  - 識別能力（F1 分數，50%）
  - 呈現方式（含解釋性）（5%）
  - 系統完成度（15%）
  - 創意度（5%）
  - 主題切合度（4%）
  - 使用 Amazon Nova 模型（加分項 1%）

**📊 F1 計算公式**：
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
```

**資料來源**：[命題文件](https://reurl.cc/9DVvon)

---

## 🔄 系統流程說明

本專案實作了一套完整的金融詐騙偵測系統，包含四個主要階段：模型訓練、詐騙預測、評估結果及AI分析。整體流程由 `main.py` 進行統一調度，以下為各階段詳細說明：

### 執行方式

```bash
python main.py
```

### 流程架構

1. **模型訓練階段** (`train.py`)
   - 載入訓練資料集
   - 生成詐騙偵測特徵
   - 訓練 XGBoost 分類模型
   - 評估模型在測試集上的效能
   - 將模型及特徵生成器儲存至 `models/` 目錄

2. **詐騙預測階段** (`predict.py`)
   - 載入測試資料集
   - 使用儲存的特徵生成器生成特徵
   - 使用儲存的模型進行預測
   - 計算每個帳戶的風險指標
   - 將結果保存至 `dataset/Output/account_risk_assessment.csv`

3. **評估結果階段** (`evaluate_results.py`)
   - 載入實際詐騙帳號清單
   - 載入預測結果
   - 計算精確率、召回率和 F1 分數
   - 分析預測分數分布和錯誤類型
   - 將評估結果保存至 `dataset/Output/evaluation_results.json`

4. **AI分析階段** (`service/claude_analyzer.py`) **新功能**
   - 使用 Amazon Bedrock Claude 3.5 Sonnet V2 模型
   - 分析詐騙檢測評估結果
   - 提供模型改進建議和風險分析
   - 生成詐騙偵測專家解讀報告

## 💡 實作技術細節

### 特徵工程（`generator/full_auto_feature_generator.py`）

此模組實作了自動特徵生成器，從交易數據中提取以下五類關鍵特徵：

1. **金額相關特徵**
   - 交易金額的統計特徵（平均值、標準差、最大值、最小值、總和）
   - 交易金額的異常程度（Z-score）
   - 大額交易比例及餘額利用率

2. **時間模式特徵**
   - 高風險時段交易（凌晨 0-6 點）的比例及數量
   - 交易間隔統計（平均、標準差、最小值）
   - 日交易頻率變化

3. **通道行為特徵**
   - 高風險通道使用比例（如：ATM、網銀、行動銀行）
   - 通道切換頻率及通道多樣性
   - 主要使用通道

4. **交易對手特徵**
   - 交易對手數量及多樣性比率
   - 交易對手集中度（最常交易對手的佔比）
   - 獨特交易對手比例

5. **風險指標特徵**
   - 設備重複使用（IP、UUID）指標
   - 帳戶異常行為標記（是否久未往來後解除、數位帳戶比例）
   - 綜合風險分數計算

### 模型訓練與評估

專案使用 XGBoost 模型進行詐騙偵測，關鍵參數設定如下：

```python
params = {
    'objective': 'binary:logistic',    # 二元分類問題
    'eval_metric': 'auc',              # 使用 AUC 作為評估指標
    'max_depth': 7,                    # 樹的最大深度
    'learning_rate': 0.03,             # 學習率
    'n_estimators': 1500,              # 樹的數量
    'subsample': 0.8,                  # 樣本抽樣比例
    'colsample_bytree': 0.8,           # 特徵抽樣比例
    'scale_pos_weight': 10,            # 類別不平衡權重
    'early_stopping_rounds': 50,       # 提前停止訓練參數
    'random_state': 42                 # 隨機種子
}
```

模型評估使用混淆矩陣和多種指標（準確率、精確率、召回率、F1分數、AUC-ROC）進行全方位性能分析。

#### 自動最佳閾值尋找機制

系統實作了自動尋找分類閾值的功能，透過以下方法優化模型效能：

- 計算驗證集上不同閾值的精確率和召回率
- 尋找能使F1分數最大化的閾值
- 將最佳閾值儲存於 `models/optimal_threshold.txt`
- 在測試集和預測階段都使用此最佳閾值進行分類

此功能顯著提升了模型在不平衡資料上的表現，能更準確地識別少數類別（詐騙案例）。

### 風險評估標準

在預測階段，系統根據以下標準識別高風險帳戶：
- 使用自動尋找的最佳閾值進行分類
- 綜合考量交易行為異常度、時間模式、通道使用和交易對手特徵
- 計算並輸出每個帳戶的詐騙風險概率值

最終評估使用 F1 分數作為主要指標，平衡精確率與召回率，確保模型能有效偵測真正的詐騙帳戶，同時減少誤判。

### AI分析功能

系統使用 Amazon Bedrock Claude 3.5 Sonnet V2 模型進行詐騙檢測分析的功能：

- **分析內容**：模型表現、詐騙模式、錯誤案例
- **提供建議**：模型優化策略、閾值調整建議
- **專家解讀**：從金融專業角度解讀詐騙行為特徵
- **風險評估**：提供整體詐騙風險評估報告

AI分析功能使用 AWS Bedrock Runtime API 實現，可以針對不同詐騙場景提供專業分析和建議。
