"""
金融詐騙檢測流程主程式：整合訓練、預測和評估功能。
"""
import time
import warnings
import sys
from dotenv import load_dotenv

import train
import predict
import evaluate_results
import predict_test
from service import claude_analyzer

# 載入環境變數
load_dotenv()

# 設定警告過濾
warnings.filterwarnings('ignore')


def display_header(title):
    """顯示帶格式的標題"""
    line = "=" * 80
    print("\n" + line)
    print(f" {title} ".center(80, "*"))
    print(line + "\n")


def run_train():
    """執行訓練流程"""
    display_header("Step 1: 訓練詐騙檢測模型")

    print("正在執行模型訓練流程...")
    # 呼叫 train.py 中的主函數
    train.main()

    print("\n模型訓練完成！")


def run_predict():
    """執行預測流程"""
    display_header("Step 2: 執行詐騙風險預測")

    print("正在執行詐騙預測流程...")
    # 呼叫 predict.py 中的主函數
    predict.main()

    print("\n詐騙預測完成！")


def run_evaluate():
    """執行評估流程"""
    display_header("Step 3: 評估預測效能")

    print("正在評估預測結果...")

    # 呼叫 evaluate_results.py 中的主函數
    evaluation_results = evaluate_results.main()

    print("\n評估完成！")
    return evaluation_results


def run_test_predict():
    """執行測試數據預測流程"""
    display_header("Step 4: 執行測試數據預測")

    print("正在執行測試數據預測流程...")
    # 呼叫 predict_test.py 中的主函數
    predict_test.main()

    print("\n測試數據預測完成！")


def main():
    """主程式：整合訓練、預測和評估流程"""
    start_time = time.time()

    display_header("金融詐騙檢測流程")
    print("即將開始執行完整流程，包含訓練、預測、評估、AI分析和測試預測五個步驟。")

    try:
        # Step 1: 訓練模型
        run_train()

        # Step 2: 進行預測
        run_predict()

        # Step 3: 評估結果
        evaluation_results = run_evaluate()

        # Step 4: 使用Claude分析評估結果
        user_input = input("\n是否要使用Claude分析評估結果? (y/n): ")
        if user_input.lower() == 'y':
            display_header("Step 4: Claude AI 詐騙檢測分析")
            claude_analyzer.analyze_fraud_detection_results(evaluation_results)

        # Step 5: 執行測試數據預測
        user_input = input("\n是否要執行測試數據預測? (y/n): ")
        if user_input.lower() == 'y':
            run_test_predict()

        # 顯示總執行時間
        total_time = time.time() - start_time
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)

        display_header("流程執行完成")
        print(f"總執行時間: {minutes} 分 {seconds} 秒")
        print("訓練模型、預測詐騙風險、評估效能、AI分析和測試預測的完整流程已順利完成。")

    except (ValueError, IOError, ImportError, RuntimeError) as e:
        print("\n執行過程中發生錯誤:")
        print(f"錯誤類型: {type(e).__name__}")
        print(f"錯誤訊息: {str(e)}")
        print("\n請檢查日誌和錯誤訊息以進行故障排除。")
        return 1

    return 0


if __name__ == "__main__":
    EXIT_CODE = main()
    sys.exit(EXIT_CODE)
