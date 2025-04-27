"""
Claude 回應分析模組: 提供分析和處理 Amazon Bedrock Claude 回應的功能。
"""
import json
import re
import os
from typing import Dict, Any
import boto3

def parse_response(response_str: str) -> Dict[str, Any]:
    """
    解析 Claude 的 JSON 回應

    Args:
        response_str: Claude API 回傳的 JSON 字串

    Returns:
        解析後的字典
    """
    try:
        response_dict = json.loads(response_str)
        return response_dict
    except json.JSONDecodeError:
        # 處理非 JSON 格式的回應
        return {
            "error": "回應格式錯誤，無法解析為 JSON",
            "raw_response": response_str
        }

def extract_content(response_dict: Dict[str, Any]) -> str:
    """
    從 Claude 回應字典中提取主要內容

    Args:
        response_dict: 已解析的回應字典

    Returns:
        提取的文字內容
    """
    try:
        # 檢查是否為新版 Anthropic 格式
        if "content" in response_dict:
            if isinstance(response_dict["content"], list):
                text_contents = []
                for item in response_dict["content"]:
                    if item.get("type") == "text":
                        text_contents.append(item.get("text", ""))
                return "\n".join(text_contents)
            return str(response_dict["content"])

        # 檢查是否為舊版格式
        if "completion" in response_dict:
            return response_dict["completion"]

        # 處理 Bedrock 特有的格式
        if "message" in response_dict:
            return response_dict["message"].get("content", "")

        # 如果找不到已知格式，返回原始字典
        return json.dumps(response_dict, ensure_ascii=False, indent=2)
    except (TypeError, KeyError, AttributeError) as e:
        return f"提取內容時發生錯誤: {str(e)}"

def count_tokens(text: str) -> int:
    """
    簡易估算文本的 token 數量

    Args:
        text: 要計算的文本

    Returns:
        估算的 token 數量
    """
    # 簡易估算: 一個中文字約為 1.5 個 token，空格和標點為 0.5 個 token
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    other_chars = len(text) - len(chinese_chars)
    return int(len(chinese_chars) * 1.5 + other_chars * 0.5)

def analyze_sentiment(content: str) -> Dict[str, Any]:
    """
    分析回應內容的情感和語氣

    Args:
        content: 回應內容

    Returns:
        包含情感分析結果的字典
    """
    # 簡易情感分析 - 檢查正面和負面關鍵詞
    positive_words = ["優點", "好處", "有利", "優勢", "便利", "效能", "優化", "成功"]
    negative_words = ["問題", "缺點", "限制", "困難", "風險", "失敗", "錯誤"]

    positive_count = sum(1 for word in positive_words if word in content)
    negative_count = sum(1 for word in negative_words if word in content)

    total = positive_count + negative_count
    if total == 0:
        sentiment = "中性"
        score = 0
    else:
        sentiment = ("正面" if positive_count > negative_count
                   else "負面" if negative_count > positive_count
                   else "中性")
        score = (positive_count - negative_count) / total

    return {
        "sentiment": sentiment,
        "score": score,
        "positive_count": positive_count,
        "negative_count": negative_count
    }

def analyze_response(response_str: str) -> Dict[str, Any]:
    """
    全面分析 Claude 的回應

    Args:
        response_str: Claude API 回傳的回應字串

    Returns:
        包含分析結果的字典
    """
    # 解析回應
    response_dict = parse_response(response_str)

    # 檢查是否有錯誤
    if "error" in response_dict:
        return response_dict

    # 提取內容
    content = extract_content(response_dict)

    # 計算字數和 token 數
    char_count = len(content)
    token_count = count_tokens(content)

    # 進行情感分析
    sentiment_analysis = analyze_sentiment(content)

    # 返回分析結果
    return {
        "content": content,
        "char_count": char_count,
        "token_count": token_count,
        "sentiment_analysis": sentiment_analysis,
        "raw_response": response_dict
    }

def query_claude(
    query_text="請簡要說明 Amazon Bedrock 的優點。",
    max_tokens=300,
    temperature=0.5,
    verbose=True
):
    """
    向 Claude 發送查詢並獲取回應

    Args:
        query_text: 要發送給 Claude 的查詢內容
        max_tokens: 回應的最大 token 數
        temperature: 溫度參數，控制隨機性 (0.0-1.0)
        verbose: 是否輸出過程信息

    Returns:
        包含 Claude 回應分析結果的字典
    """
    if verbose:
        print("正在向 Claude 發送查詢...")
        print(f"查詢內容: {query_text}")

    # 建立 Bedrock Runtime 客戶端
    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=os.getenv("AWS_DEFAULT_REGION")
    )

    # 準備 payload
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {
                "role": "user",
                "content": query_text
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    # 呼叫 API
    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        body=json.dumps(payload),
        contentType="application/json",
        accept="application/json"
    )

    # 獲取回應
    response_body = response['body'].read().decode('utf-8')

    # 分析回應
    if verbose:
        print("\n正在分析 Claude 回應...")
    analysis = analyze_response(response_body)

    # 如果需要，顯示分析結果
    if verbose:
        print("\nClaude 回應分析結果:")
        print(f"內容:\n{analysis['content']}")
        print(f"\n字數: {analysis['char_count']}")
        print(f"預估 Token 數: {analysis['token_count']}")
        print(f"情感分析: {analysis['sentiment_analysis']['sentiment']} "
              f"(分數: {analysis['sentiment_analysis']['score']:.2f})")

    return analysis

def analyze_fraud_detection_results(evaluation_results, verbose=True):
    """
    使用Claude分析詐騙檢測評估結果

    Args:
        evaluation_results: 詐騙檢測評估結果字典
        verbose: 是否顯示詳細輸出

    Returns:
        包含Claude分析結果的字典
    """
    if verbose:
        print("開始分析詐騙檢測評估結果...")

    # 生成適合Claude分析的查詢文本
    metrics = evaluation_results['metrics']
    confusion = evaluation_results['confusion_matrix']
    stats = evaluation_results['dataset_stats']

    query_text = f"""
我正在進行金融詐騙檢測模型評估，以下是我的模型評估結果，請分析這些結果並提供改進建議：

模型效能指標:
- 準確率 (Precision): {metrics['precision']:.4f}
- 召回率 (Recall): {metrics['recall']:.4f}
- F1分數: {metrics['f1_score']:.4f}

混淆矩陣：
- 正確預測的詐騙帳號 (True Positives): {confusion['true_positives']}
- 錯誤預測為詐騙 (False Positives): {confusion['false_positives']}
- 未被發現的詐騙帳號 (False Negatives): {confusion['false_negatives']}
- 正確預測的正常帳號 (True Negatives): {confusion['true_negatives']}

數據集統計：
- 總帳號數: {stats['total_accounts']}
- 實際詐騙帳號數: {stats['actual_fraud_count']}
- 預測為詐騙的帳號數: {stats['predicted_fraud_count']}

請針對以下問題進行分析：
1. 此模型的整體表現如何？主要優點和問題在哪裡？
2. 如何改進模型以提高召回率和準確率？
3. 哪些因素可能導致了錯誤預測或未能識別的詐騙案例？
4. 基於這些指標，模型在實際業務場景中的應用價值和風險是什麼？
5. 請提供3-5點具體的改進建議。
"""

    if verbose:
        print("正在向Claude發送詐騙檢測評估分析請求...")

    # 使用query_claude發送請求並獲取回應
    result = query_claude(query_text, max_tokens=1500, verbose=verbose)

    # 將Claude的分析結果保存到文件
    output_path = 'dataset/Output/claude_analysis.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"分析結果已保存至: {output_path}")

    return result

def main():
    """
    模組主函數 - 可獨立運行進行測試
    """
    # 測試 query_claude 函數
    print("測試 Claude API 調用:")
    query_claude("請簡要說明 Amazon Bedrock 的優點。", verbose=True)

    # 測試本地分析功能
    print("\n\n測試本地分析功能:")
    test_response = (
        '{"id":"msg_01ABC123","message":{'
        '"role":"assistant",'
        '"content":"Amazon Bedrock 的主要優點包括:\\n\\n'
        '1. 提供多種高性能 AI 模型選擇，包括 Anthropic Claude, Meta Llama, Amazon Titan 等\\n'
        '2. 無需自行管理基礎設施即可使用最先進的 AI 模型\\n'
        '3. 企業級安全性和隱私保護\\n'
        '4. 可根據業務需求進行模型微調和客製化\\n'
        '5. 與 AWS 生態系統深度整合，便於與其他服務協同運作"'
        '},"type":"message"}'
    )

    # 使用本地測試數據進行分析
    local_result = analyze_response(test_response)

    if 'content' in local_result:
        # 顯示分析結果
        print("Claude 回應分析結果 (本地測試):")
        print(f"內容: {local_result['content'][:100]}...")
        print(f"字數: {local_result['char_count']}")
        print(f"預估 Token 數: {local_result['token_count']}")
        print(f"情感分析: {local_result['sentiment_analysis']['sentiment']} "
              f"(分數: {local_result['sentiment_analysis']['score']:.2f})")
    else:
        print("分析本地測試資料失敗:", local_result)

if __name__ == "__main__":
    main()
