"""
Claude API測試模組：展示如何調用和分析Claude回應的各種用例
"""
import os
import json
import boto3
from dotenv import load_dotenv
from service import claude_analyzer

# 讀取 .env 檔案
load_dotenv()

# 建立 Bedrock Runtime 客戶端
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=os.getenv("AWS_DEFAULT_REGION")
)

# 準備 payload，加上必須的 anthropic_version
payload = {
    "anthropic_version": "bedrock-2023-05-31",
    "messages": [
        {
            "role": "user",
            "content": "請簡要說明 Amazon Bedrock 的優點。"
        }
    ],
    "max_tokens": 200,
    "temperature": 0.5
}

# 呼叫新版 API
response = bedrock.invoke_model(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    body=json.dumps(payload),
    contentType="application/json",
    accept="application/json"
)

# 解析結果
response_body = response['body'].read().decode('utf-8')
print("原始回應:")
print(response_body)

# 使用分析模組分析回應
print("\n使用分析模組處理回應:")
analysis = claude_analyzer.analyze_response(response_body)

# 顯示分析結果
print("\nClaude 回應分析結果:")
print(f"內容:\n{analysis['content']}")
print(f"\n字數: {analysis['char_count']}")
print(f"預估 Token 數: {analysis['token_count']}")
print(f"情感分析: {analysis['sentiment_analysis']['sentiment']} (分數: {analysis['sentiment_analysis']['score']:.2f})") 