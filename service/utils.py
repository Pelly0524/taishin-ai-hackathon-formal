"""
通用工具函數模組：提供專案中常用的工具函數
"""
import os

def ensure_directory_exists(file_path):
    """
    確保目錄存在，如果不存在則創建
    
    Args:
        file_path (str): 檔案路徑，函數會檢查其目錄部分
        
    Returns:
        None
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        print(f"目錄 '{directory}' 不存在，正在創建...")
        os.makedirs(directory)
        print(f"已成功創建目錄 '{directory}'")

def load_optimal_threshold():
    """
    載入最佳閾值，如果檔案不存在或無法讀取則使用默認值 0.5
    
    Returns:
        float: 預測模型的最佳閾值
    """
    try:
        threshold_path = 'models/optimal_threshold.txt'
        with open(threshold_path, 'r', encoding='utf-8') as f:
            threshold = float(f.read().strip())
        print(f"已載入最佳閾值: {threshold:.4f}")
        return threshold
    except FileNotFoundError:
        print("警告：找不到最佳閾值文件，將使用默認閾值 0.5")
        return 0.5
    except (ValueError, UnicodeDecodeError, PermissionError) as e:
        print(f"載入閾值時發生錯誤: {str(e)}，將使用默認閾值 0.5")
        return 0.5
