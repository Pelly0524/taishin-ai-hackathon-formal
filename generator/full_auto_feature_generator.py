"""
自動特徵生成模組：用於金融詐騙偵測的特徵工程，包含金額、時間模式、通道行為等特徵生成。
"""
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

class FraudDetectionFeatureGenerator:
    """
    詐騙偵測特徵生成器：自動從交易數據生成用於詐騙偵測的關鍵特徵。
    
    生成的特徵包含：
    - 金額相關特徵（異常交易、餘額比率）
    - 時間模式特徵（高風險時段、交易頻率）
    - 通道行為特徵（高風險通道、通道切換）
    - 交易對手特徵（新對手比例、集中度）
    - 風險指標特徵（設備重複使用、異常行為）
    """
    def __init__(self):
        # 定義高風險時段（凌晨時段）
        self.high_risk_hours = set(range(0, 6))
        # 定義高風險通道（ATM、網銀、行動銀行等）
        self.high_risk_channels = {'17', '15', '18'}

    def generate_amount_features(self, df):
        """
        金額相關特徵
        - 交易金額的統計特徵
        - 交易金額的異常程度
        - 餘額比率
        """
        features = pd.DataFrame()

        # 基本金額統計
        amount_stats = df.groupby('ACCT_NBR').agg({
            'TX_AMT': ['mean', 'std', 'max', 'min', 'sum'],
            'PB_BAL': ['mean', 'min']
        }).round(2)

        amount_stats.columns = [
            'avg_tx_amount', 'std_tx_amount', 'max_tx_amount',
            'min_tx_amount', 'total_tx_amount',
            'avg_balance', 'min_balance'
        ]

        # 計算金額異常程度
        df['amount_zscore'] = df.groupby('ACCT_NBR')['TX_AMT'].transform(
            lambda x: (x - x.mean()) / (x.std() if x.std() != 0 else 1)
        )

        # 大額交易比例（超過個人平均交易金額2倍）
        df['is_large_amount'] = df.apply(
            lambda x: x['TX_AMT'] > 2 * amount_stats.loc[x['ACCT_NBR'], 'avg_tx_amount'],
            axis=1
        )
        large_tx_ratio = df.groupby('ACCT_NBR')['is_large_amount'].mean()

        # 合併特徵
        features = pd.concat([amount_stats, large_tx_ratio.rename('large_tx_ratio')], axis=1)

        # 計算餘額利用率
        features['balance_usage_ratio'] = (
            features['total_tx_amount'] / (features['avg_balance'] + 1)
        ).round(4)

        return features

    def generate_time_pattern_features(self, df):
        """
        時間模式特徵
        - 高風險時段交易
        - 交易時間規律性
        - 交易頻率變化
        """
        features = pd.DataFrame()

        # 確保時間欄位格式正確
        df['TX_DATE'] = pd.to_datetime(df['TX_DATE'])
        df['hour'] = df['TX_TIME'].astype(int)

        # 高風險時段交易
        df['is_high_risk_hour'] = df['hour'].isin(self.high_risk_hours)
        risk_hour_stats = df.groupby('ACCT_NBR').agg({
            'is_high_risk_hour': ['mean', 'sum']
        })
        risk_hour_stats.columns = ['high_risk_hour_ratio', 'high_risk_hour_count']

        # 計算每個客戶的交易時間間隔
        time_diffs = df.sort_values('TX_DATE').groupby('ACCT_NBR')['TX_DATE'].diff()
        time_diff_hours = time_diffs.dt.total_seconds() / 3600

        # 交易間隔統計
        interval_stats = (time_diff_hours.groupby(df['ACCT_NBR'])
                         .agg(['mean', 'std', 'min'])
                         .round(2))
        interval_stats.columns = [
            'avg_tx_interval_hours', 
            'std_tx_interval_hours', 
            'min_tx_interval_hours'
        ]

        # 合併特徵
        features = pd.concat([risk_hour_stats, interval_stats], axis=1)

        # 計算日交易頻率的變化
        daily_tx_count = df.groupby(['ACCT_NBR', df['TX_DATE'].dt.date]).size()
        daily_tx_stats = daily_tx_count.groupby('ACCT_NBR').agg(['mean', 'std']).round(2)
        daily_tx_stats.columns = ['avg_daily_tx_count', 'std_daily_tx_count']

        features = pd.concat([features, daily_tx_stats], axis=1)

        return features

    def generate_channel_features(self, df):
        """
        通道行為特徵
        - 高風險通道使用
        - 通道切換模式
        - 通道多樣性
        """
        features = pd.DataFrame()

        # 高風險通道使用
        df['is_high_risk_channel'] = df['CHANNEL_CODE'].isin(self.high_risk_channels)
        risk_channel_stats = df.groupby('ACCT_NBR').agg({
            'is_high_risk_channel': ['mean', 'sum']
        })
        risk_channel_stats.columns = ['high_risk_channel_ratio', 'high_risk_channel_count']

        # 通道多樣性
        channel_diversity = df.groupby('ACCT_NBR')['CHANNEL_CODE'].nunique()

        # 主要使用通道
        main_channel = df.groupby('ACCT_NBR')['CHANNEL_CODE'].agg(
            lambda x: x.value_counts().index[0]
        )

        # 通道切換頻率
        df_sorted = df.sort_values(['ACCT_NBR', 'TX_DATE'])
        channel_changes = (
            df_sorted.groupby('ACCT_NBR')['CHANNEL_CODE'].shift() !=
            df_sorted['CHANNEL_CODE']
        ).groupby(df_sorted['ACCT_NBR']).mean()

        # 合併特徵
        features = pd.concat([
            risk_channel_stats,
            channel_diversity.rename('channel_diversity'),
            main_channel.rename('main_channel'),
            channel_changes.rename('channel_change_ratio')
        ], axis=1)

        return features

    def generate_counterparty_features(self, df):
        """
        交易對手特徵
        - 新對手帳戶比例
        - 交易對手集中度
        - 跨行交易模式
        """
        features = pd.DataFrame()

        # 對手帳戶基本統計
        counterparty_stats = df.groupby('ACCT_NBR').agg({
            'OWN_TRANS_ACCT': ['nunique', 'count']
        })
        counterparty_stats.columns = ['unique_counterparties', 'total_transactions']

        # 計算對手集中度（最常交易對手的交易佔比）
        top_counterparty_share = df.groupby(['ACCT_NBR', 'OWN_TRANS_ACCT']).size()
        top_counterparty_share = (
            top_counterparty_share.groupby('ACCT_NBR').nlargest(1) /
            top_counterparty_share.groupby('ACCT_NBR').sum()
        ).groupby('ACCT_NBR').first()

        # 合併特徵
        features = pd.concat([
            counterparty_stats,
            top_counterparty_share.rename('top_counterparty_ratio')
        ], axis=1)

        # 計算交易對手分散度
        features['counterparty_diversity_ratio'] = (
            features['unique_counterparties'] / features['total_transactions']
        ).round(4)

        return features

    def generate_risk_indicator_features(self, df):
        """
        風險指標特徵
        - 設備重複使用
        - 異常行為標記
        - 風險累積指標
        """
        features = pd.DataFrame()

        # 設備重複使用指標
        device_stats = df.groupby('ACCT_NBR').agg({
            'SAME_NUMBER_IP': ['sum', 'max'],
            'SAME_NUMBER_UUID': ['sum', 'max']
        })
        device_stats.columns = [
            'total_ip_reuse', 'max_ip_reuse',
            'total_uuid_reuse', 'max_uuid_reuse'
        ]

        # 異常行為標記
        risk_markers = df.groupby('ACCT_NBR').agg({
            'CANCEL_NO_CONTACT': 'max',
            'IS_DIGITAL': 'mean'
        })
        risk_markers.columns = ['has_cancel_no_contact', 'digital_service_ratio']

        # 合併特徵
        features = pd.concat([device_stats, risk_markers], axis=1)

        # 計算綜合風險分數
        features['risk_score'] = (
            0.3 * (features['total_ip_reuse'] > 0).astype(int) +
            0.3 * (features['total_uuid_reuse'] > 0).astype(int) +
            0.2 * features['has_cancel_no_contact'] +
            0.2 * (features['digital_service_ratio'] > 0.8).astype(int)
        )

        return features

    def generate_features(self, df):
        """
        特徵生成主函數
        """
        print("開始生成欺詐偵測特徵...")

        # 檢查必要欄位
        required_columns = {
            'TX_AMT', 'TX_DATE', 'TX_TIME', 'CHANNEL_CODE',
            'OWN_TRANS_ACCT', 'PB_BAL', 'SAME_NUMBER_IP', 'SAME_NUMBER_UUID',
            'CANCEL_NO_CONTACT', 'IS_DIGITAL'
        }

        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"缺少必要欄位: {missing_columns}")

        # 檢查ACCT_NBR
        if 'ACCT_NBR' not in df.columns:
            raise ValueError("缺少帳戶號碼欄位 'ACCT_NBR'")

        print(f"原始數據行數: {len(df)}")
        print(f"獨立帳戶數量: {df['ACCT_NBR'].nunique()}")

        # 生成各類特徵
        amount_features = self.generate_amount_features(df)
        time_features = self.generate_time_pattern_features(df)
        channel_features = self.generate_channel_features(df)
        counterparty_features = self.generate_counterparty_features(df)
        risk_features = self.generate_risk_indicator_features(df)

        # 合併所有特徵
        all_features = pd.concat([
            amount_features,
            time_features,
            channel_features,
            counterparty_features,
            risk_features
        ], axis=1)

        # 處理缺失值
        all_features = all_features.fillna(0)

        # 確保索引是ACCT_NBR
        if not all_features.index.equals(df['ACCT_NBR'].unique()):
            print("重新索引特徵...")
            all_features = all_features.reindex(df['ACCT_NBR'].unique())

        print(f"特徵矩陣行數: {len(all_features)}")

        # 將特徵展開到交易級別
        all_features_expanded = pd.merge(
            df[['ACCT_NBR']],
            all_features,
            left_on='ACCT_NBR',
            right_index=True,
            how='left'
        )

        print(f"展開後的特徵矩陣行數: {len(all_features_expanded)}")

        return all_features_expanded
