import numpy as np
import pandas as pd
import talib
from typing import List, Dict, Any
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats

class FeatureEngineer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def load_data(self, filepath: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(filepath)
            self.logger.info(f"Loaded data from {filepath}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['SMA_10'] = talib.SMA(df['Close'], timeperiod=10)
        df['SMA_30'] = talib.SMA(df['Close'], timeperiod=30)
        df['EMA_10'] = talib.EMA(df['Close'], timeperiod=10)
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        df['MACD'], df['MACD_signal'], _ = talib.MACD(df['Close'])
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        return df

    def create_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3]) -> pd.DataFrame:
        for lag in lags:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        return df

    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Daily_Return'] = df['Close'].pct_change()
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        return df

    def normalize_features(self, df: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
        features_to_normalize = ['Close', 'SMA_10', 'SMA_30', 'EMA_10', 'RSI', 'MACD', 'ATR']
        
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError("Invalid normalization method")

        df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])
        return df

    def add_market_sentiment_features(self, df: pd.DataFrame, market_indices: pd.DataFrame) -> pd.DataFrame:
        df = pd.merge(df, market_indices, on='Date', how='left')
        df['Market_Trend'] = np.where(df['Market_Return'] > 0, 1, 0)
        return df

    def detect_outliers(self, df: pd.DataFrame, column: str = 'Close', method: str = 'zscore') -> pd.DataFrame:
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(df[column]))
            df['Outlier'] = z_scores > 3
        return df

    def engineer_features(self, filepath: str) -> pd.DataFrame:
        try:
            df = self.load_data(filepath)
            df = self.calculate_technical_indicators(df)
            df = self.create_lag_features(df)
            df = self.calculate_returns(df)
            df = self.normalize_features(df)
            df = self.detect_outliers(df)
            return df
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            raise

    def save_processed_data(self, df: pd.DataFrame, output_path: str) -> None:
        try:
            df.to_csv(output_path, index=False)
            self.logger.info(f"Processed data saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving processed data: {e}")
            raise

if __name__ == "__main__":
    config = {
        "input_path": "./data/raw/stocks_raw.csv",
        "output_path": "./data/processed/stocks_processed.csv"
    }
    feature_engineer = FeatureEngineer(config)
    processed_data = feature_engineer.engineer_features(config["input_path"])
    feature_engineer.save_processed_data(processed_data, config["output_path"])