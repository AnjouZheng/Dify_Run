import numpy as np
import pandas as pd
import ta
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Any
import logging

class FeatureEngineer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scaler = MinMaxScaler()

    def load_data(self, file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Successfully loaded data from {file_path}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def extract_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()
            df['MACD'] = ta.trend.MACD(df['close']).macd()
            df['BBH'] = ta.volatility.BollingerBands(df['close']).bollinger_hband()
            df['BBL'] = ta.volatility.BollingerBands(df['close']).bollinger_lband()
            df['EMA_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
            
            self.logger.info("Technical indicators extracted successfully")
            return df
        except Exception as e:
            self.logger.error(f"Error extracting technical indicators: {e}")
            raise

    def create_lagged_features(self, df: pd.DataFrame, lag_periods: List[int] = [1, 3, 5]) -> pd.DataFrame:
        try:
            for period in lag_periods:
                df[f'close_lag_{period}'] = df['close'].shift(period)
                df[f'volume_lag_{period}'] = df['volume'].shift(period)
            
            df.dropna(inplace=True)
            self.logger.info(f"Created lagged features for periods: {lag_periods}")
            return df
        except Exception as e:
            self.logger.error(f"Error creating lagged features: {e}")
            raise

    def normalize_features(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        try:
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns
            
            df[columns] = self.scaler.fit_transform(df[columns])
            self.logger.info(f"Normalized features: {columns}")
            return df
        except Exception as e:
            self.logger.error(f"Error normalizing features: {e}")
            raise

    def generate_target_variable(self, df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
        try:
            df['target'] = df['close'].shift(-horizon)
            df['target_direction'] = np.where(df['target'] > df['close'], 1, 0)
            df.dropna(inplace=True)
            
            self.logger.info(f"Generated target variable with {horizon}-day horizon")
            return df
        except Exception as e:
            self.logger.error(f"Error generating target variable: {e}")
            raise

    def process_pipeline(self, file_path: str) -> pd.DataFrame:
        try:
            df = self.load_data(file_path)
            df = self.extract_technical_indicators(df)
            df = self.create_lagged_features(df)
            df = self.normalize_features(df)
            df = self.generate_target_variable(df)
            
            return df
        except Exception as e:
            self.logger.error(f"Error in feature engineering pipeline: {e}")
            raise

    def save_processed_data(self, df: pd.DataFrame, output_path: str) -> None:
        try:
            df.to_csv(output_path, index=False)
            self.logger.info(f"Processed data saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving processed data: {e}")
            raise