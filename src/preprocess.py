import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

class StockDataPreprocessor:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scaler = MinMaxScaler()
        self.imputer = SimpleImputer(strategy='median')

    def load_data(self, filepath: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(filepath)
            self.logger.info(f"Loaded data from {filepath}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = self.imputer.fit_transform(df[numeric_columns])
        return df

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df['returns'] = df['Close'].pct_change()
        df['rolling_mean_5'] = df['Close'].rolling(window=5).mean()
        df['rolling_std_5'] = df['Close'].rolling(window=5).std()
        df['volume_change'] = df['Volume'].pct_change()
        return df

    def normalize_features(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        if columns is None:
            columns = ['returns', 'rolling_mean_5', 'rolling_std_5', 'volume_change']
        
        df[columns] = self.scaler.fit_transform(df[columns])
        return df

    def preprocess_pipeline(self, filepath: str) -> pd.DataFrame:
        try:
            df = self.load_data(filepath)
            df = self.handle_missing_values(df)
            df = self.feature_engineering(df)
            df = self.normalize_features(df)
            
            self.logger.info("Preprocessing completed successfully")
            return df
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            raise

def main():
    config = {
        'data_path': 'data/raw/stocks_raw.csv',
        'output_path': 'data/processed/preprocessed_stocks.csv'
    }
    
    preprocessor = StockDataPreprocessor(config)
    processed_data = preprocessor.preprocess_pipeline(config['data_path'])
    processed_data.to_csv(config['output_path'], index=False)

if __name__ == '__main__':
    main()