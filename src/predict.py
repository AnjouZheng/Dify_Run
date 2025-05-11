import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from src.data_loader import load_stock_data
from src.preprocess import preprocess_data
from models.lstm_model import LSTMStockPredictor
from models.transformer_model import TransformerStockPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self, model_type: str = 'lstm', config: Optional[Dict[str, Any]] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.config = config or {}
        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        try:
            if self.model_type == 'lstm':
                model = LSTMStockPredictor(**self.config).to(self.device)
            elif self.model_type == 'transformer':
                model = TransformerStockPredictor(**self.config).to(self.device)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            model_path = os.path.join('models', 'trained_models', f'{self.model_type}_best_model.pth')
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Loaded pre-trained {self.model_type} model")
            return model
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            raise

    def predict(self, input_data: pd.DataFrame, steps_ahead: int = 5) -> np.ndarray:
        try:
            preprocessed_data = preprocess_data(input_data)
            input_tensor = torch.tensor(preprocessed_data.values, dtype=torch.float32).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(input_tensor)
            
            return predictions.cpu().numpy()
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

    def batch_predict(self, data_dir: str, output_path: str = 'predictions.csv'):
        try:
            stock_data = load_stock_data(data_dir)
            predictions = []
            
            for stock_name, stock_df in stock_data.items():
                stock_predictions = self.predict(stock_df)
                stock_df['Predicted_Price'] = stock_predictions
                predictions.append(stock_df)
            
            final_predictions = pd.concat(predictions)
            final_predictions.to_csv(output_path, index=False)
            logger.info(f"Batch predictions saved to {output_path}")
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise

def main():
    predictor = StockPredictor(model_type='lstm')
    test_data = load_stock_data('data/raw/')
    predictor.batch_predict('data/raw/')

if __name__ == "__main__":
    main()