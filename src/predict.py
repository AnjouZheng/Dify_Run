import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from src.data_loader import load_stock_data
from models.model import StockPredictionModel
from config.model_config import load_model_config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

class StockPredictor:
    def __init__(
        self, 
        model_path: str, 
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        self.device = device
        self.model = self._load_model(model_path)
        self.model.eval()

    def _load_model(self, model_path: str) -> nn.Module:
        try:
            config = load_model_config()
            model = StockPredictionModel(config)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            return model
        except Exception as e:
            logging.error(f"Model loading failed: {e}")
            raise

    def predict(
        self, 
        input_data: np.ndarray, 
        lookback_window: int = 30
    ) -> Dict[str, float]:
        try:
            if input_data.shape[0] < lookback_window:
                raise ValueError(f"Input data must have at least {lookback_window} time steps")

            input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                prediction = self.model(input_tensor)
                predicted_price = prediction.cpu().numpy()[0][0]

            return {
                'predicted_price': float(predicted_price),
                'confidence': 1.0  # Placeholder for model confidence
            }
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            raise

    def batch_predict(
        self, 
        stock_data_list: List[np.ndarray], 
        lookback_window: int = 30
    ) -> List[Dict[str, float]]:
        predictions = []
        for stock_data in stock_data_list:
            try:
                prediction = self.predict(stock_data, lookback_window)
                predictions.append(prediction)
            except Exception as e:
                logging.warning(f"Batch prediction error: {e}")
                predictions.append(None)
        return predictions

def main(
    stock_symbol: str, 
    model_path: Optional[str] = None
):
    if model_path is None:
        model_path = os.path.join('models', 'trained_models', 'best_model.pth')

    predictor = StockPredictor(model_path)
    
    try:
        stock_data = load_stock_data(stock_symbol)
        prediction = predictor.predict(stock_data)
        logging.info(f"Stock {stock_symbol} Prediction: {prediction}")
    except Exception as e:
        logging.error(f"Prediction process failed: {e}")

if __name__ == "__main__":
    main('AAPL')