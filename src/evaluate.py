import numpy as np
import torch
from torch import nn
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
import logging
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.to(device)

    def evaluate(self, test_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        try:
            self.model.eval()
            predictions = []
            actual_values = []

            with torch.no_grad():
                for batch in test_loader:
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                    pred = self.model(x)
                    
                    predictions.extend(pred.cpu().numpy())
                    actual_values.extend(y.cpu().numpy())

            predictions = np.array(predictions)
            actual_values = np.array(actual_values)

            metrics = {
                'mse': mean_squared_error(actual_values, predictions),
                'mae': mean_absolute_error(actual_values, predictions),
                'rmse': np.sqrt(mean_squared_error(actual_values, predictions)),
                'mape': mean_absolute_percentage_error(actual_values, predictions),
                'r2_score': r2_score(actual_values, predictions)
            }

            self._log_metrics(metrics)
            return metrics

        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            raise

    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name.upper()}: {value:.4f}")

    def predict(self, x_test: torch.Tensor) -> np.ndarray:
        try:
            self.model.eval()
            with torch.no_grad():
                x_test = x_test.to(self.device)
                predictions = self.model(x_test)
                return predictions.cpu().numpy()
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

def load_model(model_path: str, model_class: nn.Module) -> nn.Module:
    try:
        model = model_class()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        raise

def save_model_predictions(predictions: np.ndarray, output_path: str) -> None:
    try:
        np.save(output_path, predictions)
        logger.info(f"Predictions saved to {output_path}")
    except Exception as e:
        logger.error(f"Prediction saving error: {e}")
        raise

if __name__ == "__main__":
    from models.lstm_model import StockPriceLSTM
    from data_loader import load_test_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StockPriceLSTM().to(device)
    test_loader = load_test_data()

    evaluator = ModelEvaluator(model, device)
    metrics = evaluator.evaluate(test_loader)
    
    model_path = "models/trained_models/best_stock_model.pth"
    torch.save(model.state_dict(), model_path)
    
    predictions_path = "data/predictions/stock_predictions.npy"
    predictions = evaluator.predict(test_loader.dataset.x_test)
    save_model_predictions(predictions, predictions_path)