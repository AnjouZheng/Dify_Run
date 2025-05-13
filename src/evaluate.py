import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix, 
    roc_auc_score
)
from typing import Dict, Any
import logging
import json

class ModelEvaluator:
    def __init__(self, model, test_loader, device='cuda'):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def evaluate(self) -> Dict[str, Any]:
        self.model.eval()
        self.model.to(self.device)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.test_loader:
                inputs = batch['text'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(inputs)
                preds = torch.sigmoid(outputs).round().cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        metrics = self._compute_metrics(all_labels, all_preds)
        self._log_metrics(metrics)
        self._save_metrics(metrics)

        return metrics

    def _compute_metrics(self, true_labels, pred_labels):
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='binary'
        )

        metrics = {
            'accuracy': accuracy_score(true_labels, pred_labels),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': confusion_matrix(true_labels, pred_labels).tolist(),
            'roc_auc': roc_auc_score(true_labels, pred_labels)
        }

        return metrics

    def _log_metrics(self, metrics):
        self.logger.info("Model Evaluation Metrics:")
        for key, value in metrics.items():
            self.logger.info(f"{key}: {value}")

    def _save_metrics(self, metrics, filepath='./metrics.json'):
        try:
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=4)
        except IOError as e:
            self.logger.error(f"Error saving metrics: {e}")

def main():
    from models.model import SpamClassifier
    from torch.utils.data import DataLoader
    from src.preprocess import load_test_dataset

    model = SpamClassifier()
    test_loader = DataLoader(load_test_dataset(), batch_size=32)
    
    evaluator = ModelEvaluator(model, test_loader)
    evaluator.evaluate()

if __name__ == "__main__":
    main()