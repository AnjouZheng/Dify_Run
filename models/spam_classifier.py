import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import logging
import numpy as np

class SpamClassifier(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(SpamClassifier, self).__init__()
        self.embedding_dim = config.get('embedding_dim', 300)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.vocab_size = config.get('vocab_size', 10000)
        self.dropout_rate = config.get('dropout_rate', 0.5)

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim, 
            hidden_size=self.hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            dropout=self.dropout_rate
        )
        self.fc1 = nn.Linear(self.hidden_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        hidden = hidden[-1]
        
        x = F.relu(self.fc1(hidden))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

    def predict(self, x, threshold: float = 0.5):
        with torch.no_grad():
            prob = self.forward(x)
            return (prob > threshold).float()

    def compute_metrics(self, y_pred, y_true):
        y_pred_binary = (y_pred > 0.5).float()
        accuracy = torch.mean((y_pred_binary == y_true).float())
        
        tp = torch.sum((y_pred_binary == 1) & (y_true == 1))
        fp = torch.sum((y_pred_binary == 1) & (y_true == 0))
        fn = torch.sum((y_pred_binary == 0) & (y_true == 1))
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1_score = 2 * precision * recall / (precision + recall + 1e-7)
        
        return {
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1_score': f1_score.item()
        }

def load_model(checkpoint_path: str, config: Dict[str, Any]) -> SpamClassifier:
    try:
        model = SpamClassifier(config)
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        logging.info(f"Model loaded from {checkpoint_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def save_model(model: SpamClassifier, save_path: str):
    try:
        torch.save(model.state_dict(), save_path)
        logging.info(f"Model saved to {save_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise