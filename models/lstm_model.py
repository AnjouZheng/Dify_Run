import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any
import logging
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

class LSTMStockPredictor(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 64, 
        num_layers: int = 2, 
        dropout: float = 0.2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

def train_lstm_model(
    model: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader,
    config: Dict[str, Any]
) -> Dict[str, float]:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = Adam(
        model.parameters(), 
        lr=config.get('learning_rate', 1e-3),
        weight_decay=config.get('weight_decay', 1e-5)
    )

    best_val_loss = float('inf')
    for epoch in range(config.get('epochs', 50)):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                predictions = model(batch_x)
                val_loss += criterion(predictions, batch_y).item()

        val_loss /= len(val_loader)

        logger.info(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/trained_models/best_lstm_model.pth')

    return {
        'train_loss': train_loss,
        'val_loss': val_loss
    }

def predict_stock_prices(
    model: nn.Module, 
    test_data: torch.Tensor
) -> np.ndarray:
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with torch.no_grad():
        test_data = test_data.to(device)
        predictions = model(test_data)
        
    return predictions.cpu().numpy()