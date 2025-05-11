import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List
import logging
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

class StockTransformer(nn.Module):
    def __init__(
        self, 
        input_dim: int = 128, 
        num_layers: int = 4, 
        num_heads: int = 8, 
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.embedding = nn.Linear(input_dim, input_dim)
        self.transformer_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim, 
                nhead=num_heads, 
                dropout=dropout_rate
            ), 
            num_layers=num_layers
        )
        self.output_layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.transformer_layers(x)
        return self.output_layer(x[:, -1, :])

class StockPricePredictor:
    def __init__(
        self, 
        config: Dict[str, Any]
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = StockTransformer().to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.get('learning_rate', 1e-4)
        )
        self.criterion = nn.MSELoss()
        logging.basicConfig(level=logging.INFO)
        
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        epochs: int = 50
    ) -> List[float]:
        train_losses, val_losses = [], []
        
        for epoch in range(epochs):
            self.model.train()
            epoch_train_loss = 0.0
            
            for batch in train_loader:
                features, targets = batch
                features, targets = features.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(features)
                loss = self.criterion(predictions, targets)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            self.model.eval()
            epoch_val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    features, targets = batch
                    features, targets = features.to(self.device), targets.to(self.device)
                    
                    predictions = self.model(features)
                    loss = self.criterion(predictions, targets)
                    
                    epoch_val_loss += loss.item()
                
                avg_val_loss = epoch_val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
            
            logging.info(
                f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, "
                f"Val Loss = {avg_val_loss:.4f}"
            )
        
        return train_losses, val_losses
    
    def predict(self, test_loader: DataLoader) -> np.ndarray:
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in test_loader:
                features, _ = batch
                features = features.to(self.device)
                
                batch_predictions = self.model(features)
                predictions.append(batch_predictions.cpu().numpy())
        
        return np.concatenate(predictions)

    def save_model(self, path: str) -> None:
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load_model(self, path: str) -> None:
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])