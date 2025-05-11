import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from src.data_loader import StockDataset
from models.model import StockPricePredictor
from config.train_config import load_train_config

class StockTrainer:
    def __init__(self, config_path: str = 'config/train_config.yaml'):
        self.config = load_train_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join('logs', 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def prepare_data(self) -> tuple:
        try:
            train_dataset = StockDataset(
                data_path=self.config['data_path'],
                mode='train'
            )
            val_dataset = StockDataset(
                data_path=self.config['data_path'],
                mode='val'
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=4
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=4
            )

            return train_loader, val_loader
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            raise

    def train(self):
        train_loader, val_loader = self.prepare_data()
        
        model = StockPricePredictor(
            input_dim=self.config['input_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers']
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        scaler = GradScaler()

        best_val_loss = float('inf')
        for epoch in range(self.config['epochs']):
            model.train()
            total_train_loss = 0.0

            for batch_idx, (X, y) in enumerate(train_loader):
                X, y = X.to(self.device), y.to(self.device)

                optimizer.zero_grad()

                with autocast():
                    predictions = model(X)
                    loss = criterion(predictions, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            val_loss = self.validate(model, val_loader, criterion)

            self.logger.info(
                f'Epoch {epoch+1}/{self.config["epochs"]}: '
                f'Train Loss: {avg_train_loss:.4f}, '
                f'Val Loss: {val_loss:.4f}'
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(model, f'models/trained_models/best_model_epoch_{epoch+1}.pth')

        self.logger.info('Training completed successfully.')

    def validate(self, model, val_loader, criterion):
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)
                predictions = model(X)
                val_loss = criterion(predictions, y)
                total_val_loss += val_loss.item()

        return total_val_loss / len(val_loader)

    def save_model(self, model, path):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(model.state_dict(), path)
            self.logger.info(f'Model saved to {path}')
        except Exception as e:
            self.logger.error(f"Model saving failed: {e}")

def main():
    trainer = StockTrainer()
    trainer.train()

if __name__ == '__main__':
    main()