import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Any

from src.data_loader import SpamDataset
from src.preprocessor import TextPreprocessor
from models.spam_classifier import SpamClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train(
    config: Dict[str, Any],
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader
) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        patience=3, 
        factor=0.5
    )

    best_val_loss = float('inf')
    for epoch in range(config['epochs']):
        model.train()
        total_train_loss = 0.0

        try:
            for batch in train_loader:
                inputs = batch['text'].to(device)
                labels = batch['label'].float().to(device)

                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch['text'].to(device)
                    labels = batch['label'].float().to(device)
                    
                    outputs = model(inputs).squeeze()
                    val_loss = criterion(outputs, labels)
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            
            scheduler.step(avg_val_loss)
            
            logger.info(
                f'Epoch {epoch+1}/{config["epochs"]} '
                f'Train Loss: {avg_train_loss:.4f} '
                f'Val Loss: {avg_val_loss:.4f}'
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(
                    model.state_dict(), 
                    os.path.join(config['checkpoint_dir'], 'best_model.pth')
                )

        except Exception as e:
            logger.error(f'Training error in epoch {epoch}: {e}')
            raise

def main():
    config = {
        'learning_rate': 1e-4,
        'epochs': 50,
        'batch_size': 64,
        'checkpoint_dir': 'models/checkpoints',
        'data_path': 'data/processed/spam_dataset.csv'
    }

    preprocessor = TextPreprocessor()
    dataset = SpamDataset(
        config['data_path'], 
        preprocessor=preprocessor
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size']
    )

    model = SpamClassifier(
        input_size=dataset.feature_dim, 
        hidden_size=128, 
        num_layers=2
    )

    train(config, model, train_loader, val_loader)

if __name__ == '__main__':
    main()