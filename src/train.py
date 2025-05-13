import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from typing import Dict, Any
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

class SpamClassifier:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        
    def _load_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config['model_name'], 
            num_labels=2
        ).to(self.device)
        return model
    
    def _prepare_dataset(self, data_path: str):
        df = pd.read_csv(data_path)
        X_train, X_val, y_train, y_val = train_test_split(
            df['text'], 
            df['label'], 
            test_size=0.2, 
            random_state=42
        )
        
        train_encodings = self.tokenizer(
            X_train.tolist(), 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        val_encodings = self.tokenizer(
            X_val.tolist(), 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(train_encodings['input_ids']),
            torch.tensor(train_encodings['attention_mask']),
            torch.tensor(y_train.values)
        )
        
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(val_encodings['input_ids']),
            torch.tensor(val_encodings['attention_mask']),
            torch.tensor(y_val.values)
        )
        
        return train_dataset, val_dataset
    
    def train(self, data_path: str):
        train_dataset, val_dataset = self._prepare_dataset(data_path)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size']
        )
        
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config['learning_rate']
        )
        
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.config['epochs']):
            self.model.train()
            total_train_loss = 0
            
            for batch in train_loader:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels
                )
                
                loss = outputs.loss
                total_train_loss += loss.item()
                
                loss.backward()
                optimizer.step()
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            logging.info(f'Epoch {epoch+1}/{self.config["epochs"]}')
            logging.info(f'Average training loss: {avg_train_loss:.4f}')
            
            self._validate(val_loader, criterion)
        
        self._save_model()
    
    def _validate(self, val_loader, criterion):
        self.model.eval()
        total_val_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                
                outputs = self.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels
                )
                
                loss = outputs.loss
                total_val_loss += loss.item()
                
                predictions = torch.argmax(outputs.logits, dim=1)
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        
        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = total_correct / total_samples
        
        logging.info(f'Validation loss: {avg_val_loss:.4f}')
        logging.info(f'Validation accuracy: {accuracy:.4f}')
    
    def _save_model(self):
        os.makedirs(self.config['model_save_path'], exist_ok=True)
        self.model.save_pretrained(self.config['model_save_path'])
        self.tokenizer.save_pretrained(self.config['model_save_path'])
        logging.info(f"Model saved to {self.config['model_save_path']}")

def main():
    config = {
        'model_name': 'distilbert-base-uncased',
        'batch_size': 16,
        'learning_rate': 2e-5,
        'epochs': 5,
        'model_save_path': './models/trained_models/spam_classifier'
    }
    
    classifier = SpamClassifier(config)
    classifier.train('./data/processed/spam_dataset.csv')

if __name__ == '__main__':
    main()