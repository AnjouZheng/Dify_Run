import pytest
import torch
import numpy as np
from models.lstm_model import LSTMStockPredictor
from models.transformer_model import TransformerStockPredictor
from src.data_loader import StockDataLoader
from src.preprocess import DataPreprocessor

class TestStockPredictionModels:
    def setup_method(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.data_loader = StockDataLoader()
        self.preprocessor = DataPreprocessor()

    def test_lstm_model_shape(self):
        X_test = torch.randn(32, 10, 6)
        model = LSTMStockPredictor(input_size=6, hidden_size=64, num_layers=2)
        output = model(X_test)
        
        assert output.shape[0] == 32
        assert output.shape[1] == 1

    def test_transformer_model_shape(self):
        X_test = torch.randn(32, 10, 6)
        model = TransformerStockPredictor(input_size=6, d_model=64, nhead=4)
        output = model(X_test)
        
        assert output.shape[0] == 32
        assert output.shape[1] == 1

    def test_model_gpu_compatibility(self):
        if torch.cuda.is_available():
            device = torch.device('cuda')
            lstm_model = LSTMStockPredictor(input_size=6, hidden_size=64, num_layers=2).to(device)
            transformer_model = TransformerStockPredictor(input_size=6, d_model=64, nhead=4).to(device)
            
            X_test = torch.randn(32, 10, 6).to(device)
            
            lstm_output = lstm_model(X_test)
            transformer_output = transformer_model(X_test)
            
            assert lstm_output.device.type == 'cuda'
            assert transformer_output.device.type == 'cuda'
        else:
            pytest.skip("No CUDA device available")

    def test_model_training_process(self):
        data = self.data_loader.load_processed_data()
        X, y = self.preprocessor.prepare_sequences(data)
        
        lstm_model = LSTMStockPredictor(input_size=X.shape[2], hidden_size=64, num_layers=2)
        transformer_model = TransformerStockPredictor(input_size=X.shape[2], d_model=64, nhead=4)
        
        lstm_loss = self._train_model(lstm_model, X, y)
        transformer_loss = self._train_model(transformer_model, X, y)
        
        assert lstm_loss > 0
        assert transformer_loss > 0

    def _train_model(self, model, X, y, epochs=5):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        total_loss = 0
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(torch.FloatTensor(X))
            loss = criterion(outputs, torch.FloatTensor(y))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / epochs