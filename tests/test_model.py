import unittest
import torch
import numpy as np
from models.model import StockPredictionModel
from src.data_loader import StockDataLoader
from src.utils import seed_everything

class TestStockPredictionModel(unittest.TestCase):
    def setUp(self):
        seed_everything(42)
        self.model = StockPredictionModel()
        self.data_loader = StockDataLoader()

    def test_model_forward_pass(self):
        batch_size = 32
        sequence_length = 60
        feature_dim = 16
        
        test_input = torch.randn(batch_size, sequence_length, feature_dim)
        output = self.model(test_input)
        
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (batch_size, 1))

    def test_model_parameter_count(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertLess(total_params, 1_000_000, "Model is too large")

    def test_model_device_compatibility(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        batch_size = 32
        sequence_length = 60
        feature_dim = 16
        
        test_input = torch.randn(batch_size, sequence_length, feature_dim).to(device)
        output = self.model(test_input)
        
        self.assertEqual(output.device, device)

    def test_model_gradient_flow(self):
        batch_size = 32
        sequence_length = 60
        feature_dim = 16
        
        test_input = torch.randn(batch_size, sequence_length, feature_dim, requires_grad=True)
        test_target = torch.randn(batch_size, 1)
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        
        output = self.model(test_input)
        loss = criterion(output, test_target)
        
        optimizer.zero_grad()
        loss.backward()
        
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for parameter {name}")

    def test_model_overfitting_detection(self):
        batch_size = 32
        sequence_length = 60
        feature_dim = 16
        
        train_input = torch.randn(batch_size, sequence_length, feature_dim)
        train_target = torch.randn(batch_size, 1)
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        
        initial_loss = float('inf')
        for _ in range(10):
            output = self.model(train_input)
            loss = criterion(output, train_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if loss.item() >= initial_loss:
                break
            
            initial_loss = loss.item()
        
        self.assertLess(initial_loss, 1.0, "Model failed to learn")

if __name__ == '__main__':
    unittest.main()