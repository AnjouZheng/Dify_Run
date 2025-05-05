import pytest
import torch
from src.models.spam_classifier import SpamClassifier
from src.data_loader import EmailDataset
from torch.utils.data import DataLoader

class TestSpamClassifier:
    def setup_method(self):
        self.model = SpamClassifier(input_dim=1000, hidden_dim=256, output_dim=2)
        self.dataset = EmailDataset(data_path='data/processed/test_emails.csv')
        self.dataloader = DataLoader(self.dataset, batch_size=32, shuffle=False)

    def test_model_forward_pass(self):
        for batch in self.dataloader:
            inputs, labels = batch
            outputs = self.model(inputs)
            
            assert outputs.shape[0] == inputs.shape[0], "Batch size mismatch"
            assert outputs.shape[1] == 2, "Output dimension should be 2 (spam/not spam)"
            assert torch.isfinite(outputs).all(), "Model outputs contain NaN or Inf"

    def test_model_prediction(self):
        self.model.eval()
        with torch.no_grad():
            for batch in self.dataloader:
                inputs, labels = batch
                predictions = self.model(inputs)
                predicted_classes = torch.argmax(predictions, dim=1)
                
                assert (predicted_classes == 0) | (predicted_classes == 1), "Invalid prediction classes"
                assert predicted_classes.shape == labels.shape, "Prediction shape mismatch"

    def test_model_parameters(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        assert total_params > 0, "Model has no parameters"
        assert trainable_params > 0, "No trainable parameters"

    def test_model_gradient_flow(self):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        for batch in self.dataloader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Invalid gradient for {name}"

            optimizer.step()

    def test_model_overfitting_check(self):
        small_dataset = torch.utils.data.Subset(self.dataset, indices=range(10))
        small_loader = DataLoader(small_dataset, batch_size=10, shuffle=False)
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        
        initial_loss = None
        for epoch in range(10):
            for batch in small_loader:
                inputs, labels = batch
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                if initial_loss is None:
                    initial_loss = loss.item()
                else:
                    assert loss.item() <= initial_loss * 1.1, "Model not converging"