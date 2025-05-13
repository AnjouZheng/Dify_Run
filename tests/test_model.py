import pytest
import torch
from torch.testing import assert_close
from models.model import SpamClassifier
from src.preprocess import TextPreprocessor
from typing import Dict, Any

class TestSpamModel:
    @pytest.fixture
    def model(self) -> SpamClassifier:
        return SpamClassifier(
            embedding_dim=300,
            hidden_dim=128,
            num_classes=2,
            dropout_rate=0.3
        )

    @pytest.fixture
    def preprocessor(self) -> TextPreprocessor:
        return TextPreprocessor()

    def test_model_output_shape(self, model: SpamClassifier) -> None:
        batch_size, seq_length = 32, 100
        input_tensor = torch.randn(batch_size, seq_length)
        output = model(input_tensor)
        assert output.shape == (batch_size, 2), "模型输出形状不正确"

    def test_model_parameters(self, model: SpamClassifier) -> None:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0, "模型参数数量为零"
        assert trainable_params > 0, "可训练参数数量为零"

    def test_model_prediction_range(self, model: SpamClassifier) -> None:
        batch_size, seq_length = 16, 50
        input_tensor = torch.randn(batch_size, seq_length)
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        
        assert torch.all((probabilities >= 0) & (probabilities <= 1)), "预测概率超出[0,1]范围"
        assert torch.allclose(probabilities.sum(dim=1), torch.tensor(1.0)), "概率分布不是有效的概率分布"

    def test_model_gradient_flow(self, model: SpamClassifier) -> None:
        batch_size, seq_length = 32, 100
        input_tensor = torch.randn(batch_size, seq_length, requires_grad=True)
        output = model(input_tensor)
        loss = output.mean()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"参数 {name} 没有梯度"
                assert not torch.isnan(param.grad).any(), f"参数 {name} 梯度存在NaN"

    def test_model_serialization(self, model: SpamClassifier, tmp_path) -> None:
        model_path = tmp_path / "spam_model.pth"
        torch.save(model.state_dict(), model_path)
        
        loaded_model = SpamClassifier(
            embedding_dim=300,
            hidden_dim=128,
            num_classes=2,
            dropout_rate=0.3
        )
        loaded_model.load_state_dict(torch.load(model_path))
        
        assert len(list(model.parameters())) == len(list(loaded_model.parameters())), "模型加载后参数数量不一致"

    @pytest.mark.parametrize("input_length", [10, 50, 100, 200])
    def test_variable_input_length(self, model: SpamClassifier, input_length: int) -> None:
        input_tensor = torch.randn(1, input_length)
        output = model(input_tensor)
        assert output.shape == (1, 2), f"输入长度{input_length}时输出形状不正确"