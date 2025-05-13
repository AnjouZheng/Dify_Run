import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpamClassificationModel(nn.Module):
    def __init__(self, model_name: str = 'distilbert-base-uncased', num_classes: int = 2, dropout_rate: float = 0.3):
        super(SpamClassificationModel, self).__init__()
        
        try:
            self.bert_model = BertModel.from_pretrained(model_name)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Model initialization error: {e}")
            raise

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.bert_model.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert_model(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        pooled_output = bert_output.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)
        return logits

    def configure_optimizers(self, learning_rate: float = 2e-5) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=learning_rate)

    def predict(self, text: str) -> Dict[str, Any]:
        self.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                text, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            outputs = self(
                input_ids=inputs['input_ids'], 
                attention_mask=inputs['attention_mask']
            )
            
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            
            return {
                'class': int(predicted_class[0]),
                'probability': float(probabilities[0][predicted_class[0]])
            }

def create_model(config: Dict[str, Any]) -> SpamClassificationModel:
    model = SpamClassificationModel(
        model_name=config.get('model_name', 'distilbert-base-uncased'),
        num_classes=config.get('num_classes', 2),
        dropout_rate=config.get('dropout_rate', 0.3)
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("Model moved to CUDA")
    
    return model