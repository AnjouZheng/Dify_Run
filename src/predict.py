import os
import logging
from typing import Dict, Any, List

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class SpamPredictor:
    def __init__(self, model_path: str = 'models/spam_classifier.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def _load_model(self, model_path: str) -> torch.nn.Module:
        try:
            model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

    def preprocess_text(self, text: str) -> torch.Tensor:
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors='pt', 
                max_length=512, 
                truncation=True, 
                padding='max_length'
            ).to(self.device)
            return inputs
        except Exception as e:
            logger.error(f"Text preprocessing error: {e}")
            raise

    def predict(self, text: str) -> Dict[str, Any]:
        try:
            inputs = self.preprocess_text(text)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            return {
                'is_spam': bool(predicted_class),
                'confidence': confidence,
                'details': {
                    'ham_probability': probabilities[0][0].item(),
                    'spam_probability': probabilities[0][1].item()
                }
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        try:
            return [self.predict(text) for text in texts]
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise

def predict_from_file(input_file: str, output_file: str) -> None:
    predictor = SpamPredictor()
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            texts = f.readlines()
        
        predictions = predictor.batch_predict([text.strip() for text in texts])
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for text, pred in zip(texts, predictions):
                f.write(f"Text: {text.strip()}\n")
                f.write(f"Is Spam: {pred['is_spam']}\n")
                f.write(f"Confidence: {pred['confidence']:.4f}\n\n")
        
        logger.info(f"Predictions written to {output_file}")
    
    except Exception as e:
        logger.error(f"File processing error: {e}")
        raise

if __name__ == "__main__":
    predict_from_file('data/raw/emails.txt', 'data/processed/spam_predictions.txt')