import os
import logging
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Union, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class SpamPredictor:
    def __init__(
        self, 
        model_path: str = 'models/trained_models/spam_classifier',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        try:
            self.device = torch.device(device)
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Model initialization error: {e}")
            raise

    def preprocess(self, email_text: str) -> Dict[str, torch.Tensor]:
        try:
            inputs = self.tokenizer(
                email_text, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors='pt'
            ).to(self.device)
            return inputs
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            raise

    def predict(self, email_text: str) -> Dict[str, Union[float, str]]:
        try:
            with torch.no_grad():
                inputs = self.preprocess(email_text)
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                spam_probability = probabilities[0][1].item()
                
                result = {
                    'spam_probability': spam_probability,
                    'classification': 'spam' if spam_probability > 0.5 else 'not_spam'
                }
                return result
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

    def batch_predict(self, email_texts: list) -> list:
        try:
            results = [self.predict(text) for text in email_texts]
            return results
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise

def main():
    predictor = SpamPredictor()
    sample_email = "Get rich quick! Free money guaranteed!"
    result = predictor.predict(sample_email)
    logger.info(f"Prediction Result: {result}")

if __name__ == "__main__":
    main()