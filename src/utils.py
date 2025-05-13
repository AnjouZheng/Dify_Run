import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Union
import re
import hashlib
import json

class SpamClassificationUtils:
    @staticmethod
    def clean_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def generate_email_hash(email: str) -> str:
        return hashlib.md5(email.encode()).hexdigest()

    @staticmethod
    def extract_email_features(email_text: str) -> Dict[str, Union[int, float]]:
        features = {
            'length': len(email_text),
            'word_count': len(email_text.split()),
            'unique_word_ratio': len(set(email_text.split())) / len(email_text.split()),
            'uppercase_ratio': sum(1 for c in email_text if c.isupper()) / len(email_text),
            'contains_url': 1 if 'http' in email_text.lower() else 0,
            'contains_dollar': 1 if '$' in email_text else 0
        }
        return features

    @staticmethod
    def configure_logging(log_path: str = 'logs/spam_classifier.log') -> logging.Logger:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger('SpamClassifier')

    @staticmethod
    def load_config(config_path: str = 'config/params.yaml') -> Dict:
        try:
            with open(config_path, 'r') as file:
                return json.load(file)
        except Exception as e:
            logging.error(f"Config load error: {e}")
            return {}

    @staticmethod
    def save_prediction_log(
        email_id: str, 
        prediction: float, 
        threshold: float = 0.5, 
        log_path: str = 'logs/predictions.csv'
    ) -> None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        result_df = pd.DataFrame({
            'email_id': [email_id],
            'prediction': [prediction],
            'is_spam': [1 if prediction >= threshold else 0]
        })
        
        result_df.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)

    @staticmethod
    def get_gpu_memory() -> int:
        try:
            import torch
            return torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        except ImportError:
            return 0