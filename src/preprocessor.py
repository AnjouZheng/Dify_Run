import re
import numpy as np
import pandas as pd
from typing import List, Dict, Union
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class TextPreprocessor:
    def __init__(self, language: str = 'english'):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words(language))
        self.vectorizer = TfidfVectorizer(max_features=5000)

    def clean_text(self, text: str) -> str:
        try:
            text = text.lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            tokens = word_tokenize(text)
            tokens = [token for token in tokens if token not in self.stop_words]
            return ' '.join(tokens)
        except Exception as e:
            logger.error(f"Text cleaning error: {e}")
            return text

    def preprocess_dataset(self, data: List[str]) -> Dict[str, Union[np.ndarray, List[str]]]:
        try:
            cleaned_texts = [self.clean_text(text) for text in data]
            tfidf_matrix = self.vectorizer.fit_transform(cleaned_texts)
            
            return {
                'features': tfidf_matrix.toarray(),
                'cleaned_texts': cleaned_texts,
                'vocabulary': self.vectorizer.get_feature_names_out()
            }
        except Exception as e:
            logger.error(f"Dataset preprocessing error: {e}")
            return {}

    def extract_email_features(self, email_text: str) -> Dict[str, float]:
        try:
            features = {
                'length': len(email_text),
                'word_count': len(email_text.split()),
                'uppercase_ratio': sum(1 for c in email_text if c.isupper()) / len(email_text),
                'special_char_ratio': sum(1 for c in email_text if not c.isalnum()) / len(email_text)
            }
            return features
        except Exception as e:
            logger.error(f"Email feature extraction error: {e}")
            return {}

if __name__ == '__main__':
    preprocessor = TextPreprocessor()
    sample_emails = [
        "Free discount offer! Click here now!",
        "Meeting scheduled for tomorrow at 2 PM",
        "Urgent: Your account needs verification"
    ]
    processed_data = preprocessor.preprocess_dataset(sample_emails)
    logger.info(f"Processed {len(processed_data['cleaned_texts'])} emails")