import os
import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Optional, Dict, Any
import logging
import hashlib

class EmailPreprocessor:
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=5000)

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

    def load_data(self, file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            required_columns = ['text', 'label']
            if not all(col in df.columns for col in required_columns):
                raise ValueError("Missing required columns")
            return df
        except Exception as e:
            self.logger.error(f"Data loading error: {e}")
            raise

    def preprocess_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        X = self.vectorizer.fit_transform(df['cleaned_text']).toarray()
        y = df['label'].values

        return {
            'features': X,
            'labels': y,
            'feature_names': self.vectorizer.get_feature_names_out()
        }

    def split_data(self, X, y, test_size: float = 0.2, random_state: int = 42):
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def generate_data_hash(self, X: np.ndarray, y: np.ndarray) -> str:
        combined_data = np.concatenate([X.flatten(), y])
        return hashlib.md5(combined_data.tobytes()).hexdigest()

def main():
    preprocessor = EmailPreprocessor()
    raw_data_path = os.path.join('data', 'raw', 'emails.csv')
    processed_data_dir = os.path.join('data', 'processed')
    splits_dir = os.path.join('data', 'splits')

    os.makedirs(processed_data_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    try:
        df = preprocessor.load_data(raw_data_path)
        processed_data = preprocessor.preprocess_dataset(df)

        X_train, X_test, y_train, y_test = preprocessor.split_data(
            processed_data['features'], 
            processed_data['labels']
        )

        np.save(os.path.join(processed_data_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(processed_data_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(splits_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(splits_dir, 'y_test.npy'), y_test)

        data_hash = preprocessor.generate_data_hash(X_train, y_train)
        with open(os.path.join(processed_data_dir, 'data_hash.txt'), 'w') as f:
            f.write(data_hash)

    except Exception as e:
        preprocessor.logger.error(f"Preprocessing failed: {e}")

if __name__ == "__main__":
    main()