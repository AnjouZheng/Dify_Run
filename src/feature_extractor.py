import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

class SpamFeatureExtractor:
    def __init__(self, max_features: int = 5000):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.logger = logging.getLogger(__name__)

    def clean_text(self, text: str) -> str:
        """文本预处理：清洁文本数据"""
        try:
            text = text.lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        except Exception as e:
            self.logger.error(f"Text cleaning error: {e}")
            return text

    def extract_email_features(self, emails: List[str]) -> np.ndarray:
        """提取邮件文本特征"""
        try:
            cleaned_emails = [self.clean_text(email) for email in emails]
            features = self.tfidf_vectorizer.fit_transform(cleaned_emails)
            return features.toarray()
        except Exception as e:
            self.logger.error(f"Feature extraction error: {e}")
            return np.array([])

    def extract_metadata_features(self, email_metadata: List[Dict[str, Any]]) -> np.ndarray:
        """提取邮件元数据特征"""
        try:
            metadata_features = []
            for metadata in email_metadata:
                features = [
                    len(metadata.get('sender', '')),
                    len(metadata.get('subject', '')),
                    metadata.get('has_attachment', 0),
                    metadata.get('is_forwarded', 0)
                ]
                metadata_features.append(features)
            return np.array(metadata_features)
        except Exception as e:
            self.logger.error(f"Metadata feature extraction error: {e}")
            return np.array([])

    def combine_features(self, text_features: np.ndarray, metadata_features: np.ndarray) -> np.ndarray:
        """合并文本和元数据特征"""
        try:
            if text_features.size == 0 or metadata_features.size == 0:
                raise ValueError("Features cannot be empty")
            
            combined_features = np.hstack([text_features, metadata_features])
            return combined_features
        except Exception as e:
            self.logger.error(f"Feature combination error: {e}")
            return np.array([])

def main():
    feature_extractor = SpamFeatureExtractor()
    sample_emails = [
        "Buy now! Special discount offer",
        "Meeting agenda for next week's project"
    ]
    sample_metadata = [
        {'sender': 'marketing@spam.com', 'subject': 'Discount', 'has_attachment': 1, 'is_forwarded': 0},
        {'sender': 'colleague@company.com', 'subject': 'Project', 'has_attachment': 0, 'is_forwarded': 1}
    ]

    text_features = feature_extractor.extract_email_features(sample_emails)
    metadata_features = feature_extractor.extract_metadata_features(sample_metadata)
    combined_features = feature_extractor.combine_features(text_features, metadata_features)

    logging.info(f"Combined feature shape: {combined_features.shape}")

if __name__ == "__main__":
    main()