import os
import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.model_selection import train_test_split

class TestSpamData:
    @pytest.fixture
    def raw_data_path(self) -> str:
        return os.path.join('data', 'raw', 'emails.csv')

    @pytest.fixture
    def processed_data_path(self) -> str:
        return os.path.join('data', 'processed', 'tokenized_emails.pkl')

    def test_data_integrity(self, raw_data_path: str):
        assert os.path.exists(raw_data_path), f"Raw data file not found at {raw_data_path}"
        df = pd.read_csv(raw_data_path)
        
        assert not df.empty, "Raw data is empty"
        assert 'text' in df.columns, "Missing 'text' column"
        assert 'label' in df.columns, "Missing 'label' column"
        
        spam_ratio = df['label'].mean()
        assert 0.1 <= spam_ratio <= 0.5, f"Unexpected spam ratio: {spam_ratio}"

    def test_data_preprocessing(self, raw_data_path: str, processed_data_path: str):
        df = pd.read_csv(raw_data_path)
        
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], df['label'], 
            test_size=0.2, 
            random_state=42, 
            stratify=df['label']
        )
        
        assert len(X_train) > 0, "Training data is empty"
        assert len(X_test) > 0, "Test data is empty"
        
        train_spam_ratio = y_train.mean()
        test_spam_ratio = y_test.mean()
        
        assert abs(train_spam_ratio - test_spam_ratio) < 0.05, "Unbalanced data split"

    def test_data_distribution(self, raw_data_path: str):
        df = pd.read_csv(raw_data_path)
        
        text_lengths = df['text'].str.len()
        
        assert text_lengths.mean() > 10, "Average text length too short"
        assert text_lengths.std() > 0, "Text length variation is too low"
        
        outliers = text_lengths[
            (text_lengths < text_lengths.quantile(0.01)) | 
            (text_lengths > text_lengths.quantile(0.99))
        ]
        
        assert len(outliers) / len(df) < 0.05, "Too many text length outliers"

    def test_label_encoding(self, raw_data_path: str):
        df = pd.read_csv(raw_data_path)
        
        unique_labels = df['label'].unique()
        assert len(unique_labels) == 2, "Binary classification expected"
        assert set(unique_labels) == {0, 1}, "Labels must be 0 and 1"

    def test_data_save_load(self, raw_data_path: str, processed_data_path: str):
        df = pd.read_csv(raw_data_path)
        
        df.to_pickle(processed_data_path)
        loaded_df = pd.read_pickle(processed_data_path)
        
        assert loaded_df.equals(df), "Data save/load process failed"
        os.remove(processed_data_path)