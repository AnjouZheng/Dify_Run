import os
import pandas as pd
import numpy as np
from typing import Dict, List, Union
from sklearn.model_selection import train_test_split
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

class SpamDataLoader:
    def __init__(self, data_dir: str = 'data'):
        self.raw_data_dir = os.path.join(data_dir, 'raw')
        self.processed_data_dir = os.path.join(data_dir, 'processed')
        self.splits_data_dir = os.path.join(data_dir, 'splits')
        
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.splits_data_dir, exist_ok=True)

    def load_raw_data(self, filename: str = 'spam_dataset.csv') -> pd.DataFrame:
        try:
            file_path = os.path.join(self.raw_data_dir, filename)
            df = pd.read_csv(file_path)
            logging.info(f"Successfully loaded raw data from {file_path}")
            return df
        except FileNotFoundError:
            logging.error(f"Raw data file {filename} not found")
            raise

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        required_columns = ['text', 'label']
        if not all(col in df.columns for col in required_columns):
            logging.error("Missing required columns in dataset")
            raise ValueError("Dataset must contain 'text' and 'label' columns")

        df['text'] = df['text'].fillna('')
        df['text'] = df['text'].str.lower()
        
        logging.info("Data preprocessing completed")
        return df

    def split_dataset(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=df['label']
        )
        
        train_df, val_df = train_test_split(
            train_df, 
            test_size=0.2, 
            random_state=random_state, 
            stratify=train_df['label']
        )

        logging.info(f"Dataset split completed: Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)}")

        return {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }

    def save_dataset_splits(self, dataset_splits: Dict[str, pd.DataFrame]) -> None:
        for split_name, split_data in dataset_splits.items():
            file_path = os.path.join(self.splits_data_dir, f'{split_name}_data.csv')
            split_data.to_csv(file_path, index=False)
            logging.info(f"Saved {split_name} data to {file_path}")

    def create_tensorflow_dataset(
        self, 
        df: pd.DataFrame, 
        batch_size: int = 32
    ) -> tf.data.Dataset:
        texts = df['text'].values
        labels = df['label'].values

        dataset = tf.data.Dataset.from_tensor_slices((texts, labels))
        dataset = dataset.shuffle(buffer_size=len(df))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    def load_and_prepare_data(self) -> Dict[str, tf.data.Dataset]:
        raw_df = self.load_raw_data()
        preprocessed_df = self.preprocess_data(raw_df)
        dataset_splits = self.split_dataset(preprocessed_df)
        self.save_dataset_splits(dataset_splits)

        return {
            split_name: self.create_tensorflow_dataset(split_data)
            for split_name, split_data in dataset_splits.items()
        }