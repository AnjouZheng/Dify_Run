import pytest
import pandas as pd
import numpy as np
from src.data_loader import StockDataLoader

class TestStockDataLoader:
    def setup_method(self):
        self.data_loader = StockDataLoader()

    def test_load_raw_data(self):
        raw_data = self.data_loader.load_raw_data()
        assert isinstance(raw_data, pd.DataFrame)
        assert not raw_data.empty
        assert len(raw_data.columns) > 0

    def test_data_integrity(self):
        raw_data = self.data_loader.load_raw_data()
        assert not raw_data.isnull().any().any(), "Data contains null values"
        assert len(raw_data) > 100, "Insufficient data samples"

    def test_data_columns(self):
        raw_data = self.data_loader.load_raw_data()
        expected_columns = [
            'date', 
            'open', 
            'high', 
            'low', 
            'close', 
            'volume', 
            'stock_symbol'
        ]
        for col in expected_columns:
            assert col in raw_data.columns, f"Missing column: {col}"

    def test_data_types(self):
        raw_data = self.data_loader.load_raw_data()
        type_checks = {
            'date': pd.Timestamp,
            'open': np.float64,
            'high': np.float64,
            'low': np.float64,
            'close': np.float64,
            'volume': np.int64,
            'stock_symbol': str
        }
        for col, dtype in type_checks.items():
            assert raw_data[col].dtype == dtype, f"Incorrect type for {col}"

    def test_date_range(self):
        raw_data = self.data_loader.load_raw_data()
        date_column = raw_data['date']
        assert (date_column.max() - date_column.min()).days > 365, "Insufficient date range"

    def test_load_market_indices(self):
        indices_data = self.data_loader.load_market_indices()
        assert isinstance(indices_data, pd.DataFrame)
        assert not indices_data.empty
        assert 'index_name' in indices_data.columns
        assert 'value' in indices_data.columns

    def test_data_normalization(self):
        normalized_data = self.data_loader.normalize_data()
        assert isinstance(normalized_data, pd.DataFrame)
        assert (normalized_data['close'].min() >= 0 and normalized_data['close'].max() <= 1), "Normalization failed"

    @pytest.mark.parametrize("stock_symbol", ["AAPL", "GOOGL", "MSFT"])
    def test_specific_stock_loading(self, stock_symbol):
        stock_data = self.data_loader.load_stock_by_symbol(stock_symbol)
        assert not stock_data.empty
        assert all(stock_data['stock_symbol'] == stock_symbol)

    def test_data_augmentation(self):
        augmented_data = self.data_loader.augment_data()
        assert len(augmented_data) > len(self.data_loader.load_raw_data())