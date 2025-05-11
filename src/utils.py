import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class StockDataUtils:
    def __init__(self, log_level: str = 'INFO'):
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_stock_data(
        self,