import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any

class StockPricePredictionModel(nn.Module):
    def __init__(
        self,