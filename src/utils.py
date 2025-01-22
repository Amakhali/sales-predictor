import pandas as pd
import numpy as np
from typing import Dict, Any
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(path, 'r') as f:
        return json.load(f)

def save_metrics(metrics: Dict[str, float], path: str):
    """Save model metrics to JSON file"""
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate model performance metrics"""
    return {
        'mse': np.mean((y_true - y_pred) ** 2),
        'rmse': np.sqrt(np.mean((y_true - y_pred) ** 2)),
        'mae': np.mean(np.abs(y_true - y_pred)),
        'r2': 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    }

def setup_logging(log_path: str):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )