import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import DataPreprocessor

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-02'],
        'Customer_ID': [1001, 1002],
        'Product_ID': [5001, 5002],
        'Quantity': [2, 3],
        'Unit_Price': [49.99, 39.99],
        'Sales_Revenue': [99.98, 119.97],
        'Product_Description': ['Product A', 'Product B'],
        'Product_Category': ['Category 1', 'Category 2'],
        'Product_Line': ['Line X', 'Line Y'],
        'Raw_Material': ['Material 1', 'Material 2'],
        'Region': ['North', 'South'],
        'Latitude': [42.5, 41.5],
        'Longitude': [-71.3, -70.3]
    })

def test_data_preprocessor(sample_data):
    preprocessor = DataPreprocessor()
    processed_data, transformers = preprocessor.preprocess_data(sample_data)
    
    assert 'Year' in processed_data.columns
    assert 'Month' in processed_data.columns
    assert 'Product_Category_Encoded' in processed_data.columns
    assert 'Quantity_Scaled' in processed_data.columns