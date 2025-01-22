import pandas as pd
import numpy as np
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.feature_columns = []
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features for the model
        """
        try:
            logger.info("Starting feature engineering...")
            df_featured = df.copy()
            
            # Add rolling means
            df_featured = self._add_rolling_features(df_featured)
            
            # Add lag features
            df_featured = self._add_lag_features(df_featured)
            
            # Add interaction features
            df_featured = self._add_interaction_features(df_featured)
            
            self.feature_columns = df_featured.columns.tolist()
            logger.info("Feature engineering completed successfully")
            return df_featured
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise
            
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling mean features"""
        df = df.sort_values('Date')
        df['Rolling_7_Sales'] = df.groupby('Product_ID')['Sales_Revenue'].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        return df
        
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features"""
        for lag in [1, 7, 30]:
            df[f'Sales_Lag_{lag}'] = df.groupby('Product_ID')['Sales_Revenue'].shift(lag)
        return df
        
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features"""
        df['Price_Quantity'] = df['Unit_Price'] * df['Quantity']
        df['Revenue_Per_Unit'] = df['Sales_Revenue'] / df['Quantity']
        return df