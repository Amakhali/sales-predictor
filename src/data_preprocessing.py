import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Preprocess the input data for model training
        """
        try:
            logger.info("Starting data preprocessing...")
            df_processed = df.copy()
            
            # Handle dates
            df_processed['Date'] = pd.to_datetime(df_processed['Date'])
            date_features = self._extract_date_features(df_processed['Date'])
            df_processed = pd.concat([df_processed, date_features], axis=1)
            
            # Encode categorical variables
            categorical_cols = ['Product_Description', 'Product_Category', 
                              'Product_Line', 'Raw_Material', 'Region']
            for col in categorical_cols:
                df_processed, encoder = self._encode_categorical(df_processed, col)
                self.encoders[col] = encoder
            
            # Scale numerical variables
            numerical_cols = ['Quantity', 'Unit_Price', 'Latitude', 'Longitude']
            for col in numerical_cols:
                df_processed, scaler = self._scale_numerical(df_processed, col)
                self.scalers[col] = scaler
            
            logger.info("Data preprocessing completed successfully")
            return df_processed, {'scalers': self.scalers, 'encoders': self.encoders}
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise
            
    def _extract_date_features(self, date_series: pd.Series) -> pd.DataFrame:
        """Extract temporal features from date column"""
        return pd.DataFrame({
            'Year': date_series.dt.year,
            'Month': date_series.dt.month,
            'DayOfWeek': date_series.dt.dayofweek
        })
    
    def _encode_categorical(self, df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, LabelEncoder]:
        """Encode categorical variables"""
        encoder = LabelEncoder()
        df[f'{column}_Encoded'] = encoder.fit_transform(df[column])
        return df, encoder
    
    def _scale_numerical(self, df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, StandardScaler]:
        """Scale numerical variables"""
        scaler = StandardScaler()
        df[f'{column}_Scaled'] = scaler.fit_transform(df[[column]])
        return df, scaler