# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, TimeSeriesSplit
# from xgboost import XGBRegressor
# from typing import Dict, Tuple
# import joblib
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class SalesPredictor:
#     def __init__(self):
#         self.model = None
#         self.feature_columns = None
        
#     def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
#         """
#         Train the sales prediction model
#         """
#         try:
#             logger.info("Starting model training...")
            
#             # Initialize model
#             self.model = XGBRegressor(
#                 n_estimators=100,
#                 learning_rate=0.1,
#                 max_depth=6,
#                 subsample=0.8,
#                 colsample_bytree=0.8,
#                 random_state=42
#             )
            
#             # Perform time series split
#             tscv = TimeSeriesSplit(n_splits=5)
            
#             # Train the model
#             self.model.fit(X, y)
            
#             # Store feature columns
#             self.feature_columns = X.columns.tolist()
            
#             logger.info("Model training completed successfully")
#             return {
#                 'feature_importance': dict(zip(X.columns, self.model.feature_importances_)),
#                 'model_params': self.model.get_params()
#             }
            
#         except Exception as e:
#             logger.error(f"Error in model training: {str(e)}")
#             raise
            
#     def predict(self, X: pd.DataFrame) -> np.ndarray:
#         """
#         Make predictions using the trained model
#         """
#         if self.model is None:
#             raise ValueError("Model not trained yet")
            
#         return self.model.predict(X[self.feature_columns])
        
#     def save_model(self, path: str):
#         """Save the model to disk"""
#         if self.model is None:
#             raise ValueError("No model to save")
            
#         joblib.dump({
#             'model': self.model,
#             'feature_columns': self.feature_columns
#         }, path)
        
#     def load_model(self, path: str):
#         """Load the model from disk"""
#         model_data = joblib.load(path)
#         self.model = model_data['model']
#         self.feature_columns = model_data['feature_columns']

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from xgboost import XGBRegressor
from typing import Dict, Tuple, List
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SalesPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.categorical_columns = None
        self.encoded_feature_names = None
        
    def _preprocess_categorical_data(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Preprocess categorical columns using one-hot encoding
        """
        df_processed = df.copy()
        
        if fit:
            # Identify categorical columns (object dtype)
            self.categorical_columns = df.select_dtypes(include=['object']).columns
            logger.info(f"Detected categorical columns: {list(self.categorical_columns)}")
            
            # Apply one-hot encoding
            df_encoded = pd.get_dummies(df_processed, columns=self.categorical_columns)
            self.encoded_feature_names = df_encoded.columns
            return df_encoded
        else:
            # Apply one-hot encoding and ensure columns match training data
            df_encoded = pd.get_dummies(df_processed, columns=self.categorical_columns)
            # Add missing columns if any
            for col in self.encoded_feature_names:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
            # Remove extra columns if any
            df_encoded = df_encoded[self.encoded_feature_names]
            return df_encoded

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train the sales prediction model
        """
        try:
            logger.info("Starting model training...")
            
            # Preprocess categorical features
            logger.info("Preprocessing categorical features...")
            X_processed = self._preprocess_categorical_data(X, fit=True)
            
            # Initialize model
            self.model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            # Train the model
            self.model.fit(X_processed, y)
            
            # Store feature columns
            self.feature_columns = X_processed.columns.tolist()
            
            logger.info("Model training completed successfully")
            return {
                'feature_importance': dict(zip(X_processed.columns, self.model.feature_importances_)),
                'model_params': self.model.get_params(),
                'categorical_columns': list(self.categorical_columns) if self.categorical_columns is not None else []
            }
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Preprocess categorical features
        X_processed = self._preprocess_categorical_data(X, fit=False)
        
        return self.model.predict(X_processed[self.feature_columns])
    
    def save_model(self, path: str):
        """Save the model and preprocessing information to disk"""
        if self.model is None:
            raise ValueError("No model to save")
        
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns,
            'categorical_columns': self.categorical_columns,
            'encoded_feature_names': self.encoded_feature_names
        }, path)
    
    def load_model(self, path: str):
        """Load the model and preprocessing information from disk"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.categorical_columns = model_data['categorical_columns']
        self.encoded_feature_names = model_data['encoded_feature_names']