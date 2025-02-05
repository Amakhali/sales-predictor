# Sales Prediction Model
An advanced machine learning model for predicting sales using XGBoost with categorical data handling.

## 🎯 Project Overview
This project implements a robust sales prediction system using XGBoost regression. It features automatic handling of categorical variables, time series validation, and a modular, production-ready codebase.

### Key Features
- Automated categorical data preprocessing
- Time series cross-validation
- Feature importance analysis
- Model persistence capabilities
- Comprehensive error handling and logging
- Production-ready code structure

## 🛠️ Technical Architecture
```
sales-predictor/
├── src/
│   ├── __init__.py
│   ├── model.py          # Core model implementation
│   ├── data_preprocessing.py  # Data preprocessing utilities
│   └── utils.py         # Helper functions
|   |__feature_engineering.py
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Exploratory Data Analysis
|   |__02_feature_engineering.ipynb
│   └── 03_model_selection.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_model.py
│   └── test_preprocessing.py
├── data/
│   ├── sample_data/             # Original dataset
│   └── processed/       # Preprocessed data
├── models/              # Saved model artifacts
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore
```

## 💻 Installation & Usage

### Prerequisites
- Python 3.8+
- pip or conda

### Setup
```bash
# Clone the repository
git clone https://github.com/Amakhali/sales-predictor.git
cd sales-predictor

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start
```python
from src.model import SalesPredictor

# Initialize the model
predictor = SalesPredictor()

# Train
results = predictor.train(X_train, y_train)

# Predict
predictions = predictor.predict(X_test)

# Save model
predictor.save_model('models/sales_predictor_v1.joblib')
```

## 📊 Model Performance
- Mean Absolute Error: X.XX
- Root Mean Square Error: X.XX
- R² Score: X.XX

## 🔬 Technical Deep Dive

### Categorical Data Handling
The model automatically handles categorical variables through:
1. Automatic detection of categorical columns
2. One-hot encoding for optimal feature representation
3. Consistent preprocessing between training and prediction

### Feature Importance Analysis
```python
# Example feature importance visualization
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(importance_dict):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(importance_dict.values()), 
                y=list(importance_dict.keys()))
    plt.title('Feature Importance')
    plt.show()
```

## 🧪 Testing
```bash
# Run tests
python -m pytest tests/
```

## 📈 Future Improvements
1. Implement hyperparameter tuning using Optuna
2. Add more advanced feature engineering
3. Deploy model as a REST API
4. Add real-time monitoring capabilities

## 📚 Documentation
Detailed documentation is available in the [docs](docs/) directory.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
