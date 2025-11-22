# Stock Inside Day Pattern Predictor

A machine learning project that predicts "Inside Day" patterns in stock market data using Random Forest classification.

## Overview

This project analyzes historical stock market data to identify and predict Inside Day patterns - a technical analysis pattern where a stock's trading range (high and low) is completely contained within the previous day's range. The model uses various technical indicators and machine learning to predict when these patterns will occur.

## What is an Inside Day Pattern?

An Inside Day occurs when:
- Today's high is **lower** than yesterday's high
- Today's low is **higher** than yesterday's low

This pattern often indicates market consolidation and can signal potential breakouts.

## Features

- **Multi-stock analysis**: Processes multiple stock tickers simultaneously
- **Technical indicators**: Computes various market indicators including volatility, range compression, and price changes
- **Machine learning prediction**: Uses Random Forest classifier for pattern prediction
- **Class imbalance handling**: Implements oversampling strategies to handle imbalanced data
- **Model evaluation**: Provides comprehensive accuracy metrics and confusion matrices

## Requirements

```bash
pip install pandas numpy scikit-learn imbalanced-learn glob
```

### Python Libraries Used
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning models and preprocessing
- `imbalanced-learn` - Handling imbalanced datasets
- `glob` - File pattern matching
- `os` - Operating system interface

## Data Structure

### Input Data Format
The project expects CSV files in a `stocks/` directory with the following columns:
- `Date` - Trading date
- `Open` - Opening price
- `High` - Highest price of the day
- `Low` - Lowest price of the day
- `Close` - Closing price
- `Adj Close` - Adjusted closing price
- `Volume` - Trading volume

### Computed Features
The model creates several technical indicators:
1. **range** - Daily trading range (High - Low)
2. **prev_range** - Previous day's range
3. **compression** - Ratio of current range to previous range
4. **range_shrink** - Binary indicator if next day's range is smaller
5. **pct_change** - Percentage change in closing price
6. **volatility** - Range relative to closing price
7. **prev_volatility** - Previous day's volatility
8. **volatility_ratio** - Ratio of current to previous volatility

## Usage

### 1. Data Preparation
Place your stock CSV files in a `stocks/` directory:
```
stocks/
├── AAPL.csv
├── GOOGL.csv
├── MSFT.csv
└── ...
```

### 2. Running the Analysis

```python
# Load and combine stock data
DATA_PATH = "stocks/*.csv"
files = glob.glob(DATA_PATH)

# Process a subset (for faster training)
half = len(files) // 50  # Adjust this divisor as needed
files = files[:half]

# The notebook will automatically:
# - Load and combine all stock files
# - Compute technical indicators
# - Train the Random Forest model
# - Evaluate performance
```

### 3. Model Configuration
Current model parameters:
- **Algorithm**: Random Forest Classifier
- **Trees**: 250
- **Max Depth**: 11
- **Test Split**: 20%
- **Random State**: 42

### 4. Testing on New Data
The notebook includes code to test on individual stocks:
```python
df = pd.read_csv("stocks/SAFM.csv")  # Replace with your stock file
# The model will predict Inside Day patterns for this stock
```

## Model Performance

Based on the notebook execution:

### Training Set Results
- **Accuracy**: ~89%
- **Precision** (Class 0): 90%
- **Precision** (Class 1): 59%
- **Recall** (Class 0): 99%
- **Recall** (Class 1): 16%

### Test Set Results (SAFM stock)
- **Accuracy**: ~89.6%
- **Confusion Matrix**: Shows strong prediction for non-Inside Days, moderate performance for Inside Days

## File Structure

```
project/
├── CAP4630 Project Combined Version.ipynb  # Main notebook
├── README.md                                # This file
├── stocks/                                  # Stock data directory
│   ├── A.csv
│   ├── AA.csv
│   └── ... (other stock files)
└── requirements.txt                         # (optional) Python dependencies
```

## Important Notes

1. **Class Imbalance**: Inside Days are relatively rare (~11.5% of samples), making this an imbalanced classification problem
2. **Oversampling**: The code includes optional RandomOverSampler to balance classes during training
3. **Data Quality**: Ensure your stock data is clean and doesn't contain missing values for critical columns
4. **Computational Load**: Processing many stocks can be memory-intensive; adjust the subset size accordingly

## Potential Improvements

- Experiment with different ML algorithms (XGBoost, Neural Networks)
- Add more technical indicators (RSI, MACD, Bollinger Bands)
- Implement time-series cross-validation
- Add feature importance analysis
- Include sector/industry information
- Implement trading strategy backtesting based on predictions

## Warnings

⚠️ **Financial Disclaimer**: This is an educational project. Do not use this model for actual trading without thorough testing and risk assessment. Past performance does not guarantee future results.

⚠️ **Deprecation Warnings**: The notebook may show pandas deprecation warnings related to `groupby.apply()`. These don't affect functionality but should be addressed in future updates.

## Author

CAP4630 Final Project - Stock Analyzer

## License

This project is for educational purposes. Please ensure you have rights to use any stock data included.
