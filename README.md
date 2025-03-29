# SP500 Stock Market Predictor (ML)

Welcome to the **SP500 Stock Market Predictor** repository! This project uses machine learning to predict the future movements of the S&P 500 index. By leveraging historical stock data, we aim to provide insights that can help guide investment decisions.

## 📂 Project Overview

This machine learning project predicts whether the stock price of the S&P 500 will go up or down based on past market data. The model is trained using **Random Forest** and evaluated using key performance metrics. Additionally, the model uses **feature engineering** to enhance its predictions.

### Key Features:
- **Data Processing**: Retrieves and processes historical stock data for the S&P 500 index.
- **Machine Learning Model**: Uses a Random Forest Classifier to predict future market movements.
- **Performance Evaluation**: The model is evaluated based on precision, recall, F1-Score, and backtesting.
- **Backtesting**: Tests model performance over various historical periods to simulate real-world trading scenarios.
- **Feature Engineering**: Incorporates rolling averages and trends for better predictive accuracy.

## 🚀 Features

- **Data Collection**: Fetches historical stock data using `yfinance`.
- **Data Preprocessing**: Handles missing values, shifts data for prediction, and prepares target labels.
- **Model Training**: Trains a Random Forest Classifier on the stock data.
- **Feature Engineering**: Generates new features like rolling averages and trends based on multiple horizons.
- **Model Evaluation**: Evaluates the model using precision, recall, F1-Score, and accuracy.
- **Backtesting**: Simulates stock predictions over different time periods for performance evaluation.

## ⚙️ Technologies Used

- **Python**: The main programming language used.
- **yfinance**: For downloading historical stock data.
- **pandas**: For data manipulation and preparation.
- **matplotlib**: For plotting and visualizations.
- **scikit-learn**: For machine learning and model evaluation.

## 📈 Results

The model will output various graphs displaying:

- Historical vs predicted stock prices.

- Performance metrics for model evaluation.

## Example Outputs:

- Predicted vs Actual Prices: A plot comparing predicted stock prices to the actual ones.

- Model Evaluation: A printout of metrics like precision score, F1 score, etc.
  
Initial Evaluation:

- **Precision**: 57% of the predictions where the model predicted the market would go up were correct.

- **Recall**: 93% of actual upward market movements were correctly predicted.

- **F1-Score**: Balances precision and recall, giving an overall performance metric.

Backtest Evaluation:

- **Precision**: 98% of the backtest predictions where the market would go up were correct.

- **Recall**: The model captured 58% of actual upward movements during backtesting.

- **Accuracy**: Overall proportion of correct predictions (both upward and downward movements).
