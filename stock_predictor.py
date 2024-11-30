import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Fetch data
def fetch_data(ticker, start_date="1990-01-01"):
    data = yf.Ticker(ticker).history(period="max")
    data = data.loc[start_date:].copy()
    data.drop(columns=["Dividends", "Stock Splits"], inplace=True)
    return data

# Set up targets
def setup_targets(data):
    data["Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
    return data

# Train and evaluate the model
def train_model(train, predictors, model):
    model.fit(train[predictors], train["Target"])
    return model

def evaluate_model(test, predictors, model, threshold=0.5):
    probs = model.predict_proba(test[predictors])[:, 1]
    preds = (probs >= threshold).astype(int)
    preds = pd.Series(preds, index=test.index, name="Predictions")
    return preds

# Backtest logic
def backtest(data, model, predictors, start=2500, step=250, threshold=0.5):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[:i].copy()
        test = data.iloc[i:i + step].copy()
        preds = evaluate_model(test, predictors, model, threshold)
        combined = pd.concat([test["Target"], preds], axis=1)
        all_predictions.append(combined)
    return pd.concat(all_predictions)

# Feature engineering
def add_features(data, horizons):
    new_predictors = []
    for horizon in horizons:
        rolling_averages = data.rolling(horizon).mean()

        ratio_column = f"Close_Ratio_{horizon}"
        data[ratio_column] = data["Close"] / rolling_averages["Close"]

        trend_column = f"Trend_{horizon}"
        data[trend_column] = data.shift(1).rolling(horizon).sum()["Target"]

        new_predictors += [ratio_column, trend_column]
    return data.dropna(), new_predictors

# Main function
def main():
    # Fetch and prepare data
    sp500 = fetch_data("^GSPC")
    sp500 = setup_targets(sp500)

    # Split data
    train = sp500.iloc[:-100]
    test = sp500.iloc[-100:]
    initial_predictors = ["Close", "Volume", "Open", "High", "Low"]

    # Model setup
    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
    model = train_model(train, initial_predictors, model)

    # Evaluate initial model
    preds = evaluate_model(test, initial_predictors, model, threshold=0.6)
    print("Initial Precision:", precision_score(test["Target"], preds))
    print("Initial Recall:", recall_score(test["Target"], preds))
    print("Initial F1-Score:", f1_score(test["Target"], preds))

    # Feature engineering and retraining
    horizons = [2, 5, 60, 250, 1000]
    sp500, new_predictors = add_features(sp500, horizons)

    # Retrain the model with new predictors
    train = sp500.iloc[:-100]  # Updated training data with new features
    test = sp500.iloc[-100:]   # Updated test data with new features
    model = train_model(train, new_predictors, model)  # Retrain with new predictors

    # Backtesting
    predictions = backtest(sp500, model, new_predictors, threshold=0.6)
    print("Backtest Precision:", precision_score(predictions["Target"], predictions["Predictions"]))
    print("Backtest Recall:", recall_score(predictions["Target"], predictions["Predictions"]))
    print("Backtest F1-Score:", f1_score(predictions["Target"], predictions["Predictions"]))
    print("Accuracy:", accuracy_score(predictions["Target"], predictions["Predictions"]))

    # Plotting
    predictions["Predictions"].value_counts().plot(kind="bar", figsize=(10, 6))
    plt.title("Prediction Distribution")
    plt.show() 

if __name__ == "__main__":
    main()

'''
OUTPUT

Initial Precision: The proportion of correctly predicted upward movements (1s) out of all predicted upward movements
    Interpretation: About 57% of the times the model predicted the market would go up, it was correct.

Intial Recall: The proportion of actual upward movements that the model correctly identified
    Interpretation: The model successfully captured 93% of all actual upward movements, but it may have over-predicted.

Initial F1-Score: The harmonic mean of precision and recall, balancing the trade-off between them
    Interpretation: The model achieves a decent balance between precision and recall during the initial evaluation

Backtest Precsion: Nearly 98% of the predictions that the market would go up were correct during backtesting

Backtest F1-SCore: The model captured 58% of actual upward movements during backtesting

Accuracy: The overall proportion of correct predictions (both upward and downward movements)

'''