import pandas as pd
import os

"""
    Adds new features to Bitcoin price data:
    - 7-day and 30-day moving averages.
    - Daily return (percentage change).
    - 7-day rolling volatility (standard deviation).
    - Target column to predict the next day's closing price.
"""
def add_features(data):

    # 7-day moving average
    data['MA_7'] = data['Close'].rolling(window=7).mean()

    # 30-day moving average
    data['MA_30'] = data['Close'].rolling(window=30).mean()

    # Daily percentage change (daily return)
    data['Daily_Return'] = data['Close'].pct_change()

    # 7-day rolling standard deviation (volatility)
    data['Volatility_7'] = data['Close'].rolling(window=7).std()

    # Target column: Next day's closing price
    data['Target'] = data['Close'].shift(-1)

    # Drop rows with NaN values caused by rolling calculations
    data = data.dropna()

    return data

#Saves the enriched dataset with new features to a CSV file.
def save_features(data, save_path="./Graphs", file_name="btc_features.csv"):

    # Create the directory if it does not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Full file path
    file_path = os.path.join(save_path, file_name)

    # Save the dataset as a CSV file
    data.to_csv(file_path, index=True)
    print(f"Features saved successfully at: {file_path}")
