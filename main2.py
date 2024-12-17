from data_loader import get_btc_data
from feature_engineering import add_features, save_features
from random_forest_model import train_random_forest, predict_random_forest
from lstm_model import train_lstm_model, predict_lstm
from datetime import datetime, timedelta
import re
from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__":
    # Load Bitcoin data and add features
    btc_data = get_btc_data(start_date="2010-07-17", end_date="2024-12-16")
    btc_data_with_features = add_features(btc_data)

    # Train Random Forest and LSTM models
    rf_results = train_random_forest(btc_data_with_features)
    lstm_results = train_lstm_model(btc_data_with_features)

    # Ask the user for a future date
    while True:
        date_input = input("Enter a future date (YYYY-MM-DD): ")

        # Check if the date format is correct
        if re.match(r"^\d{4}-\d{2}-\d{2}$", date_input):
            print("Valid date format")
            
            # Convert the date to a usable format
            future_date = datetime.strptime(date_input, "%Y-%m-%d")
            today = btc_data_with_features.index[-1]
            days_ahead = (future_date - today).days

            # Ensure the future date is actually in the future
            if days_ahead < 1:
                print("The date must be in the future.")
            else:
                # Display the last known price
                last_known_price = btc_data_with_features['Close'].iloc[-1].item()  # .item() ensures we get the value
                last_known_date = btc_data_with_features.index[-1].strftime('%Y-%m-%d')
                #print(f"Last known price for {last_known_date}: {last_known_price:.2f} USD")

                # Random Forest prediction
                rf_prediction = predict_random_forest(rf_results['model'], btc_data_with_features, days_ahead)

                # LSTM prediction
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaler.fit(btc_data_with_features[['Close', 'MA_7', 'MA_30', 'Volatility_7', 'Daily_Return', 'Target']])
                lstm_prediction = predict_lstm(lstm_results[0], btc_data_with_features, days_ahead, scaler)

                # Display predictions
                print("\n*******************************************************************")
                print(f"\nLast known price for {last_known_date}: {last_known_price:.2f} USD")
                print(f"Predicted price for {future_date.date()}:")
                print(f"Random Forest: {rf_prediction:.2f} USD")
                print(f"LSTM: {lstm_prediction:.2f} USD")
            
            break  # Exit the loop after valid input

        else:
            print("Invalid date format. Please use YYYY-MM-DD.")  # Ask again if the format is incorrect
