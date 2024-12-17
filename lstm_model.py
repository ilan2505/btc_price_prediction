import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Train an LSTM model to predict BTC prices.
def train_lstm_model(data, look_back=10):
    # Features and target
    features = ['Close', 'MA_7', 'MA_30', 'Volatility_7', 'Daily_Return', 'Target']
    target = 'Target'

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[features])

    # Prepare the data for LSTM (create sequences)
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, :-1])  # Features for the look_back period
        y.append(scaled_data[i, -1])  # Target (next day price)

    X = np.array(X)
    y = np.array(y)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"LSTM RMSE: {rmse:.2f}")

    return model, scaler


# Predict the price for a future date using the LSTM model.
def predict_lstm(model, recent_data, days_ahead, scaler):
    features = ['Close', 'MA_7', 'MA_30', 'Volatility_7', 'Daily_Return', 'Target']

    # Check that all features are present in recent_data
    missing_features = [feature for feature in features if feature not in recent_data.columns]
    
    if missing_features:
        # If any features are missing, raise an error
        raise ValueError(f"The following features are missing in recent_data: {', '.join(missing_features)}")
    
    # Scale recent data
    scaled_data = scaler.transform(recent_data[features])

    # Debug: Display the shape of scaled_data to verify dimensions
    print(f"Shape of scaled_data: {scaled_data.shape}")

    # Prepare data for LSTM (using the last 10 time steps)
    lstm_input = scaled_data[-10:, :-1]  # Last 10 time steps, excluding target column
    lstm_input = np.expand_dims(lstm_input, axis=0)

    # Debug: Check the shape of lstm_input before prediction
    print(f"Shape of lstm_input before prediction: {lstm_input.shape}")

    # Predict future prices iteratively
    for _ in range(days_ahead):
        prediction = model.predict(lstm_input)
        
        # Add the prediction to the input data
        next_step = np.hstack((lstm_input[0, -1, :-1], prediction[0]))  # Combine last feature with predicted price
        next_step_scaled = np.expand_dims(next_step, axis=0)  # Shape it correctly
        
        # Ensure correct shape for vstack: adjust the first dimension of next_step_scaled
        next_step_scaled = next_step_scaled.reshape(1, 1, -1)  # Make sure the shape is (1, 1, n_features)
        
        # Now lstm_input has the shape (1, 10, n_features), so next_step_scaled should have the same feature count
        lstm_input = np.concatenate((lstm_input[:, 1:, :], next_step_scaled), axis=1)

    # Prepare the input for inverse_transform, ensure 6 features are passed to the scaler
    # Ensure all features are present
    combined_data = np.hstack((lstm_input[0, -1, :]))[np.newaxis, :]
    
    # Debug: Display the shape of combined_data before inverse_transform
    print(f"Shape of combined_data before inverse_transform: {combined_data.shape}")
    
    # Manually add the 'Target' column if necessary
    if combined_data.shape[1] != 6:
        # If we have 5 features, add the 'Target' column (the last prediction) at the end
        target_column = prediction[0]  # The current prediction for the price
        combined_data = np.hstack((combined_data, target_column.reshape(-1, 1)))  # Add the 'Target' column
        print(f"Shape of combined_data after adding Target: {combined_data.shape}")

    # Inverse transform the prediction
    next_price = scaler.inverse_transform(combined_data)[:, -1]
    return next_price[0]
