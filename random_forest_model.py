import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Train a Random Forest model to predict BTC prices.
def train_random_forest(data):
  
    # Features and target
    features = ['Close', 'MA_7', 'MA_30', 'Volatility_7', 'Daily_Return']
    target = 'Target'

    X = data[features]
    y = data[target]

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"Random Forest RMSE: {rmse:.2f}")

    return {"model": model, "rmse": rmse}


#Predict the price for a future date using the Random Forest model.
def predict_random_forest(model, recent_data, days_ahead):
    
    # Simulate features for the next 'days_ahead'
    future_data = recent_data.copy()
    for _ in range(days_ahead):
        # Predict one step ahead
        X_future = future_data[['Close', 'MA_7', 'MA_30', 'Volatility_7', 'Daily_Return']].iloc[-1].values.reshape(1, -1)
        next_price = model.predict(X_future)[0]

        # Update simulated data with the prediction
        new_row = {
            'Close': next_price,
            'MA_7': future_data['Close'].rolling(window=7).mean().iloc[-1],
            'MA_30': future_data['Close'].rolling(window=30).mean().iloc[-1],
            'Volatility_7': future_data['Close'].rolling(window=7).std().iloc[-1],
            'Daily_Return': (next_price - future_data['Close'].iloc[-1]) / future_data['Close'].iloc[-1]
        }
        future_data = pd.concat([future_data, pd.DataFrame([new_row])], ignore_index=True)

    return future_data['Close'].iloc[-1]
