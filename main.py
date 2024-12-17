from data_loader import get_btc_data
from visualization import plot_btc_data
from feature_engineering import add_features, save_features

# Main to plot the BTC price and save features.
if __name__ == "__main__":
    # Load Bitcoin data
    btc_data = get_btc_data(start_date="2010-07-17", end_date="2024-12-16")

    # Verify the data
    print(btc_data.head())

    # Plot and save the data
    plot_btc_data(btc_data, save_path="./Graphs")

    # Add with features
    btc_data_with_features = add_features(btc_data)

    print("Data with new features :")
    print(btc_data_with_features.head())

    # Save features
    save_features(btc_data_with_features, save_path="./Graphs", file_name="btc_features.csv")
