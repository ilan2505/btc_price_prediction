from data_loader import get_btc_data
from visualization import plot_btc_data

if __name__ == "__main__":
    # Load Bitcoin data
    btc_data = get_btc_data(start_date="2010-07-17", end_date="2024-12-16")

    # Verify the data
    print(btc_data.head())

    # Plot and save the data
    plot_btc_data(btc_data, save_path="./Graphs")
