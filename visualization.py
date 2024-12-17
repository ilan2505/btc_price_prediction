"""
    Plot the historical closing prices of Bitcoin and save the plot as an image file.
"""

import matplotlib.pyplot as plt
import os

def plot_btc_data(data, title="Bitcoin Historical Price", save_path="./Graphs"):
    # Ensure the save directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Closing Price', color='blue')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid()

    # Save the plot as an image
    file_path = os.path.join(save_path, "btc_price_plot.png")
    plt.savefig(file_path)
    print(f"Plot saved at {file_path}")

    # Close the plot to avoid overlap in future plots
    plt.close()
