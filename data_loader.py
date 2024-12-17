"""
from Yahoo finance : download data of btc-usd
"""

import yfinance as yf

def get_btc_data(start_date="2010-07-17", end_date="2024-12-16"):
    data = yf.download("BTC-USD", start=start_date, end=end_date)
    return data
