"""
data_collection.py

This module downloads historical stock market data for selected Exchange-Traded Funds (ETFs) using the yfinance library.
It retrieves daily stock price data and saves it as CSV files in the `data/raw/` directory for further processing.

ETFs Covered:
- XLK (Technology)
- XLV (Healthcare)
- XLE (Energy)
- XLF (Finance)

Functions:
- download_data(ticker, start_date='2014-01-01', end_date='2024-01-01'):
  Fetches stock data for a given ticker within the specified time range and saves it to CSV.

"""

import yfinance as yf
import os


def download_data(ticker, start_date='2014-01-01', end_date='2024-01-01'):
    # Download data from Yahoo Finance
    data = yf.download(ticker, start=start_date, end=end_date)

    # Save data to CSV
    data.to_csv(os.path.join(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'), f'{ticker}_data.csv'))
    print(f'Data for {ticker} saved successfully.')


# Testing
if __name__ == "__main__":
    # Download data for each sector ETF
    tickers = ['XLK', 'XLV', 'XLE', 'XLF']
    for ticker in tickers:
        download_data(ticker)
