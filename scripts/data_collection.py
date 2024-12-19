import yfinance as yf
import os


def download_data(ticker, start_date='2014-01-01', end_date='2024-01-01'):
    # Download data from Yahoo Finance
    data = yf.download(ticker, start=start_date, end=end_date)

    # Save data to CSV
    data.to_csv(os.path.join(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'), f'{ticker}_data.csv'))
    print(f'Data for {ticker} saved successfully.')


if __name__ == "__main__":
    # Download data for each sector ETF
    tickers = ['XLK', 'XLV', 'XLE', 'XLF']
    for ticker in tickers:
        download_data(ticker)
