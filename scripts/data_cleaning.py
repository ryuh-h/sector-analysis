"""
data_cleaning.py

This module performs data cleaning on raw stock market data.
It ensures proper formatting, removes missing values, and prepares the data for analysis.

Key Tasks:
- Convert date columns to datetime format
- Remove missing values and outliers
- Save cleaned data to `data/cleaned/`

Functions:
- clean_data(input_filename, output_filename):
  Reads raw CSV files, cleans them, and saves the processed version.

"""

import pandas as pd
import os


def clean_data(input_filename, output_filename):
    # Define paths
    input_path = os.path.join(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'), f'{input_filename}_data.csv')
    output_path = os.path.join(os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned'), f'{output_filename}_cleaned.csv')

    # Create the cleaned data directory if it doesn't exist
    if not os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned')):
        os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned'))

    # Load data
    data = pd.read_csv(input_path)

    # Rename the first column to 'Date' if it is incorrectly named
    if data.columns[0] != 'Date':
        data.rename(columns={data.columns[0]: 'Date'}, inplace=True)

    # Drop redundant rows 2 and 3
    data = data.drop(index=[0, 1])

    # Convert 'Date' column to datetime format
    data['Date'] = pd.to_datetime(data['Date'], format="%Y-%m-%d", errors='coerce')

    # Drop rows with NaN or infinite values
    initial_shape = data.shape
    data = data.replace([float('inf'), -float('inf')], float('nan'))
    data = data.dropna()
    print(f"Dropped rows with any missing values. Shape before: {initial_shape}, after: {data.shape}")

    # Save cleaned data to CSV if there's any data left
    if not data.empty:
        data.to_csv(output_path, index=False)
        print(f'Cleaned data saved to {output_path}')
    else:
        print(f"No data to save for {input_filename}.")


# Testing
if __name__ == "__main__":
    # Clean data for each raw dataset
    raw_files = ['XLK', 'XLV', 'XLE', 'XLF']
    for raw_file in raw_files:
        output_file = raw_file.replace('raw', 'cleaned')
        clean_data(raw_file, output_file)
