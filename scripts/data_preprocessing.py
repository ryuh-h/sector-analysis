import pandas as pd
import os


def preprocess_data(input_filename, output_filename):
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

    # Convert 'Date' column to datetime with explicit format
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

    # Show sample of 'Date' to ensure the conversion worked
    print(data[['Date']].head())

    # Drop rows with invalid 'Date' values
    initial_shape = data.shape
    data = data.dropna(subset=['Date'])
    print(f"Dropped rows with invalid 'Date'. Shape before: {initial_shape}, after: {data.shape}")

    # Drop rows with any other missing values
    initial_shape = data.shape
    data = data.dropna()
    print(f"Dropped rows with any missing values. Shape before: {initial_shape}, after: {data.shape}")

    # Standardize numeric columns (e.g., scaling numeric columns)
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    if not numeric_columns.empty:
        data[numeric_columns] = (data[numeric_columns] - data[numeric_columns].mean()) / data[numeric_columns].std()
        print(f"Standardized numeric columns: {numeric_columns.tolist()}")
    else:
        print("No numeric columns found to standardize.")

    # Perform feature engineering as needed (e.g., adding moving averages)
    if 'Close' in data.columns:
        data['Moving_Avg'] = data['Close'].rolling(window=5).mean()
    else:
        print("Column 'Close' not found for calculating moving average.")

    # Drop rows with NaN values after calculating moving average if needed
    initial_shape = data.shape
    data = data.dropna()
    print(f"Dropped rows with NaN in 'Moving_Avg'. Shape before: {initial_shape}, after: {data.shape}")

    # Save cleaned data to CSV if there's any data left
    if not data.empty:
        data.to_csv(output_path, index=False)
        print(f'Cleaned data saved to {output_path}')
    else:
        print(f"No data to save after preprocessing for file {input_filename}.")


if __name__ == "__main__":
    # Preprocess data for each raw dataset
    raw_files = ['XLK_data.csv', 'XLV_data.csv', 'XLE_data.csv', 'XLF_data.csv']
    for raw_file in raw_files:
        output_file = raw_file.replace('raw', 'cleaned')
        preprocess_data(raw_file, output_file)
