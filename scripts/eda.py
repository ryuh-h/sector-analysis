import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(filename, root_dir):
    # Define paths
    input_path = os.path.join(root_dir, 'data', 'cleaned', filename)
    output_dir = os.path.join(root_dir, 'visualisations')

    # Print input and output paths for verification
    print(f"Input Path: {input_path}")
    print(f"Output Directory: {output_dir}")

    # Create the visualisations directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data
    data = pd.read_csv(input_path)

    # Ensure 'Date' is in datime format
    data['Date'] = pd.to_datetime(data['Date'], format="%Y-%m-%d %H:%M:%S%z", errors='coerce')

    # Create 'Daily_Return' column
    data['Daily_Return'] = data['Close'].pct_change()
    data['Daily_Return'] = pd.to_numeric(data['Daily_Return'], errors='coerce')

    # Remove outliers based on Daily_Return IQR method
    # Financial data could contain extreme daily returns due to market crashes or earnings announcements,
    # removing extreme values makes the dataset more stable for analysis, clustering and LSTM model performance.
    Q1 = data['Daily_Return'].quantile(0.25)
    Q3 = data['Daily_Return'].quantile(0.75)
    IQR = Q3 - Q1
    data = data[(data['Daily_Return'] >= (Q1 - 1.5 * IQR)) & (data['Daily_Return'] <= (Q3 + 1.5 * IQR))]

    # Generate descriptive statistics
    print(f"Summary statistics for {filename}:")
    print(data.describe())

    # Calculate average returns and volatility
    avg_return = data['Daily_Return'].mean()
    volatility = data['Daily_Return'].std()
    print(f"Average Daily Return for {filename}: {avg_return:.7f}")
    print(f"Volatility for {filename}: {volatility:.7f}")

    # Drop NaN values to be sure
    data = data.dropna()

    # Extract the base name for saving plots
    base_name = filename.replace('_cleaned.csv', '')

    # Plot closing price over time
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x='Date', y='Close')
    plt.title(f'{base_name} - Closing Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    closing_price_plot_path = os.path.join(output_dir, f'{base_name}_closing_prices.png')
    print(f"Saving closing prices plot to: {closing_price_plot_path}")
    plt.savefig(closing_price_plot_path)
    plt.close()

    # Plot daily returns
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Daily_Return'], bins=50, kde=True)
    plt.title(f'{base_name} - Daily Returns Distribution (After Outlier Removal)')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.tight_layout()
    daily_return_plot_path = os.path.join(output_dir, f'{base_name}_daily_returns.png')
    print(f"Saving daily returns plot to: {daily_return_plot_path}")
    plt.savefig(daily_return_plot_path)
    plt.close()

    # Identify major macroeconomic events and their impacts (COVID-19)
    covid_start = pd.to_datetime('2020-03-01')
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x='Date', y='Close', label='Closing Price')
    plt.axvline(x=covid_start, color='r', linestyle='--', label='COVID-19 Start')
    plt.title(f'{base_name} - Impact of COVID-19 on Closing Prices')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    covid_impact_plot_path = os.path.join(output_dir, f'{base_name}_covid_impact.png')
    print(f"Saving COVID-19 impact plot to: {covid_impact_plot_path}")
    plt.savefig(covid_impact_plot_path)
    plt.close()

    # Volatility Over Time (Rolling Standard Deviation)
    plt.figure(figsize=(10, 6))
    data['Volatility'] = data['Close'].rolling(window=30).std()
    sns.lineplot(data=data, x='Date', y='Volatility')
    plt.title(f'Volatility Over Time for {filename}')
    plt.xlabel('Date')
    plt.ylabel('Volatility (Rolling Std Dev)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    volatility_plot_path = os.path.join(output_dir, f'{base_name}_volatility.png')
    print(f"Saving volatility plot to: {volatility_plot_path}")
    plt.savefig(volatility_plot_path)
    plt.close()


# Testing
if __name__ == "__main__":
    # Define paths
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cleaned_data_dir = os.path.join(ROOT_DIR, 'data', 'cleaned')

    # Get a list of all CSV files in the cleaned directory
    cleaned_files = [file for file in os.listdir(cleaned_data_dir) if file.endswith('.csv')]

    # Perform EDA for each cleaned dataset
    for cleaned_file in cleaned_files:
        perform_eda(cleaned_file, ROOT_DIR)
