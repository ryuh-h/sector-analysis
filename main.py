"""
main.py

This is the driver code that runs the full workflow of the project from data collection to model evaluation.

Key Steps:
1. Data Collection (`data_collection.py`)
2. Data Cleaning (`data_cleaning.py`)
3. Exploratory Data Analysis (`eda.py`)
4. Clustering (`clustering.py`)
5. LSTM Model Training (`lstm_model.py`)
6. Visualisations (`visualisation.py`)
7. Evaluation (`evaluation.py`)

"""

from scripts import data_collection
from scripts import data_cleaning
from scripts import eda
from scripts import clustering
from scripts import lstm_model
from scripts import visualisation
from scripts import evaluation
import os

# Define the root directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    # Define tickers and their sector names
    tickers = ['XLK', 'XLV', 'XLE', 'XLF']
    ticker_names = {
        'XLK': 'Tech',
        'XLV': 'Healthcare',
        'XLE': 'Energy',
        'XLF': 'Finance'
    }

    # Cleaned directory
    cleaned_data_dir = os.path.join(ROOT_DIR, 'data', 'cleaned')

    # Final directory
    final_data_dir = os.path.join(ROOT_DIR, 'data', 'final')

    # Model directory
    model_dir = os.path.join(ROOT_DIR, 'models')

    # Step 1: Data Collection
    print("Starting Data Collection:")
    for ticker in tickers:
        data_collection.download_data(ticker)

    # Step 2: Data Cleaning
    print("Starting Data Cleaning:")
    for ticker in tickers:
        descriptive_name = ticker_names[ticker]
        output_file = f'{descriptive_name}'
        data_cleaning.clean_data(ticker, output_file)

    # Cleaned files
    cleaned_files = [file for file in os.listdir(cleaned_data_dir) if file.endswith('.csv')]

    # Step 3: Exploratory Data Analysis
    print("Performing EDA:")
    for cleaned_file in cleaned_files:
        eda.perform_eda(cleaned_file, ROOT_DIR)

    # Step 4: Clustering
    print("Performing Clustering:")
    for cleaned_file in cleaned_files:
        clustering.perform_clustering(cleaned_file, ROOT_DIR)

    # Step 5: Model Training
    print("Training LSTM Model:")
    for filename in os.listdir(final_data_dir):
        if filename.endswith('_clustered.csv'):
            lstm_model.train_lstm(filename, ROOT_DIR)

    # Step 6: Visualisations
    print("Creating Visualisations:")
    visualisation.create_comparative_visualisations(ROOT_DIR)

    # Final data files
    final_files = [file for file in os.listdir(final_data_dir) if file.endswith('_clustered.csv')]

    # Model files
    model_files = [file for file in os.listdir(model_dir) if file.endswith('_lstm_model.keras')]

    # Step 7: Evaluation
    print("Evaluating Models:")
    for clustered_file in final_files:
        evaluation.evaluate_clustering(clustered_file, ROOT_DIR)

        # Find corresponding LSTM model file
        base_name = clustered_file.replace('_clustered.csv', '')
        model_filename = f"{base_name}_lstm_model.keras"

        if model_filename in model_files:
            evaluation.evaluate_lstm(clustered_file, model_filename, ROOT_DIR)
        else:
            print(f"No corresponding model found for {clustered_file}.")

    print("Workflow Completed Successfully.")


if __name__ == "__main__":
    main()
