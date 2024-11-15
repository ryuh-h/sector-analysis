# In evaluation.py
import os
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


# Function to evaluate the clustering effectiveness using Silhouette Score
def evaluate_clustering(filename, root_dir):
    try:
        # Load the clustered dataset
        data = pd.read_csv(os.path.join(root_dir, 'data', 'final', filename))
        # Calculate and print Silhouette Score based on 'KMeans_Cluster' labels
        score = silhouette_score(data[['Close']], data['KMeans_Cluster'])
        print(f'Silhouette Score for {filename}: {score:.4f}')
    except Exception as e:
        print(f"Error during clustering evaluation for {filename}: {e}")


# Function to evaluate LSTM model's performance
def evaluate_lstm(test_filename, model_filename, root_dir):
    try:
        # Define paths for the test data and model
        test_data_path = os.path.join(root_dir, 'data', 'final', test_filename)
        model_path = os.path.join(root_dir, 'models', model_filename)

        # Debug: Ensure paths are correct
        print(f"Loading test data from: {test_data_path}")
        print(f"Loading model from: {model_path}")

        # Check if the model file exists
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' not found. Please ensure the file path is correct.")
            return

        # Load the test dataset and the trained model
        test_data = pd.read_csv(test_data_path)['Close'].values.reshape(-1, 1)
        model = tf.keras.models.load_model(model_path)  # Load `.keras` format

        # Normalize test data
        scaler = MinMaxScaler(feature_range=(0, 1))
        test_data_normalized = scaler.fit_transform(test_data)

        # Prepare test data for LSTM model input
        timesteps = 5
        X_test = np.array([test_data_normalized[i:i + timesteps] for i in range(len(test_data_normalized) - timesteps)])
        y_test = test_data[timesteps:].flatten()

        # Make predictions
        predictions = model.predict(X_test).flatten()
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()  # Rescale predictions back to original scale

        # Adjust actual values to match the length of predictions
        actual_values = y_test
        min_length = min(len(actual_values), len(predictions))

        # Truncate both arrays to have the same length
        actual_values = actual_values[:min_length]
        predictions = predictions[:min_length]

        # Calculate evaluation metrics
        mae = mean_absolute_error(actual_values, predictions)
        rmse = np.sqrt(mean_squared_error(actual_values, predictions))

        # Print the evaluation results
        print(f'Model Evaluation for {test_filename}:')
        print(f'Mean Absolute Error (MAE): {mae:.4f}')
        print(f'Root Mean Square Error (RMSE): {rmse:.4f}')
    except Exception as e:
        print(f"Error during LSTM evaluation for {test_filename}: {e}")


if __name__ == "__main__":
    # Define the root directory of the project
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Define directories for clustered CSV files and model files
    final_data_dir = os.path.join(root_dir, 'data', 'final')
    model_dir = os.path.join(root_dir, 'models')

    # List all clustered CSV files and model files
    clustered_files = [file for file in os.listdir(final_data_dir) if file.endswith('_clustered.csv')]
    model_files = [file for file in os.listdir(model_dir) if file.endswith('_lstm_model.keras')]

    # Iterate through all clustered CSV files to evaluate clustering and LSTM models
    for clustered_file in clustered_files:
        evaluate_clustering(clustered_file, root_dir)

        # Find corresponding LSTM model file
        base_name = clustered_file.replace('_clustered.csv', '')
        model_filename = f"{base_name}_lstm_model.keras"

        if model_filename in model_files:
            evaluate_lstm(clustered_file, model_filename, root_dir)
        else:
            print(f"No corresponding model found for {clustered_file}.")
