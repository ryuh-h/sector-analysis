import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os


def train_lstm(filename, root_dir):
    # Define paths
    input_path = os.path.join(root_dir, 'data', 'final', filename)
    output_dir = os.path.join(root_dir, 'models')

    # Print input and output paths for verification
    print(f"Input Path: {input_path}")
    print(f"Output Directory: {output_dir}")

    # Create the models directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data
    data = pd.read_csv(input_path)

    # Use 'Close' price as the target variable for forecasting
    data = data[['Close']].values

    # Normalize data to range [0, 1] using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    # Split data into training and testing datasets
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    # Prepare training data (X_train, y_train) to predict the next value
    timesteps = 14  # Use more historical data for better predictions
    X_train = np.array([train_data[i:i + timesteps] for i in range(len(train_data) - timesteps)])
    y_train = np.array([train_data[i + timesteps] for i in range(len(train_data) - timesteps)])

    # Prepare testing data (X_test, y_test) for evaluation
    X_test = np.array([test_data[i:i + timesteps] for i in range(len(test_data) - timesteps)])
    y_test = np.array([test_data[i + timesteps] for i in range(len(test_data) - timesteps)])

    # Define LSTM model
    model = Sequential()
    model.add(Input(shape=(timesteps, 1)))
    model.add(LSTM(700, return_sequences=False))  # Predict only the next value, hence return_sequences=False
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train LSTM model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Save the trained model to the models directory in the new Keras format
    basename = filename.replace('_clustered.csv', '')
    model_output_path = os.path.join(output_dir, f'{basename}_lstm_model.keras')
    model.save(model_output_path)
    print(f'Model for {basename} saved successfully.')


# Testing
if __name__ == "__main__":
    # Determine paths
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    final_data_dir = os.path.join(ROOT_DIR, 'data', 'final')

    # Print debug information for root directory and final data directory
    print(f"Root Directory: {ROOT_DIR}")
    print(f"Final Data Directory: {final_data_dir}")

    # Train LSTM model for each clustered dataset in the final directory
    for filename in os.listdir(final_data_dir):
        if filename.endswith('_clustered.csv'):
            train_lstm(filename, ROOT_DIR)
