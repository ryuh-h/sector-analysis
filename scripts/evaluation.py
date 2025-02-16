import os
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns


# Function to measures how well K-Means clustering assigned each data to each cluster, measured using Silhouette score
# that measures the likeliness of the points within each cluster measured from 0 to 1
def evaluate_clustering(filename, root_dir):
    try:
        # Load the clustered dataset
        data = pd.read_csv(os.path.join(root_dir, 'data', 'final', filename))
        # Calculate Silhouette Score based on 'KMeans_Cluster' labels
        score = silhouette_score(data[['Close']], data['KMeans_Cluster'])
        print(f'Silhouette Score for {filename}: {score:.4f}')
        return score
    except Exception as e:
        print(f"Error during clustering evaluation for {filename}: {e}")
        return None


# Function to evaluate LSTM model's performance and store metrics
def evaluate_lstm(test_filename, model_filename, root_dir):
    try:
        # Define paths
        test_data_path = os.path.join(root_dir, 'data', 'final', test_filename)
        model_path = os.path.join(root_dir, 'models', model_filename)

        # Print input and output paths for verification
        print(f"Loading test data from: {test_data_path}")
        print(f"Loading model from: {model_path}")

        # Check if the model file exists
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' not found.")
            return None, None, None, None

        # Load the test dataset and the trained model
        test_data = pd.read_csv(test_data_path)['Close'].values.reshape(-1, 1)
        model = tf.keras.models.load_model(model_path)  # Load .keras format

        # Normalize test data, LSTM works better when data normalised from 0 to 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        test_data_normalized = scaler.fit_transform(test_data)

        # Prepare training data (X_train, y_train) to predict the next value
        # LSTM remembers n-periods in its memory cells, then uses this n-periods to make a prediction on the n+1 day
        # 5 days are used as prediction periods to predict the 15th day's stock prices
        timesteps = 5
        X_test = np.array([test_data_normalized[i:i + timesteps] for i in range(len(test_data_normalized) - timesteps)])
        y_test = test_data[timesteps:].flatten()

        # Make predictions
        predictions = model.predict(X_test).flatten()
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()  # Undo normalisation

        # Ensures the actual and predicted data have the same number of points
        actual_values = y_test
        min_length = min(len(actual_values), len(predictions))
        # Truncate both arrays to have the same length
        actual_values = actual_values[:min_length]
        predictions = predictions[:min_length]

        # Calculate evaluation metrics
        # 1.Mean absolute error - Calculates the average error between actual vs predicted, measures how wrong the
        # model is on average, treats all errors equally, applied when all errors should be minimised
        mae = mean_absolute_error(actual_values, predictions)
        # 2.Root Mean Squared Error - Calculates the average error between actual vs predicted, measures how wrong the
        # model is on average, but penalises large errors more than MAE, applied when large prediction errors are costly
        rmse = np.sqrt(mean_squared_error(actual_values, predictions))

        # Print the evaluation results
        print(f'Model Evaluation for {test_filename}:')
        print(f'Mean Absolute Error (MAE): {mae:.4f}')
        print(f'Root Mean Square Error (RMSE): {rmse:.4f}')

        return mae, rmse, actual_values, predictions

    except Exception as e:
        print(f"Error during LSTM evaluation for {test_filename}: {e}")
        return None, None, None, None


# Testing
if __name__ == "__main__":
    # Define paths
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    final_data_dir = os.path.join(root_dir, 'data', 'final')
    model_dir = os.path.join(root_dir, 'models')
    visualisations_dir = os.path.join(root_dir, 'visualisations')

    # Create the visualisations directory if it doesn't exist
    if not os.path.exists(visualisations_dir):
        os.makedirs(visualisations_dir)

    # List all clustered CSV files and model files
    clustered_files = [file for file in os.listdir(final_data_dir) if file.endswith('_clustered.csv')]
    model_files = [file for file in os.listdir(model_dir) if file.endswith('_lstm_model.keras')]

    # Initialize lists to store metrics for all sectors
    silhouette_scores = []
    mae_scores = []
    rmse_scores = []
    sector_names = []
    lstm_predictions = []
    lstm_actuals = []

    # Iterate through all clustered CSV files to evaluate clustering and LSTM models
    for clustered_file in clustered_files:
        sector_name = clustered_file.replace('_clustered.csv', '')
        sector_names.append(sector_name)

        # Evaluate Clustering
        silhouette_score_value = evaluate_clustering(clustered_file, root_dir)
        if silhouette_score_value is not None:
            silhouette_scores.append(silhouette_score_value)

        # Find corresponding LSTM model file
        model_filename = f"{sector_name}_lstm_model.keras"
        if model_filename in model_files:
            mae, rmse, actual_values, predictions = evaluate_lstm(clustered_file, model_filename, root_dir)
            if mae is not None and rmse is not None:
                mae_scores.append(mae)
                rmse_scores.append(rmse)
                lstm_predictions.append(predictions)
                lstm_actuals.append(actual_values)
        else:
            print(f"No corresponding model found for {clustered_file}.")

    # Visualise Silhouette Scores for all sectors
    if silhouette_scores:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=sector_names, y=silhouette_scores, hue=sector_names, palette='viridis', dodge=False, legend=False)
        plt.ylabel("Silhouette Score")
        plt.title("Clustering Evaluation - Silhouette Scores for All Sectors")
        output_path_silhouette = os.path.join(visualisations_dir, 'all_sectors_silhouette_scores.png')
        plt.savefig(output_path_silhouette)
        plt.close()
        print(f"Silhouette Scores visualisation saved to: {output_path_silhouette}")

    # Visualise MAE and RMSE for all sectors
    if mae_scores and rmse_scores:
        plt.figure(figsize=(12, 6))
        x = np.arange(len(sector_names))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        ax.bar(x - width/2, mae_scores, width, label='MAE', color='skyblue')
        ax.bar(x + width/2, rmse_scores, width, label='RMSE', color='steelblue')

        # Add labels and title
        ax.set_ylabel('Error Value')
        ax.set_title('Model Evaluation Metrics (MAE and RMSE) for All Sectors')
        ax.set_xticks(x)
        ax.set_xticklabels(sector_names)
        ax.legend()

        # Save the visualisation
        output_path_metrics = os.path.join(visualisations_dir, 'all_sectors_evaluation_metrics.png')
        plt.savefig(output_path_metrics)
        plt.close()
        print(f"Evaluation metrics visualisation saved to: {output_path_metrics}")

    # Visualise LSTM Predictions vs Actual for all sectors
    if lstm_predictions and lstm_actuals:
        plt.figure(figsize=(14, 8))
        for i, sector_name in enumerate(sector_names):
            if i < len(lstm_predictions):
                plt.plot(lstm_actuals[i], label=f'Actual Values - {sector_name}', linestyle='-', alpha=0.6)
                plt.plot(lstm_predictions[i], label=f'Predicted Values - {sector_name}', linestyle='--', alpha=0.6)

        plt.xlabel('Time')
        plt.ylabel('Closing Price')
        plt.title('LSTM Model Predictions vs Actual for All Sectors')
        plt.legend()
        output_path_pred = os.path.join(visualisations_dir, 'all_sectors_lstm_predictions.png')
        plt.savefig(output_path_pred)
        plt.close()
        print(f"LSTM Predictions visualisation saved to: {output_path_pred}")
