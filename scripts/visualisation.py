import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def create_visualisations(filename, root_dir):
    try:
        # Load the final clustered dataset
        input_path = os.path.join(root_dir, 'data', 'final', filename)
        visualisations_dir = os.path.join(root_dir, 'visualizations')

        # Print debug information for verification
        print(f"Input Path: {input_path}")
        print(f"Visualizations Directory: {visualisations_dir}")

        # Load data
        data = pd.read_csv(input_path)
        filename = filename.replace('_clustered.csv', '')

        # Ensure 'Date' is parsed correctly as datetime
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        else:
            raise ValueError("The 'Date' column is missing from the dataset.")

        # Visualization 1: Closing Prices Over Time
        if 'Close' in data.columns:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=data, x='Date', y='Close')
            plt.title(f'Closing Prices Over Time for {filename}')
            plt.xlabel('Date')
            plt.ylabel('Close Price')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(visualisations_dir, f'{filename}_closing_prices.png'))
            plt.close()

        # Visualization 2: K-Means Clustering Results
        if 'KMeans_Cluster' in data.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=data, x='Date', y='Close', hue='KMeans_Cluster', palette='viridis')
            plt.title(f'K-Means Clustering Results for {filename}')
            plt.xlabel('Date')
            plt.ylabel('Close Price')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(visualisations_dir, f'{filename}_kmeans_clusters.png'))
            plt.close()

        # Visualization 3: Gaussian Mixture Model Clustering Results
        if 'GMM_Cluster' in data.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=data, x='Date', y='Close', hue='GMM_Cluster', palette='plasma')
            plt.title(f'GMM Clustering Results for {filename}')
            plt.xlabel('Date')
            plt.ylabel('Close Price')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(visualisations_dir, f'{filename}_gmm_clusters.png'))
            plt.close()

        # Visualization 4: Dynamic Time Warping Distances Over Time
        if 'DTW_Distance' in data.columns:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=data, x='Date', y='DTW_Distance', color='red')
            plt.title(f'DTW Distances Over Time for {filename}')
            plt.xlabel('Date')
            plt.ylabel('DTW Distance')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(visualisations_dir, f'{filename}_dtw_distances.png'))
            plt.close()

        # Visualization 5: Evaluation Metrics (MAE and RMSE)
        evaluation_metrics_path = os.path.join(root_dir, 'data', 'evaluation_metrics.csv')
        if os.path.exists(evaluation_metrics_path):
            eval_data = pd.read_csv(evaluation_metrics_path)
            eval_data_filtered = eval_data[eval_data['Filename'] == filename]
            if not eval_data_filtered.empty:
                mae = eval_data_filtered['MAE'].values[0]
                rmse = eval_data_filtered['RMSE'].values[0]

                # Create bar chart for MAE and RMSE
                plt.figure(figsize=(8, 5))
                sns.barplot(x=['MAE', 'RMSE'], y=[mae, rmse], palette='Blues')
                plt.title(f'Model Evaluation Metrics for {filename}')
                plt.ylabel('Error Value')
                plt.tight_layout()
                plt.savefig(os.path.join(visualisations_dir, f'{filename}_evaluation_metrics.png'))
                plt.close()

        print(f'Visualizations for {filename} saved successfully.')

    except Exception as e:
        print(f"Error during visualization creation for {filename}: {e}")
        if 'data' in locals():
            print(f"Available columns in dataset: {data.columns.tolist()}")


if __name__ == "__main__":
    # Define the root directory for testing
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Define final data directory
    final_data_dir = os.path.join(ROOT_DIR, 'data', 'final')

    # Print debug information
    print(f"Root Directory: {ROOT_DIR}")
    print(f"Final Data Directory: {final_data_dir}")

    # Run visualization for each clustered dataset in the final directory
    for filename in os.listdir(final_data_dir):
        if filename.endswith('_clustered.csv'):
            create_visualisations(filename, ROOT_DIR)
