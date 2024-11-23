import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_visualisations(filename, root_dir):
    try:
        # Load the final clustered dataset
        input_path = os.path.join(root_dir, 'data', 'final', filename)
        visualisations_dir = os.path.join(root_dir, 'visualisations')

        # Print debug information for verification
        print(f"Input Path: {input_path}")
        print(f"Visualisations Directory: {visualisations_dir}")

        # Load data
        data = pd.read_csv(input_path)
        filename = filename.replace('_clustered.csv', '')

        # Ensure 'Date' is parsed correctly as datetime
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        else:
            raise ValueError("The 'Date' column is missing from the dataset.")

        # Visualisation 1: Closing Prices Over Time (Individual)
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

        # Visualisation 2: Volatility Over Time (Rolling Standard Deviation)
        if 'Close' in data.columns:
            plt.figure(figsize=(10, 6))
            data['Volatility'] = data['Close'].rolling(window=30).std()
            sns.lineplot(data=data, x='Date', y='Volatility')
            plt.title(f'Volatility Over Time for {filename}')
            plt.xlabel('Date')
            plt.ylabel('Volatility (Rolling Std Dev)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(visualisations_dir, f'{filename}_volatility.png'))
            plt.close()

        print(f'Visualisations for {filename} saved successfully.')

    except Exception as e:
        print(f"Error during visualisation creation for {filename}: {e}")
        if 'data' in locals():
            print(f"Available columns in dataset: {data.columns.tolist()}")

def create_comparative_visualisations(root_dir):
    try:
        # Define final data directory
        final_data_dir = os.path.join(root_dir, 'data', 'final')
        visualisations_dir = os.path.join(root_dir, 'visualisations')

        # Load all clustered datasets into a single DataFrame
        combined_data = pd.DataFrame()
        for filename in os.listdir(final_data_dir):
            if filename.endswith('_clustered.csv'):
                sector_data = pd.read_csv(os.path.join(final_data_dir, filename))
                sector_name = filename.replace('_clustered.csv', '')
                sector_data['Sector'] = sector_name
                if 'Date' in sector_data.columns:
                    sector_data['Date'] = pd.to_datetime(sector_data['Date'], errors='coerce')
                combined_data = pd.concat([combined_data, sector_data], ignore_index=True)

        # Comparative Visualisation 1: K-Means Clustering Results (All Sectors, Improved)
        if 'KMeans_Cluster' in combined_data.columns:
            plt.figure(figsize=(12, 8))
            sns.scatterplot(data=combined_data, x='Date', y='Close', hue='KMeans_Cluster', style='Sector', palette='viridis')
            plt.title('K-Means Clustering Results for All Sectors')
            plt.xlabel('Date')
            plt.ylabel('Close Price')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(visualisations_dir, 'improved_all_sectors_kmeans_clusters.png'))
            plt.close()

        # Comparative Visualisation 2: Gaussian Mixture Model Clustering Results (All Sectors, Improved)
        if 'GMM_Cluster' in combined_data.columns:
            plt.figure(figsize=(12, 8))
            sns.scatterplot(data=combined_data, x='Date', y='Close', hue='GMM_Cluster', style='Sector', palette='plasma')
            plt.title('GMM Clustering Results for All Sectors')
            plt.xlabel('Date')
            plt.ylabel('Close Price')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(visualisations_dir, 'improved_all_sectors_gmm_clusters.png'))
            plt.close()

        # New Comparative Visualisation: Cluster Membership Over Time for Each Sector (Stacked for All Sectors)
        if 'KMeans_Cluster' in combined_data.columns:
            sectors = combined_data['Sector'].unique()
            fig, axes = plt.subplots(4, 1, figsize=(12, 20), sharex=True)
            colors = ['red', 'blue', 'green', 'orange']
            for i, sector in enumerate(sectors):
                sector_data = combined_data[combined_data['Sector'] == sector]
                sns.lineplot(ax=axes[i], data=sector_data, x='Date', y='KMeans_Cluster', color=colors[i])
                axes[i].set_title(f'Cluster Membership Over Time (K-Means) for {sector}')
                axes[i].set_xlabel('Date')
                axes[i].set_ylabel('Cluster')
                axes[i].tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(visualisations_dir, 'stacked_cluster_membership_over_time_kmeans.png'))
            plt.close()

        if 'GMM_Cluster' in combined_data.columns:
            fig, axes = plt.subplots(4, 1, figsize=(12, 20), sharex=True)
            for i, sector in enumerate(sectors):
                sector_data = combined_data[combined_data['Sector'] == sector]
                sns.lineplot(ax=axes[i], data=sector_data, x='Date', y='GMM_Cluster', color=colors[i])
                axes[i].set_title(f'Cluster Membership Over Time (GMM) for {sector}')
                axes[i].set_xlabel('Date')
                axes[i].set_ylabel('Cluster')
                axes[i].tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(visualisations_dir, 'stacked_cluster_membership_over_time_gmm.png'))
            plt.close()

        # Comparative Visualisation 3: Dynamic Time Warping Distances Over Time (Subplots for All Sectors, Improved)
        if 'DTW_Distance' in combined_data.columns:
            sectors = combined_data['Sector'].unique()
            fig, axes = plt.subplots(4, 1, figsize=(12, 20), sharex=True)
            for i, sector in enumerate(sectors):
                sector_data = combined_data[combined_data['Sector'] == sector]
                sector_data = sector_data.copy()  # Avoid SettingWithCopyWarning
                sector_data['DTW_Distance_Smoothed'] = sector_data['DTW_Distance'].rolling(window=30).mean()
                sns.lineplot(ax=axes[i], data=sector_data, x='Date', y='DTW_Distance_Smoothed', color='red')
                axes[i].set_ylim(0, 0.06)  # Set consistent y-axis limit for comparison
                axes[i].set_title(f'DTW Distances Over Time for {sector}')
                axes[i].set_xlabel('Date')
                axes[i].set_ylabel('Smoothed DTW Distance')
                axes[i].tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(visualisations_dir, 'all_sectors_dtw_distances.png'))
            plt.close()

        # Comparative Visualisation 4: Time-Sliced Correlation Heatmaps of Closing Prices
        pivot_data = combined_data.pivot(index='Date', columns='Sector', values='Close')
        time_slices = ['2014-2016', '2017-2019', '2020-2024']
        for time_slice in time_slices:
            start_year, end_year = time_slice.split('-')
            sliced_data = pivot_data.loc[f'{start_year}':f'{end_year}']
            plt.figure(figsize=(10, 8))
            sns.heatmap(sliced_data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
            plt.title(f'Correlation Heatmap of Closing Prices ({time_slice})')
            plt.tight_layout()
            plt.savefig(os.path.join(visualisations_dir, f'correlation_heatmap_{time_slice}.png'))
            plt.close()

        # Comparative Visualisation 5: Volatility Over Time for All Sectors (Combined Plot for Comparison)
        plt.figure(figsize=(12, 8))
        for sector in sectors:
            sector_data = combined_data[combined_data['Sector'] == sector].copy()
            sector_data['Volatility'] = sector_data['Close'].rolling(window=30).std()
            sns.lineplot(data=sector_data, x='Date', y='Volatility', label=sector)
        plt.title('Volatility Over Time for All Sectors (Combined)')
        plt.xlabel('Date')
        plt.ylabel('Volatility (Rolling Std Dev)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(visualisations_dir, 'all_sectors_combined_volatility.png'))
        plt.close()

        # Cluster Centroid Visualisations
        for cluster_col in ['KMeans_Cluster', 'GMM_Cluster']:
            if cluster_col in combined_data.columns:
                centroids = combined_data.groupby(['Sector', cluster_col])[
                    ['Smoothed_Close', 'Volatility']].mean().reset_index()

                # Plot Centroids for Smoothed Close Price
                plt.figure(figsize=(12, 8))
                sns.barplot(data=centroids, x='Sector', y='Smoothed_Close', hue=cluster_col, palette='viridis')
                plt.title(f'Cluster Centroids for Smoothed Close Price (by {cluster_col})')
                plt.xlabel('Sector')
                plt.ylabel('Average Smoothed Close')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(visualisations_dir, f'{cluster_col}_centroids_smoothed_close.png'))
                plt.close()

                # Plot Centroids for Volatility
                plt.figure(figsize=(12, 8))
                sns.barplot(data=centroids, x='Sector', y='Volatility', hue=cluster_col, palette='plasma')
                plt.title(f'Cluster Centroids for Volatility (by {cluster_col})')
                plt.xlabel('Sector')
                plt.ylabel('Average Volatility')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(visualisations_dir, f'{cluster_col}_centroids_volatility.png'))
                plt.close()

        print('Comparative visualisations saved successfully.')

    except Exception as e:
        print(f"Error during comparative visualisation creation: {e}")
        if 'combined_data' in locals():
            print(f"Available columns in combined dataset: {combined_data.columns.tolist()}")

if __name__ == "__main__":
    # Define the root directory for testing
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Define final data directory
    final_data_dir = os.path.join(ROOT_DIR, 'data', 'final')

    # Print debug information
    print(f"Root Directory: {ROOT_DIR}")
    print(f"Final Data Directory: {final_data_dir}")

    # Run visualisation for each clustered dataset in the final directory
    for filename in os.listdir(final_data_dir):
        if filename.endswith('_clustered.csv'):
            create_visualisations(filename, ROOT_DIR)

    # Run comparative visualisations for all datasets
    create_comparative_visualisations(ROOT_DIR)
