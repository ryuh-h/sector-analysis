import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def create_comparative_visualisations(root_dir):
    try:
        # Define paths
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

        # Ensure the 'Volatility' column exists by calculating it if missing
        if 'Volatility' not in combined_data.columns and 'Close' in combined_data.columns:
            combined_data['Volatility'] = combined_data.groupby('Sector')['Close'].transform(lambda x: x.rolling(window=30).std())

        # Comparative Visualisation 1: K-Means Clustering Results (All Sectors)
        if 'KMeans_Cluster' in combined_data.columns:
            plt.figure(figsize=(12, 8))
            sns.scatterplot(data=combined_data, x='Date', y='Close', hue='KMeans_Cluster', style='Sector', palette='viridis')
            plt.title('K-Means Clustering Results for All Sectors')
            plt.xlabel('Date')
            plt.ylabel('Close Price')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(visualisations_dir, 'all_sectors_kmeans_clusters.png'))
            plt.close()

        # Comparative Visualisation 2: Gaussian Mixture Model Clustering Results (All Sectors)
        if 'GMM_Cluster' in combined_data.columns:
            plt.figure(figsize=(12, 8))
            sns.scatterplot(data=combined_data, x='Date', y='Close', hue='GMM_Cluster', style='Sector', palette='plasma')
            plt.title('GMM Clustering Results for All Sectors')
            plt.xlabel('Date')
            plt.ylabel('Close Price')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(visualisations_dir, 'all_sectors_gmm_clusters.png'))
            plt.close()

        # Comparative Visualisation 3: K-Means Cluster Membership Over Time for Each Sector (All Sectors)
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

        # Comparative Visualisation 4: GMM Cluster Membership Over Time for Each Sector (All Sectors)
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

        # Comparative Visualisation 5: Dynamic Time Warping Distances Over Time (All Sectors)
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

        # Comparative Visualisation 6: Time-Sliced Correlation Heatmaps of Closing Prices
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

        # Comparative Visualisation 7: Volatility Over Time for All Sectors
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

        # Cluster Centroid Visualisation
        for cluster_col in ['KMeans_Cluster', 'GMM_Cluster']:
            if cluster_col in combined_data.columns:
                centroids = combined_data.groupby(['Sector', cluster_col])[
                    ['Smoothed_Close', 'Volatility']].mean().reset_index()

                # Comparative Visualisation  8 & 9: Plot Centroids for Smoothed Close Price
                plt.figure(figsize=(12, 8))
                sns.barplot(data=centroids, x='Sector', y='Smoothed_Close', hue=cluster_col, palette='viridis')
                plt.title(f'Cluster Centroids for Smoothed Close Price (by {cluster_col})')
                plt.xlabel('Sector')
                plt.ylabel('Average Smoothed Close')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(visualisations_dir, f'{cluster_col}_centroids_smoothed_close.png'))
                plt.close()

                # Comparative Visualisation  10 & 11: Plot Centroids for Volatility
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
        print(f"Error during creating comparative visualisation: {e}")
        if 'combined_data' in locals():
            print(f"Available columns in combined dataset: {combined_data.columns.tolist()}")


# Testing
if __name__ == "__main__":
    # Define paths
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    final_data_dir = os.path.join(ROOT_DIR, 'data', 'final')

    # Print debug information root directory and final data directory
    print(f"Root Directory: {ROOT_DIR}")
    print(f"Final Data Directory: {final_data_dir}")

    # Run visualisation for each clustered dataset in the final directory
    for filename in os.listdir(final_data_dir):
        if filename.endswith('_clustered.csv'):
            create_visualisations(filename, ROOT_DIR)

    # Run comparative visualisations for all datasets
    create_comparative_visualisations(ROOT_DIR)
