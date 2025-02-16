from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import pandas as pd
from dtaidistance import dtw
import os

def perform_clustering(filename, root_dir, n_clusters=3):
    # Define paths
    input_path = os.path.join(root_dir, 'data', 'cleaned', filename)
    output_dir = os.path.join(root_dir, 'data', 'final')

    # Print input and output paths for verification
    print(f"Input Path: {input_path}")
    print(f"Output Directory: {output_dir}")

    # Create the final directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data
    data = pd.read_csv(input_path)

    # Apply a moving average to smooth the 'Close' column
    # Helps capture general trend and remove the noise of daily fluctuations
    data['Smoothed_Close'] = data['Close'].rolling(window=30, min_periods=1).mean()

    # Standardise the smoothed 'Close' columns for clustering
    # All values Mean: 0 and Standard Deviation: 1
    # All values are centered around the mean at 0 to ensure the min & max values have similiar importance in clustering
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data[['Smoothed_Close']])

    # K-Means Clustering
    # Groups similar stock performance to see if there is a pattern/trend, splits data into n_clusters
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    data['KMeans_Cluster'] = kmeans.fit_predict(normalized_data)

    # Gaussian Mixture Model (GMM) Clustering
    # Similar to K-Means, but assigns probability to each point rather than hard assignment, making it more flexible
    # and can capture more complex market movements
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    data['GMM_Cluster'] = gmm.fit_predict(normalized_data)

    # Dynamic Time Warping (DTW) Distance Calculation
    # Measures how similarly time-series data moves, even if movement at different speeds
    # Measures whether 2+ stocks move in unison, even at varying rates
    dtw_distances = []
    for i in range(len(normalized_data) - 1):
        dtw_distance = dtw.distance(normalized_data[i], normalized_data[i + 1])
        dtw_distances.append(dtw_distance)
    data['DTW_Distance'] = [0] + dtw_distances  # Pad first value with 0

    # Add marker for major market event (for context in later visualizations)
    data['Major_Event'] = data['Date'].apply(lambda x: 'Pandemic' if '2020' in str(x) else 'None')

    # Extract base name for output file
    basename = filename.replace('_cleaned.csv', '')
    output_filename = f'{basename}_clustered.csv'
    output_path = os.path.join(output_dir, output_filename)

    # Save clustered data to final directory
    print(f"Saving clustered data to: {output_path}")
    data.to_csv(output_path, index=False)


# Testing
if __name__ == "__main__":
    # Define paths
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cleaned_data_dir = os.path.join(ROOT_DIR, 'data', 'cleaned')

    # Get a list of all CSV files in the cleaned directory
    cleaned_files = [file for file in os.listdir(cleaned_data_dir) if file.endswith('.csv')]

    # Perform clustering for each cleaned dataset
    for cleaned_file in cleaned_files:
        perform_clustering(cleaned_file, ROOT_DIR)
