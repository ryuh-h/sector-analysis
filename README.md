Sector Analysis using LSTM and Clustering Techniques

This project analyzes key financial sectors including Technology, Healthcare, Energy, and Finance, using historical stock data to understand market trends and predict future movements. The project leverages LSTM (Long Short-Term Memory) neural networks for time-series forecasting and employs clustering techniques (K-Means and Gaussian Mixture Models) to identify underlying patterns in the data. Visualizations are used throughout to provide intuitive insights.


Project Overview

Sectors Analyzed: Technology (XLK), Healthcare (XLV), Energy (XLE), and Finance (XLF).

Key Techniques: LSTM for time-series forecasting, K-Means and GMM for clustering analysis, and Dynamic Time Warping (DTW) for time series similarity.

Visualizations: Created to explore trends, evaluate the impact of key macroeconomic events (e.g., COVID-19), and analyze clusters.

The project is organized into multiple steps: data collection, preprocessing, exploratory data analysis (EDA), clustering, LSTM model training, visualizations, and evaluation.


Project Structure

FYP Code/
├── data/
│   ├── raw/                 # Raw downloaded data
│   ├── cleaned/             # Cleaned data after preprocessing
│   ├── final/               # Final data after clustering
│   ├── evaluation_metrics.csv # Evaluation results (MAE, RMSE)
├── models/                  # Trained LSTM model files (.keras)
├── scripts/
│   ├── data_collection.py   # Script for data collection
│   ├── data_preprocessing.py# Script for data cleaning and feature engineering
│   ├── eda.py               # Script for exploratory data analysis
│   ├── clustering.py        # Script for clustering data
│   ├── lstm_model.py        # Script for training LSTM models
│   ├── visualisation.py     # Script for generating visualizations
│   ├── evaluation.py        # Script for model evaluation
├── visualizations/          # Generated visualizations
├── main.py                  # Main script to run the full analysis


Features

Data Collection: Downloads historical data for the four sectors (XLK, XLV, XLE, XLF).

Data Preprocessing: Cleans data, converts data types, removes outliers, and calculates new features like daily returns.

Exploratory Data Analysis (EDA): Generates visualizations for closing prices, daily returns, and the impact of key macroeconomic events.

Clustering: Uses K-Means and GMM to cluster the daily returns and gain insight into market behavior.

LSTM Model Training: Trains LSTM models on the clustered data to predict future price movements.

Visualization: Provides visual insights into price changes over time, clustering results, DTW distances, and model performance metrics.

Evaluation: Evaluates the LSTM models using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).


Visualizations

Closing Prices Over Time: Line plots of closing prices for each sector.

Daily Returns Distribution: Histogram showing the distribution of daily returns (after outlier removal).

Impact of COVID-19: Line plots showing the effect of the COVID-19 pandemic on each sector's closing prices.

Clustering Results: Scatter plots of K-Means and GMM clustering results.

Dynamic Time Warping (DTW): Plots showing the DTW distances over time.

Evaluation Metrics: Bar charts showing MAE and RMSE of trained LSTM models.

Setup Instructions


How to Use

Run Full Analysis: Execute main.py to go through the entire pipeline from data collection to model evaluation.

Visualize Results: The visualisations directory contains plots of different aspects of the analysis.

Experimentation: Modify LSTM parameters like timesteps, units, and epochs in lstm_model.py to experiment with predictions.


Acknowledgements

Pandas, Scikit-learn, TensorFlow, and Matplotlib: Key tools used for data handling, clustering, and modeling.

Yahoo Finance API: Used for downloading sectoral data.


