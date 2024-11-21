# Sector Analysis using LSTM and Clustering Techniques

This project analyzes key financial sectors, including **Technology**, **Healthcare**, **Energy**, and **Finance**, using historical stock data to understand market trends and predict future movements. The project leverages **LSTM** (Long Short-Term Memory) neural networks for time-series forecasting and employs clustering techniques (**K-Means** and **Gaussian Mixture Models**) to identify underlying patterns in the data. Visualizations are used throughout to provide intuitive insights.

## Project Overview

- **Sectors Analyzed**: 
  - Technology (XLK)
  - Healthcare (XLV)
  - Energy (XLE)
  - Finance (XLF)

- **Key Techniques**: 
  - LSTM for time-series forecasting
  - K-Means and GMM for clustering analysis
  - Dynamic Time Warping (DTW) for time series similarity

- **Visualisations**: Created to explore trends, evaluate the impact of key macroeconomic events (e.g., COVID-19), and analyze clusters.

The project is organized into multiple steps: **data collection**, **preprocessing**, **exploratory data analysis (EDA)**, **clustering**, **LSTM model training**, **visualizations**, and **evaluation**.

## Project Structure
The project is organized into the following main folders and scripts:

### **Directory Structure**
```
FYP Code/
│
├── data/
│   ├── raw/                # Raw data files downloaded using yfinance
│   ├── cleaned/            # Cleaned datasets after preprocessing
│   ├── final/              # Final clustered datasets ready for analysis
│
├── models/                 # Saved LSTM models for each sector
│
├── scripts/
│   ├── data_collection.py  # Downloads historical stock data from Yahoo Finance
│   ├── data_preprocessing.py # Preprocesses raw data (cleaning, feature engineering)
│   ├── eda.py              # Conducts exploratory data analysis and visualizations
│   ├── clustering.py       # Clusters sectors based on historical trends
│   ├── lstm_model.py       # Trains LSTM models for sector prediction
│   ├── visualisation.py    # Generates final visualizations of results
│   └── evaluation.py       # Evaluates clustering and prediction performance
│
├── visualisations/         # Contains generated plots and visual reports
├── main.py                 # Orchestrates the entire workflow end-to-end
└── README.md               # Documentation for running the project
```

## Features

1. **Data Collection**:
   - Downloads historical data for the four sectors (XLK, XLV, XLE, XLF).

2. **Data Preprocessing**:
   - Cleans data, converts data types, removes outliers, and calculates new features like daily returns.

3. **Exploratory Data Analysis (EDA)**:
   - Generates visualizations for closing prices, daily returns, and the impact of key macroeconomic events.

4. **Clustering**:
   - Uses K-Means and GMM to cluster the daily returns and gain insight into market behaviour.

5. **LSTM Model Training**:
   - Trains LSTM models on the clustered data to predict future price movements.

6. **Visualisation**:
   - Provides visual insights into price changes over time, clustering results, DTW distances, and model performance metrics.

7. **Evaluation**:
   - Evaluates the LSTM models using **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)**.

## Visualisations

- **Closing Prices Over Time**: Line plots of closing prices for each sector.
- **Daily Returns Distribution**: Histogram showing the distribution of daily returns (after outlier removal).
- **Impact of COVID-19**: Line plots showing the effect of the COVID-19 pandemic on each sector's closing prices.
- **Clustering Results**: Scatter plots of K-Means and GMM clustering results.
- **Dynamic Time Warping (DTW)**: Plots showing the DTW distances over time.
- **Evaluation Metrics**: Bar charts showing MAE and RMSE of trained LSTM models.

## Results Overview
The project results include:
- **Clusters of Financial Sectors**: Using K-Means and GMM, each sector has been grouped based on historical behavior, revealing similarities.
- **LSTM Predictions**: LSTM models were used to predict future trends based on historical data, with evaluation metrics showing the performance for each sector.
- **Visualization Reports**: Detailed visualizations that illustrate sector trends, the impact of major events like COVID-19, and clustering results.
