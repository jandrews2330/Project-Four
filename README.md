# Project-Four
# Earthquake Forecasting and Clustering in the Pacific Northwest

This project explores earthquake occurrence and magnitude prediction in the Pacific Northwest using machine learning and unsupervised clustering techniques. It combines geospatial, temporal, and seismic data features to model earthquake risk.

## Project Objectives

- **Forecast Earthquake Occurrence** using a Random Forest Classifier  
- **Predict Earthquake Magnitude** with a Random Forest Regressor  
- **Identify Seismic Zones** using KMeans clustering on spatiotemporal features  
- **Visualize Results** through interactive maps and performance metrics  

## Dataset

- **Source**: USGS Earthquake Catalog (1970–2025)  
- **Features**: `latitude`, `longitude`, `depth`, `mag`, `gap`, `dmin`, `rms`, `nst`, `time`  
- **Region Focus**: Northern California, Oregon, and Washington  
- **Format**: Preprocessed CSV (`pnw_final.csv`)  

## Methodology

### Data Engineering

- Spatiotemporal binning (0.5° × 0.5°, monthly intervals)  
- Lag features and previous magnitude indicators  
- Handling missing data and outlier filtering  

### Classification Model

- **Target**: Earthquake Occurrence (`quake_occurred`)  
- **Model**: `RandomForestClassifier`  
- **Resampling**: SMOTE for class imbalance  
- **Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix  

### Regression Model

- **Target**: Earthquake Magnitude (`mag`)  
- **Model**: `RandomForestRegressor`  
- **Metrics**: RMSE, R² Score, Residual Error Analysis  

### Clustering Analysis

- **Algorithm**: KMeans  
- **Input Features**: Geospatial bins and magnitude  
- **Purpose**: Identify latent seismic zones and patterns  

## Visualizations

- Feature importance bar charts for both models  
- Residual maps for both models error analysis  
- Scatter plots of earthquake magnitude predictions vs. actuals  
- Clustering results overlayed on maps  

## Technologies Used

- Python (Pandas, NumPy, Scikit-learn, imbalanced-learn)  
- Plotly for interactive visualizations  
- Jupyter Notebook  

## Key Findings

- Earthquake **occurrence** is best predicted by **spatial location and timing**  
- Earthquake **magnitude** is more influenced by **prior seismic intensity**  
- Clustering revealed **distinct seismic hotspots** that align with fault zones 

## References 
1. Scikit-learn Documentation: https://scikit-learn.org/stable/documentation.html
> Used for RandomForestClassifier, RandomForestRegressor and evaluation metrics.
2. imblearn (imbalanced-learn): https://imbalanced-learn.org/stable/
> Used for SMOTE and other class balancing techniques.
3. Plotly Express & Dash: https://plotly.com/python/
> Used for interactive error maps and visualizations.
4. Scientific Reports - Improving earthquake prediction accuracy in Los Angeles with machine learning:  https://www.nature.com/articles/s41598-024-76483-x
5. Pandas Documentation: https://pandas.pydata.org/docs/
> Essential for data manipulation, cleaning, temporal grouping, and spatial binning
6. NumPy Documentation: https://numpy.org/doc/
> Core library for numerical computing, used for spatial grids, feature calculations
7. USGS Earthquake Catalog API: https://earthquake.usgs.gov/fdsnws/event/1/
> Source of historical and real-time earthquake data for North America and beyond
8. Scikit-learn KMeans Clustering: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
> For grouping earthquake epicenters or locations by magnitude/frequency/depth
9. Class Discussions and Activities
 


