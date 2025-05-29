# Project-Four
# Earthquake Forecasting and Clustering Analysis in the Pacific Northwest

This project applies machine learning and geospatial analysis techniques to forecast earthquake activity and identify seismic risk zones in the Pacific Northwest (Northern California, Oregon, and Washington). It combines classification, regression, and clustering models to support hazard mitigation and planning efforts.

---

## Project Features

### 1. Earthquake Occurrence Classification
- **Model**: Random Forest Classifier
- **Goal**: Predict whether an earthquake will occur in a given region and month.
- **Methods**:
  - Spatiotemporal feature engineering
  - Seasonal signal extraction using sine/cosine transformations
  - Class imbalance handled with SMOTE
- **Evaluation**: Precision, recall, F1-score, confusion matrix

### 2. Earthquake Magnitude Forecasting
- **Model**: Random Forest Regressor
- **Goal**: Estimate earthquake magnitude for regions where an event is predicted.
- **Features**:
  - Depth, latitude, longitude
  - Prior quake count and max magnitude
- **Metrics**: RMSE, R² score

### 3. Clustering Analysis
- **Method**: KMeans clustering
- **Goal**: Discover spatial groupings based on earthquake features
- **Visualization**: Residual maps, cluster boundaries

---

## Key Techniques

- Temporal binning (monthly)
- Spatial binning (0.5° × 0.5° grid)
- Synthetic oversampling with SMOTE
- Feature engineering from datetime and geolocation data
- Classification threshold tuning

---

## Dataset

- Source: USGS / PNSN
- Time frame: 1970-2025
- Filtered for: Magnitude >= 2.0 and valid coordinates
- Columns: `time`, `latitude`, `longitude`, `depth`, `mag`, `gap`, `rms`, `nst`, and derived features


