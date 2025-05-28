# Project-Four
🔧 Phase 1: Data Preparation (ETL.ipynb)
Loaded and cleaned the PNW earthquake dataset.

Dropped irrelevant columns (id, time, updated, place, etc.).

Removed null values to ensure clean modeling input.

Prepared versions of the data for both supervised and unsupervised learning by:

One-hot encoding categorical variables like magType.

Scaling numeric features when needed.

📈 Phase 2: Supervised Learning (supervised_model.ipynb)
Goal: Predict earthquake magnitude (mag).

➤ Model 1: Random Forest Regressor
RMSE: ~0.41

R²: ~0.30

Feature importance showed:

Most predictive: year, depth, latitude, longitude, and rms.

Caution around magType due to label leakage.

➤ Model 2: Linear Regression
RMSE: ~0.50

R²: ~-0.02 (worse than baseline)

Linear models were not a good fit for this dataset.

➤ Model 3: Neural Network (MLPRegressor)
RMSE: ~0.44

R²: ~0.20

Performed worse than Random Forest, with underprediction of larger quakes.

✅ Takeaway:
Random Forest was the most effective supervised model.

Removing magType improved generalizability at the cost of some accuracy.

📊 Phase 3: Unsupervised Learning (unsupervised_model.ipynb)
Goal: Cluster earthquakes by geospatial and geophysical features.

➤ Preprocessing
Dropped mag and other non-predictive columns.

Scaled the dataset and applied K-Means clustering.

➤ Elbow Method
Optimal number of clusters determined to be 4.

➤ PCA + KMeans
Applied PCA for 2D visualization of clusters.

Plotted results to interpret spatial separation of events.

➤ Cluster Profiles
Cluster 1:

Shallowest depth (~4 km)

Lower depthError and horizontalError

Cluster 3:

Highest depthError, potentially imprecise events

All clusters had similar magnitude (~3.0)

➤ Visualizations
Created:

PCA scatter plot by cluster

Map using Folium with colored CircleMarkers and legend

Bar chart comparing average values by cluster

🧩 Tableau Dashboard Recommendations
To visualize your work, include:

Model Performance

Bar chart comparing RMSE/R² across models

Feature importance plot (Random Forest)

Unsupervised Insights

PCA 2D scatter of clusters

Folium screenshot or spatial cluster heatmap

Cluster profile bar chart (depth, mag, error metrics)

Context Panel

Summary text box on label leakage

Description of preprocessing logic

Interpretation of cluster geography and error precision