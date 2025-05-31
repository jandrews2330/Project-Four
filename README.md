# 🌎 Pacific Northwest Earthquake Prediction & Visualization 

This project visualizes and predicts seismic activity in the Pacific Northwest (PNW) using historical earthquake data, machine learning models, and interactive dashboards. It includes exploratory data analysis (EDA), supervised and unsupervised learning, and a Dash-based web application for public-facing insights.

## Project Objectives

- **Forecast Earthquake Occurrence** using a Random Forest Classifier  
- **Predict Earthquake Magnitude** with a Random Forest Regressor  
- **Identify Seismic Zones** using KMeans clustering on spatiotemporal features  
- **Visualize Results** through interactive maps and performance metrics  

```
## 📁 Project Structure

pnw-earthquake-project/
├── Resources/                      # Raw and cleaned datasets (pnw_final.csv)
├── models/                         # Saved ML models (.pkl files)
├── maps/                           # Static Folium HTML maps
├── assets/                         # Images, slide deck, and static media
├── pnw_dash_app.py                 # Main Dash app
├── EDA.ipynb                       # Exploratory data analysis
├── PNW_ETL.ipynb                   # ETL pipeline and preprocessing
├── initial_models.ipynb            # First round of supervised learning models
├── unsupervised_model.ipynb        # KMeans clustering and exploration
├── finalized_earthquake_notebook.ipynb # Final regression/classification models
├── app_instructions.md             # Basic usage instructions
├── create_earthquake_table.sql     # Table schema
├── earthquake_data.sqlite          # Sqlite output  
└── README.md                       # This file

```

## ⚙️ How It Works

### 🧼 1. Data Preparation 

- Loads raw USGS earthquake data
- Cleans and normalizes columns
- Adds time and spatial binning (e.g., `lat_bin`, `lon_bin`, `time_bin`)
- Saves final dataset to `Resources/pnw_final.csv`

### 📊 2. Data Exploration 

- Visualizes magnitude distribution, quake count by year/month
- Identifies geospatial trends and outliers
- Produces heatmaps and scatter plots for the dashboard

### 🧠 3. Machine Learning Models

- `initial_models.ipynb`: Trains and evaluates Random Forest for classification (quake/no quake)
- `finalized_earthquake_notebook.ipynb`: Refines models with temporal and aggregate features
- `unsupervised_model.ipynb`: Applies KMeans clustering to identify seismic risk zones

### 💻 4. Interactive Dashboard 

- Built with Dash + Plotly + Folium
- Includes 6 interactive tabs:
  - 🔍 **Overview**: Static summary charts
  - 📌 **Quake Maps**: Toggle between Folium and Plotly maps
  - 📊 **Model Indicators**: Feature importance and regression performance
  - 🔮 **ML Prediction**: Predicts quake probability and magnitude by date/location
  - 🗺️ **Residual Error Maps**: Visualizes ML model performance
  - 📝 **Slide Deck**: Built-in PDF presentation


## 📷 Sample Visuals

- Earthquake distribution over time
- Epicenter clustering and depth-magnitude overlays
- Feature importance bar charts for both models  
- Residual maps for both models error analysis  
- Scatter plots of earthquake magnitude predictions vs. actuals  
- Clustering results overlayed on maps

## Key Findings

- Earthquake **occurrence** is best predicted by **spatial location and timing**  
- Earthquake **magnitude** is more influenced by **prior seismic intensity**  
- Clustering revealed **distinct seismic hotspots** that align with fault zones

## Project Summary

Earthquake frequency has increased slightly in recent years, possibly due to improved detection or tectonic shifts. Most earthquakes in the PNW are of low to moderate magnitude (M < 4), with occasional spikes in higher magnitudes. Epicenters are concentrated along tectonic boundaries—especially the Cascadia Subduction Zone. KMeans clustering revealed distinct seismic zones, helping define risk-prone regions for targeted monitoring. 

Earthquake occurrence does not show strong seasonal trends, but monthly breakdowns support the creation of cyclical features (e.g., month_sin, month_cos) for modeling.Earthquake occurrence does not show strong seasonal trends, but monthly breakdowns support the creation of cyclical features (e.g., month_sin, month_cos) for modeling. However, aggregating historical quakes into temporal bins (e.g., time_bin) allowed richer features for prediction models. 

Random Forest achieved strong performance with accuracy and recall above baseline using features like location, historical quake count, and time bin. For magnitude, model captured general trends, but high residuals in outlier cases suggest limits in predicting extreme magnitudes. 

Error maps reveal some spatial clusters where prediction models consistently underperform—highlighting areas for future model refinement or data enhancement. Occurrence predictions have more consistent accuracy than magnitude estimates, which can be more volatile.

## 🧑‍🤝‍🧑 Team

- **Asres Dagnew**
- **Sofonias Abebe**
- **Jennefir Andrews**
- **Leonardo Rios**

---

## 📬 License / Credit

This project was created for educational purposes. If you use or modify this work, please credit the team above.
