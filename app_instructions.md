
# 🌎 Pacific Northwest Earthquake Dashboard

This Dash app visualizes and predicts seismic activity in the Pacific Northwest using historical earthquake data and machine learning models. It includes interactive maps, statistical insights, and model-driven predictions.

---

## 📦 Features

- **Overview Dashboard**: Visual summary of earthquakes by year, magnitude, and location.
- **Interactive Maps**:
  - 🌐 *Folium Earthquake Map*: Static marker-clustered map of historical quakes.
  - 🔶 *Plotly Cluster Map*: Dynamic scatter map of earthquakes colored by magnitude.
- **Model Indicator Charts**: Feature importance, correlation heatmap, and prediction accuracy.
- **ML Predictions**: Predicts future earthquake occurrence and magnitude using date and location.
- **Residual Error Maps**: Visualizes model residuals for magnitude and occurrence predictions.
- **Slide Deck**: Embedded PDF slide presentation for quick walkthroughs.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/jandrews2330/Project-Four
cd pnw-earthquake-dashboard
```

### 2. Set up the environment

It's recommended to use a virtual environment or `conda`.

```bash
conda create -n pnw_env python=3.10
conda activate pnw_env
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install dash pandas plotly folium branca dash-bootstrap-components scikit-learn joblib
```

---


## 🧠 How to Run the App

Make sure you're in the project directory and your environment is active.

```bash
python pnw_dash_app.py
```

Then open your browser to:

```
http://127.0.0.1:8050/
```

---

## 📝 Notes

- Make sure the `Resources/pnw_final.csv` file exists and is cleaned/formatted correctly.
- The app dynamically loads the Folium map (`maps/pnw_quakes_map.html`) and ML models from `models/`.
- Assets like images and the slide deck should be placed in the `assets/` folder so Dash can detect them automatically.
- If using VSCode or Jupyter, use `!python pnw_dash_app.py` or run through the terminal.

---


## 📌 Requirements

- Python 3.8+
- dash
- pandas
- plotly
- folium
- branca
- dash-bootstrap-components
- scikit-learn
- joblib

---

## 📬 Contact

Developed by Sofonias Abebe, Asres Dagnew, Jennefir Andrews, and Leonardo Rios.
 
If you use or modify this project, credit is appreciated!
