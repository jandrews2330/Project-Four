# === Import Required Libraries ===
import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
import folium
from folium.plugins import MarkerCluster
from branca.element import Element
import os
import joblib
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans

# Load and preprocess data
df = pd.read_csv("Resources/pnw_final.csv")
df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
df = df[df['time'].notna()]

# Extract date/time features for modeling and visualization
df['year'] = df['time'].dt.year
df['month'] = df['time'].dt.month
df['hour'] = df['time'].dt.hour
df['time_bin'] = df['time'].dt.to_period('M').astype(str)
df['lat_bin'] = (df['latitude'] // 0.5) * 0.5
df['lon_bin'] = (df['longitude'] // 0.5) * 0.5
df['time_bin_num'] = df['time_bin'].str.replace('-', '').astype(int)

# Apply clustering for Plotly map
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(df[['latitude', 'longitude']])

# Load pre-trained classifier and regressor models
clf = joblib.load("models/quake_classifier.pkl")
reg = joblib.load("models/quake_regressor.pkl")

# Create historical aggregate features for predictive models
agg = df.groupby(['lat_bin', 'lon_bin', 'time_bin']).agg(
    count_prev=('mag', 'count'),
    max_mag_prev=('mag', 'max')
).reset_index()
agg['time_bin_num'] = agg['time_bin'].str.replace('-', '').astype(int)

# Merge original and aggregate features
merged = pd.merge(df, agg, on=['lat_bin', 'lon_bin', 'time_bin'], how='left')
merged['month_sin'] = np.sin(2 * np.pi * merged['month'] / 12)
merged['month_cos'] = np.cos(2 * np.pi * merged['month'] / 12)
merged['time_bin_num'] = merged['time_bin'].str.replace('-', '').astype(int)

# Prepare input and output for residual map calculation
X_reg = merged[reg.feature_names_in_].copy()
y_reg = merged['mag']
y_pred_reg = reg.predict(X_reg)

# Calculate residuals and error metrics for map display
residual_map = X_reg.copy()
residual_map['actual'] = y_reg.values
residual_map['predicted'] = y_pred_reg
residual_map['residual'] = residual_map['predicted'] - residual_map['actual']
residual_map['abs_error'] = residual_map['residual'].abs()
residual_map['lat'] = merged['lat_bin']
residual_map['lon'] = merged['lon_bin']
residual_map['hover'] = (
    "Lat: " + residual_map['lat'].astype(str) +
    "<br>Lon: " + residual_map['lon'].astype(str) +
    "<br>Actual Mag: " + residual_map['actual'].round(2).astype(str) +
    "<br>Predicted Mag: " + residual_map['predicted'].round(2).astype(str) +
    "<br>Residual: " + residual_map['residual'].round(2).astype(str)
)

# Plot residual error map for magnitude prediction
fig_residual_1 = px.scatter_mapbox(
    residual_map,
    lat="lat",
    lon="lon",
    color="abs_error",
    size="abs_error",
    color_continuous_scale="RdYlGn_r",
    size_max=12,
    zoom=5,
    height=700,
    mapbox_style="carto-positron",
    title="Earthquake Magnitude Prediction Error Map"
)

# Plot residual error map for occurrence prediction
fig_residual_2 = px.scatter_mapbox(
    residual_map,
    lat="lat",
    lon="lon",
    color="residual",
    size="abs_error",
    color_continuous_scale="RdBu",
    size_max=12,
    zoom=5,
    height=700,
    mapbox_style="carto-positron",
    title="Earthquake Occurrence Prediction Error Map"
)

# # Generate and save the original marker-based Folium map
os.makedirs("maps", exist_ok=True)
m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=6)
marker_cluster = MarkerCluster().add_to(m)
for _, row in df.iterrows():
    popup = f"<b>Location:</b> {row['place']}<br><b>Mag:</b> {row['mag']}<br><b>Depth:</b> {row['depth']} km"
    folium.CircleMarker(
        location=(row['latitude'], row['longitude']),
        radius=3 + row['mag'],
        color='red' if row['mag'] >= 4 else 'blue',
        fill=True, fill_opacity=0.6, popup=popup
    ).add_to(marker_cluster)
m.save("maps/pnw_quakes_map.html")

# # Create Plotly map showing clusters colored by magnitude
quake_features = df.copy()
fig_cluster_map = px.scatter_mapbox(
    quake_features,
    lat="latitude",
    lon="longitude",
    color="mag",
    size="mag",
    color_continuous_scale="RdYlGn_r",
    hover_data={"depth": True, "mag": True, "cluster": True},
    title="Earthquake Clusters Colored by Magnitude (Green = Low, Red = High)",
    zoom=5,
    height=700
)
fig_cluster_map.update_layout(
    mapbox_style="carto-positron",
    margin={"r": 0, "t": 40, "l": 0, "b": 0}
)

# Set up the Dash app layout and theming
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.PULSE], suppress_callback_exceptions=True)
app.title = "PNW Earthquake Dashboard"

# Define layout with tabs for maps, stats, predictions, residuals
app.layout = dbc.Container([
    html.H1("🌎 Pacific Northwest Earthquake Dashboard", className="text-center my-4"),

    dcc.Tabs([
        # Overview tab with static PNGs
        dcc.Tab(label='🔍 Overview', children=[
            html.Div([
                html.Img(src="/assets/earthquakes_per_year.png"),
                html.Img(src="/assets/magnitude_distribution.png"),
                html.Img(src="/assets/epicenter_scatter.png"),
                html.Img(src="/assets/correlation_heatmap.png")
            ], className="p-4")
        ]),

        # Map tab with toggle between Folium and Plotly
        dcc.Tab(label='📌 Quake Maps', children=[
            html.Div([
                html.Div([
                    dcc.RadioItems(
                        id='folium-map-toggle',
                        options=[
                            {'label': 'Earthquake Map (Folium)', 'value': 'folium'},
                            {'label': 'Cluster Map (Plotly)', 'value': 'plotly'}
                        ],
                        value='folium',
                        labelStyle={'display': 'inline-block', 'margin-right': '20px'}
                    )
                ], style={'textAlign': 'center', 'margin': '10px'}),

                html.Div(id='map-container')
            ])
        ]),

        # Static charts and model insights
        dcc.Tab(label='📊 Model Indicators', children=[
            html.Div([
                html.Img(src="/assets/feature_importance_occ_pred.png"),
                html.Img(src="/assets/feature_importance_mag_pred.png"),
                html.Img(src="/assets/predicted_vs_actual_regression.png")
            ], className="p-4")
        ]),

        # ML Prediction interface
        dcc.Tab(label='🔮 ML Prediction', children=[
            html.Div([
                html.H5("Predict Earthquake Occurrence and Magnitude"),
                dbc.Row([
                    dbc.Col(dcc.Dropdown(
                        id='input-location',
                        options=[
                            {'label': f"Lat {lat}, Lon {lon}", 'value': f"{lat},{lon}"}
                            for lat, lon in sorted(set(zip(df['lat_bin'], df['lon_bin'])))
                        ],
                        placeholder='Select PNW Location'
                    )),
                    dbc.Col(dcc.DatePickerSingle(
                        id='input-date',
                        placeholder='Select Date',
                        date='2015-08-01'
                    )),
                    dbc.Col(html.Button('Predict', id='predict-btn', n_clicks=0))
                ], className='my-2'),
                html.Div(id='prediction-output')
            ], className="m-4")
        ]),

         # Residual error maps (toggleable)
        dcc.Tab(label='🗺️ Residual Error Maps', children=[
            html.Div([
                html.Div([
                    dcc.RadioItems(
                        id='residual-map-toggle',
                        options=[
                            {'label': 'Magnitude Prediction', 'value': 'map1'},
                            {'label': 'Occurrence Prediction', 'value': 'map2'}
                        ],
                        value='map1',
                        labelStyle={'display': 'inline-block', 'margin-right': '20px'}
                    )
                ], style={'textAlign': 'center'}),
                dcc.Graph(id='residual-map-display')
            ], className="p-4")
        ]),

        # Slide deck presentation (PDF iframe)
        dcc.Tab(label='📝 Slide Deck', children=[
            html.Div([
                html.H5("Slide Deck Presentation", className="mt-3"),
                html.Iframe(
                    src="/assets/placeholder_slide_deck.pdf",
                    style={"width": "100%", "height": "800px", "border": "none"}
                )
            ], className="p-4")
        ])
    ])
], fluid=True)

# App callbacks 
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('input-location', 'value'),
    State('input-date', 'date')
)
def predict_earthquake(n_clicks, location, date):
    if not location or not date:
        return "Please select both location and date."

    try:
        lat, lon = map(float, location.split(','))
        dt = pd.to_datetime(date)
        month = dt.month
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        time_bin_num = int(dt.to_period("M").strftime('%Y%m'))
        lat_bin = np.floor(lat * 2) / 2
        lon_bin = np.floor(lon * 2) / 2

        prev_bin = (dt.to_period("M") - 1).strftime('%Y-%m')
        subset = df[(df['time_bin'] == prev_bin) &
                    (df['lat_bin'] == lat_bin) &
                    (df['lon_bin'] == lon_bin)]
        count_prev = subset.shape[0]
        max_mag_prev = subset['mag'].max() if not subset.empty else 0

        X_pred = pd.DataFrame([{
            'lat_bin': lat_bin,
            'lon_bin': lon_bin,
            'month_sin': month_sin,
            'month_cos': month_cos,
            'count_prev': count_prev,
            'max_mag_prev': max_mag_prev,
            'time_bin_num': time_bin_num
        }])

        X_pred_clf = X_pred[clf.feature_names_in_]
        X_pred_reg = X_pred[reg.feature_names_in_]

        prob_quake = clf.predict_proba(X_pred_clf)[0][1]
        mag_pred = reg.predict(X_pred_reg)[0]

        return html.Div([
            html.H6(f"Predicted Earthquake Probability: {prob_quake:.2%}"),
            html.H6(f"Predicted Maximum Magnitude: {mag_pred:.2f}")
        ])

    except Exception as e:
        return f"Prediction error: {str(e)}"

@app.callback(
    Output('residual-map-display', 'figure'),
    Input('residual-map-toggle', 'value')
)
def toggle_residual_map(selected):
    return fig_residual_2 if selected == 'map2' else fig_residual_1

@app.callback(
    Output('map-container', 'children'),
    Input('folium-map-toggle', 'value')
)
def update_map_view(map_choice):
    if map_choice == 'folium':
        return html.Iframe(
            srcDoc=open("maps/pnw_quakes_map.html", "r").read(),
            width='100%', height='800'
        )
    else:
        return dcc.Graph(figure=fig_cluster_map)

if __name__ == '__main__':
    app.run(debug=True)
