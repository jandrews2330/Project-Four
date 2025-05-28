import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
import folium
from folium.plugins import MarkerCluster
import os
import joblib
import numpy as np
from datetime import datetime

# Load data
df = pd.read_csv("Resources/pnw_final.csv")
df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
df['time_bin'] = df['time'].dt.to_period('M').astype(str)
df['lat_bin'] = (df['latitude'] // 0.5) * 0.5
df['lon_bin'] = (df['longitude'] // 0.5) * 0.5

# Load models
clf = joblib.load("models/quake_classifier.pkl")
reg = joblib.load("models/quake_regressor.pkl")

# Generate and save Folium map to HTML
def create_folium_map(dataframe, output_path="maps/pnw_quakes_map.html"):
    os.makedirs("maps", exist_ok=True)
    m = folium.Map(location=[dataframe['latitude'].mean(), dataframe['longitude'].mean()], zoom_start=6)
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in dataframe.iterrows():
        popup = f"<b>Location:</b> {row['place']}<br><b>Mag:</b> {row['mag']}<br><b>Depth:</b> {row['depth']} km"
        folium.CircleMarker(
            location=(row['latitude'], row['longitude']),
            radius=3 + row['mag'],
            color='red' if row['mag'] >= 4 else 'blue',
            fill=True,
            fill_opacity=0.6,
            popup=popup
        ).add_to(marker_cluster)

    m.save(output_path)

# Create the map
create_folium_map(df)

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.PULSE], suppress_callback_exceptions=True)
app.title = "PNW Earthquake Dashboard"

# Layout
app.layout = dbc.Container([
    html.H1("\U0001F30EPacific Northwest Earthquake Dashboard", className="text-center my-4"),

    dcc.Tabs([        
        dcc.Tab(label='\U0001F50DOverview', children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=px.histogram(df, x='year', title="Earthquakes per Year")), width=12),
                dbc.Col(dcc.Graph(figure=px.histogram(df, x='mag', nbins=30, title="Magnitude Distribution")), width=12),
                dbc.Col(dcc.Graph(figure=px.histogram(df, x='depth', nbins=30, title="Depth Distribution")), width=12),
            ], className="gx-4")
        ]),

        dcc.Tab(label='\U0001F4CCFolium Map', children=[
            html.Div([
                html.Iframe(
                    id='folium-map',
                    srcDoc=open("maps/pnw_quakes_map.html", "r").read(),
                    width='100%',
                    height='800'
                )
            ], style={"padding": "0", "margin": "0"})
        ]),

        dcc.Tab(label='\U0001F4CAStatistics', children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=px.box(df, x='year', y='mag', title="Magnitude by Year")), width=12),
                dbc.Col(dcc.Graph(figure=px.box(df, x='month', y='depth', title="Depth by Month")), width=12),
            ], className="gx-4")
        ]),

        dcc.Tab(label='\U0001F52EML Prediction', children=[
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
                    dbc.Col(dcc.DatePickerSingle(id='input-date', placeholder='Select Date')),
                    dbc.Col(html.Button('Auto-Fill Prev Stats', id='fill-btn', n_clicks=0)),
                    dbc.Col(dcc.Input(id='input-prev-count', type='number', placeholder='Prev Quake Count')),
                    dbc.Col(dcc.Input(id='input-prev-mag', type='number', placeholder='Prev Max Mag')),
                    dbc.Col(html.Button('Predict', id='predict-btn', n_clicks=0))
                ], className='my-2'),
                html.Div(id='prediction-output')
            ], className="m-4")
        ])
    ])
], fluid=True, style={"maxWidth": "100vw", "padding": "0", "margin": "0"})

# Auto-fill callback (corrected)
@app.callback(
    [Output('input-prev-count', 'value'),
     Output('input-prev-mag', 'value')],
    [Input('fill-btn', 'n_clicks')],
    [State('input-location', 'value'),
     State('input-date', 'date')]
)
def autofill_prev_stats(n_clicks, location, date):
    if None in [location, date]:
        return None, None

    lat, lon = map(float, location.split(','))
    prev_bin = (pd.to_datetime(date).to_period('M') - 1).strftime('%Y-%m')

    lat_bin = np.floor(lat * 2) / 2
    lon_bin = np.floor(lon * 2) / 2

    subset = df[(df['time_bin'] == prev_bin) &
                (df['lat_bin'] == lat_bin) &
                (df['lon_bin'] == lon_bin)]

    if subset.empty:
        return None, None

    return int(subset.shape[0]), float(subset['mag'].max())

# Run server
if __name__ == '__main__':
    app.run(debug=True)
