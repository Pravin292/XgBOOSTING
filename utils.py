import joblib
import json
import pandas as pd
import numpy as np
import plotly.express as px

def load_model_and_metrics():
    """Loads the serialized XGBoost model and performance metrics."""
    try:
        model = joblib.load("xgb_model.pkl")
        features = joblib.load("feature_columns.pkl")
        with open("metrics.json", "r") as f:
            metrics = json.load(f)
        return model, features, metrics
    except FileNotFoundError:
        return None, None, None

def generate_feature_importance_plot(metrics):
    """Generates a styled Plotly chart for XGBoost Feature Importance."""
    fi = metrics.get('feature_importance', {})
    if not fi:
        return None
        
    df_fi = pd.DataFrame({
        'Feature': list(fi.keys()),
        'Importance': list(fi.values())
    }).sort_values(by='Importance', ascending=True) # Ascending for horizontal bar chart

    fig = px.bar(
        df_fi, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        title='XGBoost Feature F-Scores (Importance)'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        title_font=dict(size=20, color='#38bdf8'),
        xaxis=dict(title="Relative Importance Weight", gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(title="")
    )
    fig.update_traces(marker_color='#8b5cf6')
    
    return fig
