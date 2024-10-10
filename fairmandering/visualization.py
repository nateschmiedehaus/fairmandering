# fairmandering/visualization.py

import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import plotly.express as px
import plotly.graph_objects as go
import logging
from .config import Config
import numpy as np
import pandas as pd
import json

logger = logging.getLogger(__name__)

def visualize_district_map(data, assignment):
    """
    Creates an interactive map of the redistricting plan using Plotly.

    Args:
        data (GeoDataFrame): Geospatial data with district assignments.
        assignment (np.ndarray): Array assigning each unit to a district.

    Returns:
        str: JSON representation of the Plotly figure.
    """
    logger.info("Creating interactive map of the districting plan using Plotly.")
    data = data.copy()
    data['district'] = assignment

    fig = px.choropleth_mapbox(
        data,
        geojson=data.geometry.__geo_interface__,
        locations=data.index,
        color='district',
        color_continuous_scale='Viridis',
        mapbox_style="carto-positron",
        zoom=7,
        center={"lat": data.geometry.centroid.y.mean(), "lon": data.geometry.centroid.x.mean()},
        opacity=0.5,
        labels={'district': 'District'}
    )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    # Convert Plotly figure to JSON for embedding in HTML
    map_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    fig.write_html("district_map.html")
    logger.info("District map created and saved as 'district_map.html'.")
    return map_json

def plot_fairness_metrics(fairness_metrics):
    """
    Creates an interactive bar chart of fairness metrics using Plotly.

    Args:
        fairness_metrics (dict): Dictionary of fairness metrics.

    Returns:
        str: JSON representation of the Plotly figure.
    """
    logger.info("Creating interactive bar chart for fairness metrics using Plotly.")
    metrics = list(fairness_metrics.keys())
    values = list(fairness_metrics.values())

    fig = go.Figure(data=[go.Bar(x=metrics, y=values, marker_color='indianred')])
    fig.update_layout(
        title='Fairness Metrics',
        xaxis_title='Metrics',
        yaxis_title='Values',
        template='plotly_white'
    )

    metrics_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    fig.write_html("fairness_metrics.html")
    logger.info("Fairness metrics plot created and saved as 'fairness_metrics.html'.")
    return metrics_json

def visualize_district_characteristics(data):
    """
    Creates interactive visualizations for district characteristics using Plotly.

    Args:
        data (GeoDataFrame): Geospatial data with district assignments.

    Returns:
        dict: JSON representations of the generated figures.
    """
    logger.info("Creating interactive visualizations for district characteristics using Plotly.")
    # Population Distribution
    district_populations = data.groupby('district')['P001001'].sum().reset_index()
    fig_pop = px.bar(district_populations, x='district', y='P001001',
                     labels={'P001001': 'Total Population', 'district': 'District'},
                     title='Population Distribution Across Districts',
                     color='P001001', color_continuous_scale='Blues')
    pop_json = json.dumps(fig_pop, cls=plotly.utils.PlotlyJSONEncoder)
    fig_pop.write_html("district_populations.html")

    # Minority Representation
    minority_populations = data.groupby('district')['P005004'].sum().reset_index()
    total_minority_population = data['P005004'].sum()
    minority_populations['Representation (%)'] = (minority_populations['P005004'] / total_minority_population) * 100
    fig_min = px.bar(minority_populations, x='district', y='Representation (%)',
                    labels={'Representation (%)': 'Minority Representation (%)', 'district': 'District'},
                    title='Minority Representation Across Districts',
                    color='Representation (%)', color_continuous_scale='Greens')
    min_json = json.dumps(fig_min, cls=plotly.utils.PlotlyJSONEncoder)
    fig_min.write_html("minority_representation.html")

    # Political Fairness
    party_a_votes = data.groupby('district')['votes_party_a'].sum().reset_index()
    party_b_votes = data.groupby('district')['votes_party_b'].sum().reset_index()
    merged_votes = pd.merge(party_a_votes, party_b_votes, on='district')
    merged_votes['Vote Share Party A (%)'] = (merged_votes['votes_party_a'] / (merged_votes['votes_party_a'] + merged_votes['votes_party_b'])) * 100
    merged_votes['Vote Share Party B (%)'] = 100 - merged_votes['Vote Share Party A (%)']

    fig_pol = go.Figure()
    fig_pol.add_trace(go.Bar(
        x=merged_votes['district'],
        y=merged_votes['Vote Share Party A (%)'],
        name='Party A',
        marker_color='blue'
    ))
    fig_pol.add_trace(go.Bar(
        x=merged_votes['district'],
        y=merged_votes['Vote Share Party B (%)'],
        name='Party B',
        marker_color='red'
    ))
    fig_pol.update_layout(
        barmode='stack',
        title='Political Fairness Across Districts',
        xaxis_title='District',
        yaxis_title='Vote Share (%)',
        template='plotly_white'
    )
    pol_json = json.dumps(fig_pol, cls=plotly.utils.PlotlyJSONEncoder)
    fig_pol.write_html("political_fairness.html")

    return {
        'population_distribution': pop_json,
        'minority_representation': min_json,
        'political_fairness': pol_json
    }

def visualize_trend_analysis(data):
    """
    Creates interactive trend analysis visualizations using Plotly.

    Args:
        data (GeoDataFrame): Geospatial data with district assignments.

    Returns:
        str: JSON representation of the Plotly figure.
    """
    logger.info("Creating interactive trend analysis visualizations using Plotly.")
    # Example: Population Trends over the TREND_YEARS
    trend_data = {}
    for year in Config.TREND_YEARS:
        year_column = f'population_trend_{year}'
        if year_column in data.columns:
            trend_data[year] = data.groupby('district')[year_column].sum().reset_index()
        else:
            logger.warning(f"Trend data for year {year} not found in data columns.")

    fig_trend = go.Figure()
    for year, df in trend_data.items():
        fig_trend.add_trace(go.Scatter(
            x=df['district'],
            y=df[f'population_trend_{year}'],
            mode='lines+markers',
            name=str(year)
        ))

    fig_trend.update_layout(
        title='Population Trends Over Years',
        xaxis_title='District',
        yaxis_title='Population Trend',
        template='plotly_white'
    )

    trend_json = json.dumps(fig_trend, cls=plotly.utils.PlotlyJSONEncoder)
    fig_trend.write_html("population_trends.html")
    logger.info("Trend analysis visualizations created and saved as 'population_trends.html'.")
    return trend_json

def generate_explainable_report(fairness_metrics, analysis_results):
    """
    Generates a user-friendly report explaining the redistricting outcomes.

    Args:
        fairness_metrics (dict): Fairness metrics of the plan.
        analysis_results (dict): Detailed analysis results per district.

    Returns:
        None
    """
    logger.info("Generating explainable report.")
    with open('redistricting_report.txt', 'w') as report:
        report.write("Fairmandering Redistricting Report\n")
        report.write("===============================\n\n")
        report.write("Fairness Metrics:\n")
        for metric, value in fairness_metrics.items():
            report.write(f"- {metric.replace('_', ' ').title()}: {value:.4f}\n")
        report.write("\nAnalysis Results:\n")
        for key, value in analysis_results.items():
            report.write(f"- {key.replace('_', ' ').title()}: {value}\n")
    logger.info("Explainable report generated as 'redistricting_report.txt'.")
