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
from typing import Dict, Any
import os
import json

logger = logging.getLogger(__name__)

def visualize_district_map(data: 'GeoDataFrame', assignment: np.ndarray) -> str:
    """
    Creates an interactive map of the redistricting plan using Folium and returns the HTML file path.

    Args:
        data (GeoDataFrame): The geospatial data with demographic and other attributes.
        assignment (np.ndarray): Array assigning each unit to a district.

    Returns:
        str: Path to the saved district map HTML file.
    """
    logger.info("Creating interactive map of the districting plan.")
    data = data.copy()
    data['district'] = assignment

    # Create a color palette
    num_districts = len(np.unique(assignment))
    colors = px.colors.qualitative.Plotly
    if num_districts > len(colors):
        # Extend the color palette if necessary
        colors = colors * (num_districts // len(colors) + 1)
    color_dict = {district: colors[district % len(colors)] for district in range(num_districts)}

    # Initialize Folium map
    m = folium.Map(location=[data.geometry.centroid.y.mean(), data.geometry.centroid.x.mean()], zoom_start=7)

    # Add districts to the map
    for district in range(num_districts):
        district_data = data[data['district'] == district]
        folium.GeoJson(
            district_data,
            style_function=lambda feature, color=color_dict[district]: {
                'fillColor': color,
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.6,
            },
            tooltip=folium.GeoJsonTooltip(fields=['GEOID'], aliases=['GEOID:'])
        ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Save the map
    map_filename = 'static/district_map.html'
    m.save(map_filename)
    logger.info(f"District map saved as '{map_filename}'.")
    return map_filename

def plot_fairness_metrics(fairness_metrics: Dict[str, float]) -> str:
    """
    Plots the fairness metrics using Plotly for interactive visualization and returns the HTML file path.

    Args:
        fairness_metrics (Dict[str, float]): Dictionary containing fairness metrics.

    Returns:
        str: Path to the saved fairness metrics HTML file.
    """
    logger.info("Plotting fairness metrics.")
    metrics = list(fairness_metrics.keys())
    values = list(fairness_metrics.values())

    fig = px.bar(
        x=values,
        y=metrics,
        orientation='h',
        labels={'x': 'Metric Value', 'y': 'Fairness Metrics'},
        title='Fairness Metrics Overview',
        text=values,
        color=metrics,
        color_discrete_sequence=px.colors.qualitative.Vivid
    )

    fig.update_traces(texttemplate='%{text:.4f}', textposition='inside')
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=150, r=50, t=50, b=50))

    fairness_metrics_filename = 'static/fairness_metrics.html'
    fig.write_html(fairness_metrics_filename, full_html=False, include_plotlyjs='cdn')
    logger.info(f"Fairness metrics plot saved as '{fairness_metrics_filename}'.")
    return fairness_metrics_filename

def visualize_district_characteristics(data: 'GeoDataFrame') -> Dict[str, str]:
    """
    Visualizes characteristics of the districts using Plotly for interactive plots and returns file paths.

    Args:
        data (GeoDataFrame): The geospatial data with demographic and other attributes.

    Returns:
        Dict[str, str]: Dictionary containing file paths of the saved visualizations.
    """
    logger.info("Visualizing district characteristics.")
    visualizations = {}

    # Population Distribution
    population_data = data.groupby('district')['P001001'].sum().reset_index()
    fig_pop = px.bar(
        population_data,
        x='district',
        y='P001001',
        labels={'district': 'District', 'P001001': 'Total Population'},
        title='Population Distribution Across Districts',
        color='district',
        color_continuous_scale='Viridis'
    )
    fig_pop.update_layout(margin=dict(l=50, r=50, t=50, b=50))
    pop_filename = 'static/district_populations.html'
    fig_pop.write_html(pop_filename, full_html=False, include_plotlyjs='cdn')
    visualizations['population_distribution'] = pop_filename
    logger.info(f"Population distribution plot saved as '{pop_filename}'.")

    # Median Income Distribution
    median_income_data = data.groupby('district')['median_income'].median().reset_index()
    fig_income = px.bar(
        median_income_data,
        x='district',
        y='median_income',
        labels={'district': 'District', 'median_income': 'Median Income'},
        title='Median Income Distribution Across Districts',
        color='median_income',
        color_continuous_scale='Blues'
    )
    fig_income.update_layout(margin=dict(l=50, r=50, t=50, b=50))
    income_filename = 'static/median_income_distribution.html'
    fig_income.write_html(income_filename, full_html=False, include_plotlyjs='cdn')
    visualizations['median_income_distribution'] = income_filename
    logger.info(f"Median income distribution plot saved as '{income_filename}'.")

    return visualizations

def visualize_trend_analysis(data: 'GeoDataFrame') -> str:
    """
    Creates visualizations to showcase trend analysis results using Plotly and returns the HTML file path.

    Args:
        data (GeoDataFrame): The geospatial data with demographic and trend attributes.

    Returns:
        str: Path to the saved population trends HTML file.
    """
    logger.info("Visualizing trend analysis results.")
    # Population Trend Mean by District
    trend_data = data.groupby('district')['population_trend_mean'].mean().reset_index()
    fig_trend = px.line(
        trend_data,
        x='district',
        y='population_trend_mean',
        labels={'district': 'District', 'population_trend_mean': 'Average Population Trend'},
        title='Average Population Trend by District',
        markers=True
    )
    fig_trend.update_layout(margin=dict(l=50, r=50, t=50, b=50))
    trend_filename = 'static/population_trends.html'
    fig_trend.write_html(trend_filename, full_html=False, include_plotlyjs='cdn')
    logger.info(f"Population trends plot saved as '{trend_filename}'.")
    return trend_filename

def generate_explainable_report(fairness_metrics: Dict[str, float], analysis_results: Dict[int, Dict[str, float]]) -> str:
    """
    Generates a user-friendly HTML report explaining the redistricting outcomes.

    Args:
        fairness_metrics (Dict[str, float]): Dictionary containing fairness metrics.
        analysis_results (Dict[int, Dict[str, float]]): Dictionary containing analysis results per district.

    Returns:
        str: Path to the saved report HTML file.
    """
    logger.info("Generating explainable report.")
    report_filename = 'static/redistricting_report.html'

    try:
        with open(report_filename, 'w') as report:
            report.write("<html><head><title>Redistricting Report</title></head><body>")
            report.write("<h1>Fairmandering Redistricting Report</h1>")
            
            # Fairness Metrics Section
            report.write("<h2>Fairness Metrics</h2>")
            report.write("<iframe src='fairness_metrics.html' width='100%' height='400px'></iframe>")
            
            # Analysis Results Section
            report.write("<h2>Analysis Results</h2>")
            for district, metrics in analysis_results.items():
                report.write(f"<h3>District {district}</h3><ul>")
                for metric, value in metrics.items():
                    report.write(f"<li><strong>{metric.replace('_', ' ').title()}:</strong> {value}</li>")
                report.write("</ul>")
            
            # Visualizations Section
            report.write("<h2>Visualizations</h2>")
            report.write("<ul>")
            report.write("<li><a href='district_map.html' target='_blank'>Interactive District Map</a></li>")
            report.write("<li><a href='district_populations.html' target='_blank'>Population Distribution</a></li>")
            report.write("<li><a href='median_income_distribution.html' target='_blank'>Median Income Distribution</a></li>")
            report.write("<li><a href='population_trends.html' target='_blank'>Population Trends</a></li>")
            report.write("</ul>")
            
            report.write("</body></html>")
        logger.info(f"Explainable report generated as '{report_filename}'.")
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        raise

    return report_filename

def generate_comparative_analysis_plot(plans_metrics: List[Dict[str, float]]) -> str:
    """
    Generates a comparative analysis plot for multiple districting plans using Plotly and returns the HTML file path.

    Args:
        plans_metrics (List[Dict[str, float]]): List of dictionaries containing fairness metrics for each plan.

    Returns:
        str: Path to the saved comparative analysis HTML file.
    """
    logger.info("Generating comparative analysis plot.")
    # Convert list of metrics to DataFrame
    df = pd.DataFrame(plans_metrics)
    df['Plan'] = [f'Plan {i+1}' for i in range(len(plans_metrics))]
    df = df.set_index('Plan')

    # Melt the DataFrame for Plotly
    df_melted = df.reset_index().melt(id_vars='Plan', var_name='Metric', value_name='Value')

    fig = px.bar(
        df_melted,
        x='Metric',
        y='Value',
        color='Plan',
        barmode='group',
        title='Comparative Analysis of Redistricting Plans',
        labels={'Value': 'Metric Value', 'Metric': 'Fairness Metrics'}
    )

    fig.update_layout(margin=dict(l=50, r=50, t=50, b=50), legend_title_text='Plans')

    comparative_filename = 'static/comparative_analysis.html'
    fig.write_html(comparative_filename, full_html=False, include_plotlyjs='cdn')
    logger.info(f"Comparative analysis plot saved as '{comparative_filename}'.")
    return comparative_filename
