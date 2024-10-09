# fairmandering/visualization.py

import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import plotly.express as px
import logging
from .config import Config
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def visualize_district_map(data, assignment):
    """
    Creates an interactive map of the redistricting plan.
    """
    logger.info("Creating interactive map of the districting plan.")
    data['district'] = assignment
    m = folium.Map(location=[data.geometry.centroid.y.mean(), data.geometry.centroid.x.mean()], zoom_start=7)
    folium.Choropleth(
        geo_data=data,
        name='Districts',
        data=data,
        columns=['GEOID', 'district'],
        key_on='feature.properties.GEOID',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Districts'
    ).add_to(m)
    folium.LayerControl().add_to(m)
    m.save('district_map.html')
    logger.info("District map saved as 'district_map.html'.")

def plot_fairness_metrics(fairness_metrics):
    """
    Plots the fairness metrics for comparison.
    """
    logger.info("Plotting fairness metrics.")
    metrics_df = pd.DataFrame.from_dict(fairness_metrics, orient='index', columns=['Value'])
    metrics_df.plot(kind='barh', legend=False)
    plt.xlabel('Metric Value')
    plt.title('Fairness Metrics')
    plt.tight_layout()
    plt.savefig('fairness_metrics.png')
    plt.close()
    logger.info("Fairness metrics plot saved as 'fairness_metrics.png'.")

def visualize_district_characteristics(data):
    """
    Visualizes characteristics of the districts.
    """
    logger.info("Visualizing district characteristics.")
    # Example: Plot population distribution across districts
    district_populations = data.groupby('district')['P001001'].sum()
    district_populations.plot(kind='bar')
    plt.xlabel('District')
    plt.ylabel('Total Population')
    plt.title('Population Distribution Across Districts')
    plt.tight_layout()
    plt.savefig('district_populations.png')
    plt.close()
    logger.info("District characteristics visualized and saved.")

def visualize_trend_analysis(data):
    """
    Creates visualizations to showcase trend analysis results.
    """
    logger.info("Visualizing trend analysis results.")
    # Example: Plot population trends
    population_trends = data.groupby('district')['population_trend'].mean()
    population_trends.plot(kind='bar')
    plt.xlabel('District')
    plt.ylabel('Population Trend')
    plt.title('Average Population Trend by District')
    plt.tight_layout()
    plt.savefig('population_trends.png')
    plt.close()
    logger.info("Trend analysis visualizations saved.")

def generate_explainable_report(fairness_metrics, analysis_results):
    """
    Generates a user-friendly report explaining the redistricting outcomes.
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
