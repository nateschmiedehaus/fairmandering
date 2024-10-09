# setup.py

from setuptools import setup, find_packages

setup(
    name='fairmandering',
    version='1.0.0',
    author='Your Name',
    author_email='youremail@example.com',
    description='A Fair Redistricting System',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'geopandas',
        'numpy',
        'scipy',
        'requests',
        'census',
        'pymoo',
        'matplotlib',
        'seaborn',
        'folium',
        'python-dotenv',
        'joblib',
        'scikit-learn',
        'cryptography',
        'us',
        'plotly',
        'redis',
        'Flask'
    ],
    entry_points={
        'console_scripts': [
            'fairmandering = fairmandering.main:main',
        ],
    },
)
