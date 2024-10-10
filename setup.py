from setuptools import setup, find_packages

setup(
    name='fairmandering',
    version='1.0.0',
    author='Nathaniel Schmiedehaus',
    author_email='nate@schmiedehaus.com',
    description='A Fair Redistricting System',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    python_requires='>=3.8',
    packages=find_packages(),
    install_requires=[
        'Here's the final file for the setup:

```python
from setuptools import setup, find_packages

setup(
    name='fairmandering',
    version='1.0.0',
    author='Nathaniel Schmiedehaus',
    author_email='nate@schmiedehaus.com',
    description='A Fair Redistricting System',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    python_requires='>=3.8',
    packages=find_packages(),
    install_requires=[
        'pandas>=1.3,<2.0',
        'geopandas>=0.9,<1.0',
        'numpy>=1.21,<2.0',
        'scipy>=1.7,<2.0',
        'requests>=2.25,<3.0',
        'census==0.8.3',
        'pymoo==0.6.1.3',  # Ensure compatibility
        'matplotlib>=3.4,<4.0',
        'seaborn>=0.11,<1.0',
        'folium>=0.12,<1.0',
        'python-dotenv>=0.19,<1.0',
        'joblib>=1.0,<2.0',
        'scikit-learn>=0.24,<1.0',
        'cryptography>=3.4,<4.0',
        'us>=2.0,<3.0',
        'plotly>=5.0,<6.0',
        'redis>=3.5,<5.0',
        'Flask>=2.0,<3.0',
        'tailwindcss'  # Ensure this works in the environment
    ],
    entry_points={
        'console_scripts': [
            'fairmandering = fairmandering.main:main',
        ],
    },
)
