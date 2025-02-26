# setup.py
from setuptools import setup, find_packages

setup(
    name='bias_copilot',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'aif360',
        'pandas',
        'numpy',
        'click',
        'scikit-learn',
        'ethnicolr'
    ],
    entry_points={
        'console_scripts': [
            'bias_copilot = bias_copilot.cli:cli'
        ]
    }
)