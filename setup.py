import os
from setuptools import setup

setup(
    name="GimmeAllTheTrails",
    version="0.1",
    description="Visual Analytics final project - Visualizing hiking trails in NS",
    author="Razia, Helia and Noah",
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "beautifulsoup4==4.10.0",
        "bs4==0.0.1",
        "numpy==1.21.2",
        "pandas==1.3.3",
        "pymongo==3.12.0",
        "python-dateutil==2.8.2",
        "pytz==2021.1",
        "selenium==3.141.0",
        "six==1.16.0",
        "soupsieve==2.2.1",
        "urllib3==1.26.6",
        "plotly",
        "dash",
        "pyLDAvis",
        "textblob",
        "wordcloud",
        "dash_bootstrap_components",
        "dash_bootstrap_templates"
    ],
    entry_points={
        "console_scripts": ["GimmeAllTheTrails = GimmeAllTheTrails.main:run"]
    },
)
