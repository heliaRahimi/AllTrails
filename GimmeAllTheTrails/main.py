import pandas as pd
us_cities = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/us-cities-top-1k.csv")

import plotly.express as px
from blah.GimmeAllTheTrails.GimmeAllTheTrails.dataset import AllTrails
import click

@click.command()
@click.option('--csv_dir', default='', help='directory of csv files')
def run(csv_dir):
    llt = AllTrails(csv_dir).lat_lon_trail_id
    fig = px.scatter_mapbox(llt,
                            lat=llt.latitude,
                            lon=llt.longitude,
                            hover_name="trail_id",
                            zoom=1)

    fig.show()
