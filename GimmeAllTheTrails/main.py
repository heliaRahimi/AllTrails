import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

import click
from GimmeAllTheTrails.plotly_components import mapbox
@click.command()
@click.option('--csv_dir', default='', help='directory of csv files')
def run(csv_dir):
    app = dash.Dash(__name__,
                    external_stylesheets=[dbc.themes.CYBORG])

    app.layout = dbc.Container([html.H1("Our Title", style={'text-align': 'center'}),
                       dbc.Row(
                           children=[dcc.Graph(id='mapbox',
                                                       figure=mapbox(csv_dir))],
                                                       width=12)]
                            )
    app.run_server(debug=True)

if __name__ == "__main__":
    run(csv_dir=r"C:\Users\NoahB\Desktop\School\first year MCSC (2021-2022)\CS6612\group_proj\GimmeAllTheTrails\data\csv")