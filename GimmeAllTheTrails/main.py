import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
# from dash.dependencies import Output, Input
# from dash import no_update
from dash import Dash, dcc, html, Input, Output, no_update, State
from dash_bootstrap_components._components.Container import Container
import plotly.express as px
import plotly.graph_objects as go
from GimmeAllTheTrails.dataset import AllTrails

# FIGURES #
def mapbox(map_data):
    fig = go.Figure(go.Scattermapbox(
        lat=map_data.latitude.tolist(),
        lon=map_data.longitude.tolist(),
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=9
        ),

        text=map_data.trail_id.tolist()
    )
    )
    # https://plotly.github.io/plotly.py-docs/generated/plotly.express.scatter_mapbox.html
    fig.update_layout(
        autosize=True,
        geo=dict(bgcolor='rgba(0,0,0,1)'),
        hovermode='closest',
        mapbox_style='carto-darkmatter',
        mapbox=dict(
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=45,
                lon=-63
            ),
            pitch=0,
            zoom=6.5
        ),
        margin=go.layout.Margin(
                        l=0,
                        r=0,
                        b=0,
                        t=0),
        clickmode='event+select',
    )
    return fig



# APP #
# @click.command()
# @click.option('--csv_dir', default='', help='directory of csv files')
def run(csv_dir):
    load_figure_template("bootstrap")
    # init app #
    app = dash.Dash(__name__,
                    external_stylesheets=[dbc.themes.CYBORG])
    # app.css.append_css(
    #     { "external_url": "stylesheets/graph.css" }
    # )

    # GLOBALS #
    # load data #
    data = AllTrails(csv_dir)

    main_map = mapbox(data.main_map_data)
    # COMPONENTS #
    # NAVBAR #
    # PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"
    # search_bar = dbc.Row(
    #     [
    #         dbc.Col(dbc.Input(type="search", placeholder="Filter", style={"color":"white"})),
    #         dbc.Col(
    #             dbc.Button(
    #                 "Filter", color="primary", className="ms-2", n_clicks=0
    #             ),
    #             width="auto",
    #         ),
    #     ],
    #     className="g-0 ms-auto flex-nowrap mt-3 mt-md-0",
    #     align="center",
    # )

    # navbar = dbc.Navbar(
    #     dbc.Container(
    #         [
    #             html.A(
    #                 # Use row and col to control vertical alignment of logo / brand
    #                 dbc.Row(
    #                     [
    #                         dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
    #                         dbc.Col(dbc.NavbarBrand("Navbar", className="ms-2")),
    #                     ],
    #                     align="center",
    #                     className="g-0",
    #                 ),
    #                 href="https://plotly.com",
    #                 style={ "textDecoration": "none" },
    #             ),
    #             dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
    #             dbc.Collapse(
    #                 search_bar,
    #                 id="navbar-collapse",
    #                 is_open=False,
    #                 navbar=True,
    #             ),
    #         ]
    #     ),
    #     color="dark",
    #     dark=True,
    # )

    # TRAIL CARD POPUP #
    # mini map view #
    mini_map = dbc.Col(
        html.Div(
            [
                        dbc.Col(dcc.Graph(figure={},
                                      config={
                                          'displayModeBar': False,
                                          'staticPlot': True
                                      },
                                      style={ 'width': '100%', 'height': '100%', 'display': 'flex'
                                              },
                                        id="mini-map"
                                      ),
                                style={ 'width': '100%', 'height': '100%'},
                        ),

                        dbc.Col(html.Div("One of three columns", style={ 'width': '100%', 'height': '50%'})),
                    ],
                    style = { 'width': '100%', 'height': '75%' }
        ),
        md=6,
        className="h-100 p-5 bg-light border rounded-3",
        style={ "height": "50%", "width":"33vw", "padding-top":"1.5vh", "padding-right":"0.5vh"  }
    )
    # description of trail #
    trail_description = dbc.Col(
        html.Div(
            children=[],
            className="h-100 p-5 bg-light border rounded-3",
        ),
        md=6,
        style={ "height": "50%", "width":"33vw", "padding-top":"1.5vh", "padding-left":"0.5vh" },
        id="trail-card"
    )
    # modal container for trail view #
    trail_view = html.Div(
        [
            dbc.Modal(
                [
                    dbc.ModalHeader(children=[], id="modal-header"),
                    dbc.ModalBody([dbc.Row(
                                    [mini_map, trail_description],
                                    className="align-items-md-stretch"
                                    )]),
                ],
                id="modal",
                fullscreen=True,
                is_open=False
            )
        ],
    )

    # Filter bar on left #
    filter_type = html.Div(
        [
            dbc.Label("Trail Type", html_for="dropdown"),
            dcc.Dropdown(
                id="filter_type",
                # in meta reviews
                options=[
                    { "label": t_type, "value": t_type } for t_type in data.main_map_data["type"].unique()
                ] + [{ "label": "all", "value": "all" }],
                value="all"
            ),
        ],
        className="mb-3",
    )

    filter_rating = html.Div(
        [
            dbc.Label("Star Rating", html_for="slider"),
            dcc.Slider(id="filter_rating", min=0, max=5, step=1, value=3),
        ],
        className="mb-3",
    )

    filter_length = html.Div(
        [
            dbc.Label("Trail Length", html_for="range-slider"),
            # set min and max to the length column
            dcc.RangeSlider(id="filter_length", min=data.main_map_data["length"].min(),
                            max=data.main_map_data["length"].max(), value=[0, 50]),
        ],
        className="mb-3",
    )
    filter_elev = html.Div(
        [
            dbc.Label("Trail Elevation", html_for="range-slider"),
            dcc.RangeSlider(id="filter_elev", min=data.main_map_data["elevation"].min(),
                            max=data.main_map_data["elevation"].max(), value=[0, 10000]),
        ],
        className="mb-3",
    )
    # final filter form #
    filter_form = dbc.Form([filter_type, filter_rating, filter_length,  filter_elev])
    filters = dbc.Card(
        dbc.CardBody(
            [
                html.H4("Filters", className="card-title"),
                # html.H6("Card subtitle", className="card-subtitle"),
                filter_form, dbc.Col(dbc.Button("Submit", color="primary", id="filter_submit"), width="auto"),
            ]
        ),
        style={ "height": "80vh", "width": "18vw",
                "padding-top": "5vh",
                "padding-left": "0.5vh" ,
                "position": "absolute",
                "z-index": "2",
                "opacity":"0.8"},
    )
    # list of selected trails #
    trail_list = dbc.Card(
        dbc.CardBody(
            [
                html.H4("Compare Trails", className="card-title"),
                html.H6("Card subtitle", className="card-subtitle"),
                html.P(
                    "Some quick example text to build on the card title and make "
                    "up the bulk of the card's content.",
                    className="card-text",
                ),
                dbc.CardLink("Card link", href="#"),
                dbc.CardLink("External link", href="https://google.com"),
            ]
        ),
        style={ "height": "60vh", "width": "20vw",
                "padding-top": "5vh",
                "padding-left": "0.5vh",
                 "position": "absolute",
                  "bottom": "25px",
                  "right": "25px",
                "z-index": "2",
                "opacity": "0.8" },
    )
    main_map_graph = html.Div([dcc.Graph(figure={},
                          id='main-map',
                          config={
                              'displayModeBar': False
                          },
                          clear_on_unhover = True,
                          style={ 'width': '100vw', 'height': '100vh', 'display': 'flex',
                                  "position": "absolute", "z-index": "1"}
                  ),
            dcc.Tooltip(id="main-map-hover",style={"position": "absolute", "z-index": "2"})])
    main = html.Div(
        [
            # navbar,
            trail_view,
            main_map_graph,
            filters,
            trail_list
        ],
        style={ 'width': '100%', 'height': '100%' })


    app.layout = main


    # FUNCTIONALITY

    @app.callback(
        [Output("main-map-hover", "show"),
         Output("main-map-hover", "bbox"),
         Output("main-map-hover", "children")],
        [Input("main-map", "hoverData")]
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update

        # demo only shows the first point, but other points may also be available
        pt = hoverData["points"][0]
        bbox = pt["bbox"]

        num = pt["pointNumber"]
        print(pt, bbox, num)
        # df_row = df.iloc[num]
        # img_src = df_row['IMG_URL']
        # name = df_row['NAME']
        # form = df_row['FORM']
        # desc = df_row['DESC']
        # if len(desc) > 300:
        #     desc = desc[:100] + '...'

        children = [
            html.Div([
                # html.Img(src=img_src, style={ "width": "100%" }),
                html.H2(f"test", style={ "color": "darkblue" }),
                html.P(f"test"),
                html.P(f"test"),
            ], style={ 'width': '200px', 'white-space': 'normal'})
        ]

        return True, bbox, children

    @app.callback(
        Output("main-map", "figure"),
        Input("filter_submit", 'n_clicks'),
        [
            State("filter_type", "value"),
            State("filter_rating","value"),
            State("filter_length","value"),
            State("filter_elev","value")
        ]
    )
    def filter_data(n_clicks, filter_type, filter_rating, filter_length, filter_elev):
        # filter data based on results
        filtered_data = data.main_map_data.copy(deep=False)
        # type
        if filter_type != "all" and filter_type is not None:
            filtered_data = filtered_data[filtered_data["type"] == filter_type]
        # # length
        filtered_data = filtered_data[filtered_data["length"] >= filter_length[0]]
        filtered_data = filtered_data[filtered_data["length"] < filter_length[1]]
        # # elevation
        filtered_data = filtered_data[filtered_data["elevation"] >= filter_elev[0]]
        filtered_data = filtered_data[filtered_data["elevation"] < filter_elev[1]]
        # # rating
        filtered_data = filtered_data[filtered_data["avg_rating"] >= filter_rating]
        print(filtered_data)
        main_map = mapbox(filtered_data)
        main_map.update_traces(hoverinfo="none", hovertemplate=None)
        return main_map

    @app.callback(
        [
         Output("modal-header", "children"),
         Output("trail-card", 'children'),
         Output("mini-map", 'figure'),
         Output("modal", "is_open")],
        [Input('main-map', 'clickData')],
        [State("modal", "is_open")]
    )
    def display_click_data(clickData, is_open):
        # get trail id
        trail_id = clickData['points'][0]['text']

        # get trail name
        trail_name = " ".join(trail_id.split("/")[-1].split("-"))
        # get trail description #
        trail_desc = data.trail_descriptions[data.trail_descriptions["trail_id"] == trail_id]["description"].iloc[0]
        child = html.Div(
            [
                html.P(
                    trail_desc
                ),
                html.Hr(className="my-2"),
                dbc.Button("Example Button", color="secondary", outline=True),
            ],
            className="h-100 p-5 bg-light border rounded-3",
        )
        header = dbc.ModalTitle(trail_name)
        # change state of modal
        modal_state = not is_open
        # update mini map
        mini_map = mapbox(data.main_map_data)
        mini_map.update_layout(mapbox=dict(
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=float(clickData['points'][0]['lat']),
                lon=float(clickData['points'][0]['lon'])
            ),
            pitch=0,
            zoom=6.5
        ))

        return header, child, mini_map, modal_state



    app.run_server(debug=True)



if __name__ == "__main__":
    run(csv_dir=r"C:\Users\NoahB\Desktop\School\first year MCSC (2021-2022)\CS6612\group_proj\GimmeAllTheTrails\data\csv")