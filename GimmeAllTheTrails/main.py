import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from wordcloud import WordCloud
from scipy.spatial.distance import hamming
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

    if "is_selected" in map_data.columns:
        fig.update_traces(marker = go.scattermapbox.Marker(color = map_data["is_selected"].astype(int), size=map_data["is_selected"].astype(int).apply(lambda x: 15 if x == 1 else 9),
                                                            )
        )


    return fig

def cluster_mapbox(map_data):
    fig = go.Figure(go.Scattermapbox(
        lat=map_data.latitude.tolist(),
        lon=map_data.longitude.tolist(),
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=9, color = map_data.clusters.tolist()
        ),

        text=map_data.trail_id.tolist()
    )
    )
    # https://plotly.github.io/plotly.py-docs/generated/plotly.express.scatter_mapbox.html
    fig.update_layout(
        autosize=True,
        hovermode='closest',
        mapbox_style='carto-darkmatter',
        mapbox=dict(
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=45,
                lon=-63
            ),
            pitch=0,
            zoom=5
        ),
        margin=go.layout.Margin(
                        l=0,
                        r=0,
                        b=0,
                        t=0),
        clickmode='event+select',
    )
    return fig

def compute_hamming_dist(selected_point, all_points):
    return [hamming(selected_point, cur_point) for cur_point in all_points]

def plot_hamming(hamming_df):

    fig = go.Figure(data=[
        go.Scatter(
            mode='markers',
            x=hamming_df["distance"],
            y=[-0.08 for x in hamming_df["distance"]],
            line=dict(
                color='Black'
            ),
            marker=dict(
                color='LightSkyBlue',
                size=20,
                line=dict(
                    color='MediumPurple',
                    width=2
                )
            ),
        )
    ])

    fig.update_layout(template="simple_white")

    fig.update_layout(
        autosize=False,
        height=300,
        paper_bgcolor="White",
        )

    fig.update_layout(
        xaxis=dict(
            tickangle=45,
            title_font={"size": 20},
            title_standoff=10),
            )

    fig.update_yaxes(visible=False)
    fig.update_yaxes(range=[-0.1, 0.1])


    fig.update_xaxes(range=[0, hamming_df["distance"]])
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')

    # Set custom x-axis labels
    fig.update_xaxes(
        ticktext = hamming_df["trail_name"].tolist(),
        tickvals = hamming_df["distance"].tolist()
    )



    return fig
def wordcloud(word_cloud):
    fig = px.imshow(word_cloud)
    fig.update_layout(
        autosize=True,
        margin=go.layout.Margin(
            l=0,
            r=0,
            b=0,
            t=0),
            template="plotly_dark"
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig

def sentiment_scatter_plot(data, selected_trails, current_trail, trail_name):
    selected_data = data.sentiment_analysis_data[data.sentiment_analysis_data["trail_id"].isin(selected_trails)]
    ind = selected_data[selected_data["trail_id"] == current_trail].index.item()
    selected_data.at[ind, 'Analysis'] = trail_name
    # fig = px.histogram(selected_data, x="Subjectivity")
    sent_an_plot = px.scatter(selected_data,
                              x='Polarity',
                              y='Subjectivity',
                              color='Analysis',
                              size='Subjectivity',
                              template="plotly_dark",
                              color_discrete_sequence=px.colors.qualitative.G10)
    return sent_an_plot

def get_sentiment_analysis(data, selected_trails, full_set=False):
    selected_data = data.sentiment_analysis_data[data.sentiment_analysis_data["trail_id"].isin(selected_trails)]
    if full_set:
        words = selected_data["clean_reviews"].tolist()
        if not len(words):
            words = ["NA"]
        word_cloud_data = " ".join(words)

        full_wordcloud = WordCloud(width=1000, height=600, margin=0).generate(word_cloud_data)
        return wordcloud(full_wordcloud)

    else:
        sent_an_plot = px.scatter(selected_data,
                                  x='Polarity',
                                  y='Subjectivity',
                                  color='Analysis',
                                  size='Subjectivity',
                                  template = "plotly_dark")
        sent_an_plot.update_layout(margin = go.layout.Margin(
                                    l=0,
                                    r=0,
                                    b=0,
                                    t=0))
        positive = selected_data[
            selected_data["Analysis"] == "Positive"]["clean_reviews"].tolist()
        if len(positive):
            pos_word_cloud_data = " ".join(selected_data[
                                               selected_data["Analysis"] == "Positive"]["clean_reviews"].tolist())
            pos_wordcloud = WordCloud(width=620, height=480, margin=0).generate(pos_word_cloud_data)
            pos_wordcloud = wordcloud(pos_wordcloud)
        else:
            empty_fig = go.Figure()
            empty_fig.update_layout(template="plotly_dark")
            pos_wordcloud = empty_fig
        negative = selected_data[
            selected_data["Analysis"] == "Negative"][
            "clean_reviews"].tolist()
        if len(negative):
            neg_word_cloud_data = " ".join(negative)
            neg_wordcloud = WordCloud(width=620, height=480, margin=0).generate(neg_word_cloud_data)
            neg_wordcloud = wordcloud(neg_wordcloud)
        else:
            empty_fig = go.Figure()
            empty_fig.update_layout(template="plotly_dark")
            neg_wordcloud = empty_fig
        return sent_an_plot, pos_wordcloud, neg_wordcloud
def get_filtered_map_data(data, selected_trails, filter_type, filter_rating, filter_length, filter_elev):
    # filter data based on results
    filtered_data = data.main_map_data.copy(deep=False)
    selected_trails = filtered_data[filtered_data["trail_id"].isin(selected_trails)]
    to_filter = filtered_data[~filtered_data["trail_id"].isin(selected_trails)]
    # type
    if filter_type != "all" and filter_type is not None:
        to_filter = to_filter[to_filter["type"] == filter_type]
    # # length
    to_filter = to_filter[to_filter["length"] >= filter_length[0]]
    to_filter = to_filter[to_filter["length"] < filter_length[1]]
    # # elevation
    to_filter = to_filter[to_filter["elevation"] >= filter_elev[0]]
    to_filter = to_filter[to_filter["elevation"] < filter_elev[1]]
    # # rating
    to_filter = to_filter[to_filter["avg_rating"] >= filter_rating]
    to_filter.index = to_filter["trail_id"]
    selected_trails.index = selected_trails["trail_id"]
    filtered_data = pd.concat([to_filter, selected_trails], axis=0)
    return filtered_data


# APP #
# @click.command()
# @click.option('--csv_dir', default='', help='directory of csv files')
def run(csv_dir):
    # init app #
    app = dash.Dash(__name__,
                    external_stylesheets=[dbc.themes.CYBORG])
    load_figure_template("bootstrap")
    # load data #
    data = AllTrails(csv_dir)

    # COMPONENTS #

    # TRAIL CARD POPUP #
    # mini map view #
    # modal container for trail view #
    trail_view = html.Div(
        [
            dbc.Modal(
                [
                    dbc.ModalHeader(children=[], id="modal-header"),
                    dbc.ModalBody([dbc.Row(
                                    children=[],
                                    className="align-items-md-stretch", id="modal-content"
                                    ),dbc.Button("Select", color="primary", id="select_trail", n_clicks=0)]),
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
                filter_form,
                html.Div(dbc.Col([dbc.Button("Filter", color="primary", id="filter_submit"),
                        dbc.Button("Select All Filtered Trails", color="primary", id="select_all_filtered"),
                        dbc.Button("Select All Trails", color="primary", id="select_all", n_clicks=0),
                        dbc.Button("Unselect All", color="primary", id="unselect_all", n_clicks=0)],
                        ), style={'width': 'auto', 'height': 'auto', 'display': 'flex'})

            ]
        ),
        style={ "height": "66vh", "width": "20vw",
                # "padding-top": "5vh",
                # "padding-left": "0.5vh",
                "position": "absolute",
                "top": "1vh",
                "left": "1vw",
                "z-index": "2",
                "opacity":"0.8"},
    )
    fig = px.scatter(data.sentiment_analysis_data,
                         x='Polarity',
                         y='Subjectivity',
                         color = 'Analysis',
                         size='Subjectivity',
                         template = "plotly_dark")
    fig.update_layout(margin=go.layout.Margin(
        l=0,
        r=0,
        b=0,
        t=0))
    # sentiment analysis
    empty_fig = go.Figure()
    empty_fig.update_layout(template="plotly_dark")
    sent_analysis = html.Div([
        dcc.Graph(figure=empty_fig,
                  config={
                      'displayModeBar': False,
                      'staticPlot': True
                  },
                  style={ 'width': '100%', 'height': '100%', 'display': 'flex', "border-radius": "25px"
                          },
                  id="sentiment-analysis"
                  )
    ],

        style={ "height": "31vh", "width": "20vw",
                #"padding-top": "3vh",
                #"padding-left": "0.5vh",
                "position": "absolute",
                "top": "1vh",
                "right": "1vw",
                "z-index": "2",
                "opacity": "0.8" , "border-radius": "25px"},
        className="border rounded-3"
    )

    # positive word cloud
    empty_fig = go.Figure()
    empty_fig.update_layout(template = "plotly_dark")
    positive_sentiment = html.Div([
        dcc.Graph(figure=empty_fig,
                  config={
                      'displayModeBar': False,
                      'staticPlot': True
                  },
                  style={ 'width': '100%', 'height': '100%', 'display': 'flex'
                          },
                  id="positive-sentiment"
                  )
    ],
        style={ "height": "31vh", "width": "20vw",
                #"padding-top": "2vh",
                #"padding-left": "0.5vh",
                "position": "absolute",
                "top": "33vh",
                "right": "1vw",
                "z-index": "2",
                "opacity": "0.8" },
        className="border rounded-3"
    )

    # negative word cloud
    empty_fig = go.Figure()
    empty_fig.update_layout(template="plotly_dark")
    negative_sentiment = html.Div([
        dcc.Graph(figure=empty_fig,
                  config={
                      'displayModeBar': False,
                      'staticPlot': True
                  },
                  style={ 'width': '100%', 'height': '100%', 'display': 'flex'
                          },
                  id="negative-sentiment"
                  )
    ],
        style={ "height": "31vh", "width": "20vw",
                # "padding-top": "2vh",
                # "padding-left": "0.5vh",
                "position": "absolute",
                "top": "65vh",
                "right": "1vw",
                "z-index": "2",
                "opacity": "0.8" },
        className="border rounded-3"
    )
    # cluster map
    # clusters = mapbox(data.cluster_map_data)
    # print(data.cluster_map_data.cluster)
    # clusters.update_layout(mapbox=dict(
    #         zoom=5
    #     ),
    #     marker=go.scattermapbox.Marker(
    #         size=9, color=data.cluster_map_data.cluster
    #     ),
    # )
    cluster_map = html.Div([html.A(id='top', children=dcc.Graph(figure=cluster_mapbox(data.cluster_map_data),
                            id='cluster-map',
                            config={
                                'displayModeBar': False
                            },
                            clear_on_unhover=True,
                  style={ "height": "100%", "width": "100%"}))],
        style={ "height": "31vh", "width": "20vw",
                "position": "absolute",
                "top": "65.5vh",
                "left": "1vw",
                "z-index": "2",
                "opacity": "0.8",
                },
        className="border rounded-3"
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
                dcc.Tooltip(id="main-map-hover", children=[],show=False, style={ "position": "absolute", "z-index": "2" }),
                               ])
    main = html.Div(
        [
            # navbar,
            trail_view,
            sent_analysis,
            positive_sentiment,
            negative_sentiment,
            main_map_graph,
            filters,
            cluster_map,
            dcc.Store(id='current_trail_id', data=None),
            dcc.Store(id="selected_trails", data=[]),
            dcc.Store(id="selected_filter", data=None),
            dcc.Store(id="select_trail_clicks", data=0),
            dcc.Store(id="filter_trail_clicks", data=0),
            dcc.Store(id="has_been_init", data=False),
            dcc.Store(id="select_all_clicks", data=0),
            dcc.Store(id="select_filter_clicks", data=0),
            dcc.Store(id="unselect_all_clicks", data=0),
            # trail_list
        ],
        style={ 'width': '100%', 'height': '100%' })


    app.layout = main


    # FUNCTIONALITY

    @app.callback(
        Output("main-map-hover", "show"),
         Output("main-map-hover", "bbox"),
         Output("main-map-hover", "children"),
        Input("main-map", "hoverData")
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update

        # demo only shows the first point, but other points may also be available
        pt = hoverData["points"][0]
        bbox = pt["bbox"]

        num = pt["pointNumber"]
        # print(pt, bbox, num)
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
                html.H2(f"test"),
                html.P(f"test"),
                html.P(f"test"),
            ], style={ 'width': '200px'})
        ]

        return True, bbox, children



    @app.callback(
        [
         Output("modal-header", "children"),
         Output("modal", 'children'),
         Output("modal", "is_open"),
         Output("current_trail_id", 'data'),
         Output("selected_trails", 'data'),
         Output("filter_trail_clicks", 'data'),
         Output("select_trail_clicks", 'data'),
         Output("main-map", "figure"),
         Output("has_been_init", "data"),
         Output("cluster-map", "figure"),
         Output("sentiment-analysis", "figure"),
         Output("select_all_clicks", 'data'),
         Output("select_filter_clicks", 'data'),
         Output("unselect_all_clicks", 'data'),
         Output("positive-sentiment", "figure"),
         Output("negative-sentiment", "figure")
        ],
        [Input('main-map', 'clickData'),
         Input("select_trail", 'n_clicks'),
         Input("filter_submit", 'n_clicks'),
         Input("select_all", 'n_clicks'),
         Input("select_all_filtered", 'n_clicks'),
         Input("unselect_all", 'n_clicks'),
         Input('current_trail_id', 'data'),
         Input("selected_trails", 'data'),
         Input("filter_trail_clicks", 'data'),
         Input("select_trail_clicks", 'data'),
         Input("has_been_init", 'data'),
         Input("select_all_clicks", 'data'),
         Input("select_filter_clicks", 'data'),
         Input("unselect_all_clicks", 'data'),
         ],
        [State("modal", "is_open"),
         State("filter_type", "value"),
         State("filter_rating", "value"),
         State("filter_length", "value"),
         State("filter_elev", "value")
         ]
    )
    def main_callback(clickData, selected_cur_clicks, filter_cur_clicks,
                      select_all_cur_clicks, select_filter_cur_clicks, unselect_all_cur_clicks,
                      current_trail_id, selected_trails, filter_trail_clicks,
                      select_trail_clicks, has_been_init, select_all_clicks,
                      select_filter_clicks, unselect_all_clicks,
                      is_open, filter_type, filter_rating, filter_length, filter_elev):
        # SELECT TRAIL BUTTON #
        if selected_cur_clicks:
            if selected_cur_clicks > select_trail_clicks:
                if current_trail_id:
                    selected_trails.append(current_trail_id)
                    filter_data = get_filtered_map_data(data, selected_trails, filter_type, filter_rating, filter_length, filter_elev)
                    selected_trails = set(selected_trails)
                    # update main map
                    filter_data["is_selected"] = filter_data["trail_id"].isin(selected_trails)
                    main_map = mapbox(filter_data)
                    # update sentiment analysis plots
                    sent_an_plot, pos_wordcloud, neg_wordcloud = get_sentiment_analysis(data, selected_trails)
                    return no_update, no_update, False, None, list(selected_trails), no_update, \
                           selected_cur_clicks, main_map, no_update, no_update, sent_an_plot, no_update, no_update, no_update, pos_wordcloud, neg_wordcloud
                else:
                    return no_update, no_update, False, None, no_update, no_update, selected_cur_clicks, \
                           no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

        # FILTER BUTTON #
        if not has_been_init:
            filter_data = get_filtered_map_data(data, selected_trails, filter_type, filter_rating, filter_length, filter_elev)
            main_map = mapbox(filter_data)
            cluster_data = data.cluster_map_data[data.cluster_map_data["trail_id"].isin(filter_data["trail_id"])]
            cluster_map = cluster_mapbox(cluster_data)
            return no_update, no_update, no_update, no_update, no_update, 0, \
                   no_update, main_map, True, cluster_map, no_update, no_update, no_update, no_update, no_update, no_update

        elif filter_cur_clicks:
            if filter_cur_clicks > filter_trail_clicks:
                filter_data = get_filtered_map_data(data, selected_trails, filter_type, filter_rating, filter_length, filter_elev)
                filter_data["is_selected"] = filter_data["trail_id"].isin(selected_trails)
                main_map = mapbox(filter_data)
                cluster_data = data.cluster_map_data[data.cluster_map_data["trail_id"].isin(filter_data["trail_id"])]
                cluster_map = cluster_mapbox(cluster_data)
                return no_update, no_update, no_update, no_update, no_update, \
                       filter_cur_clicks, no_update, main_map, no_update, cluster_map, \
                       no_update, no_update, no_update, no_update, no_update, no_update

        # SELECT ALL BUTTON #
        if select_all_cur_clicks:
            if select_all_cur_clicks > select_all_clicks:
                selected_trails = data.main_map_data["trail_id"].tolist()
                filter_data = data.main_map_data.copy(deep=False)
                filter_data["is_selected"] = filter_data["trail_id"].isin(selected_trails)
                main_map = mapbox(filter_data)
                # update sentiment analysis plots
                sent_an_plot, pos_wordcloud, neg_wordcloud = get_sentiment_analysis(data, selected_trails)
                return no_update, no_update, no_update, None, selected_trails, \
                       no_update, no_update, main_map, no_update, no_update, sent_an_plot, \
                       select_all_cur_clicks, no_update, no_update, pos_wordcloud, neg_wordcloud

        # SELECT FILTER BUTTON #
        if select_filter_cur_clicks:
            if select_filter_cur_clicks > select_filter_clicks:
                selected_trails+= get_filtered_map_data(data, selected_trails, filter_type, filter_rating, filter_length, filter_elev)["trail_id"].tolist()
                filter_data = get_filtered_map_data(data, selected_trails, filter_type, filter_rating, filter_length,
                                                    filter_elev)
                filter_data["is_selected"] = filter_data["trail_id"].isin(selected_trails)
                main_map = mapbox(filter_data)
                sent_an_plot, pos_wordcloud, neg_wordcloud = get_sentiment_analysis(data, selected_trails)
                return no_update, no_update, no_update, None, selected_trails, \
                       no_update, no_update, main_map, no_update, no_update, sent_an_plot, \
                        no_update, select_filter_cur_clicks, no_update, pos_wordcloud, neg_wordcloud

        # UNSELECT ALL BUTTON #
        if unselect_all_cur_clicks:
            if unselect_all_cur_clicks > unselect_all_clicks:
                selected_trails = []
                filter_data = get_filtered_map_data(data, selected_trails, filter_type, filter_rating, filter_length,
                                                    filter_elev)
                filter_data["is_selected"] = filter_data["trail_id"].isin(selected_trails)
                main_map = mapbox(filter_data)
                sent_an_plot, pos_wordcloud, neg_wordcloud = get_sentiment_analysis(data, selected_trails)
                return no_update, no_update, no_update, None, selected_trails, \
                       no_update, no_update, main_map, no_update, no_update, sent_an_plot, \
                       no_update, no_update, unselect_all_cur_clicks, pos_wordcloud, neg_wordcloud


       # ELSE OPEN UP TRAIL CARD #

        modal_state = not is_open
        # get trail id
        trail_id = clickData['points'][0]['text']
        # get trail name
        trail_name = " ".join(trail_id.split("/")[-1].split("-"))

        if len(trail_name) > 7:
            shortened_trail_name = trail_name[:8] + "..."
        else:

            shortened_trail_name = trail_name
        # update mini map
        # show only points that exist in same cluster

        cluster = data.cluster_map_data[data.cluster_map_data["trail_id"] == trail_id]["clusters"].iloc[0]
        cluster_data = data.cluster_map_data[data.cluster_map_data["clusters"] == cluster]
        mini_map = cluster_mapbox(cluster_data)
        mini_map.update_layout(mapbox=dict(
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=float(clickData['points'][0]['lat']),
                lon=float(clickData['points'][0]['lon'])
            ),
            pitch=0,
            zoom=5.0
        ))

        mini_map = html.Div(
            [
                dcc.Graph(figure=mini_map,
                          config={
                              'displayModeBar': False,
                              'staticPlot': False
                          },
                          style={ 'width': '100%', 'height': '100%', 'display': 'flex'
                                  },
                          id="mini-map"
                          )
            ],

            # className="border rounded-3",
            style={ "height": "35vh", "width": "30vw",
                    "position": "absolute",
                    "top": "1vh",
                    "left": "1vw",
                    "z-index": "2",
                    "opacity": "0.8" },
            className="border rounded-3"
        )

        # get trail description #
        trail_desc = data.trail_descriptions[data.trail_descriptions["trail_id"] == trail_id]["description"].iloc[0]
        # trail card #
        avg_stars = int(data.main_map_data[data.main_map_data["trail_id"] == trail_id]['avg_rating'].iloc[0])
        keywords = data.key_words[data.key_words["trail_id"] == trail_id]["set"]
        trail_card = html.Div(
            [
                html.H3(("★" * avg_stars) + ("☆" * (5- avg_stars))),
                html.P(
                    trail_desc
                ),

            ] + [dbc.Badge(kw, color="dark", className="me-1") for kw in keywords.iloc[0]],

        )

        # description of trail #
        trail_description =html.Div(
                children=trail_card,
                className="p-3 bg-light border rounded-3",
                style={ "height": "35vh", "width": "40vw",
                        # "padding-top": "5vh",
                        # "padding-left": "0.5vh",
                        "position": "absolute",
                        "top": "1vh",
                        "left": "31.5vw",
                        "z-index": "2",
                        "opacity": "0.8" },
                id="trail-card"
            )
        length_height = data.main_map_data[data.main_map_data["trail_id"] == trail_id]
        normalized_length = length_height["normalized_elevation"].iloc[0]
        normalized_elevation = length_height["normalized_length"].iloc[0]
        reviews = data.num_reviews[data.num_reviews["trail_id"] == trail_id]
        normalized_ratings = reviews["ratings_normalized"].iloc[0]
        normalized_written = reviews["written_normalized"].iloc[0]

        fig1 = px.bar(x=["Length", "Elevation"], y=[normalized_length, normalized_elevation])
        fig1.update_yaxes(visible=False)
        fig1.update_xaxes(visible=False)
        fig1.update_layout( title="Trail Stats",
                            template = "plotly_dark",
                            margin=go.layout.Margin(
                                l=0,
                                r=0,
                                b=0,
                                t=100)
                            )


        fig2 = px.bar(x=["num ratings", "num reviews"], y=[normalized_ratings, normalized_written])
        fig2.update_yaxes(visible=False)
        fig2.update_xaxes(visible=False)
        fig2.update_layout(title="Review Stats",
                           template = "plotly_dark",
                            margin=go.layout.Margin(
                                l=0,
                                r=0,
                                b=0,
                                t=100)
                           )
        # length versus elev #
        length_height = html.Div(
            children=dbc.Row([dbc.Col(dcc.Graph(figure=fig1,
                                         style={ 'width': '100%', 'height': '100%'
                                              },
                               config={
                                    'displayModeBar': False,
                                    'staticPlot': False
                                    },
                                   ), style={ 'width': '50%', 'height': '100%', "padding-left": "0.5vh", "padding-right": "0.5vh"
                                              }, className="border rounded-3"),
                      dbc.Col(dcc.Graph(figure=fig2,
                                style={ 'width': '100%', 'height': '100%'},
                               config={
                                    'displayModeBar': False,
                                    'staticPlot': False
                                    },
                                   ), style={ 'width': '50%', 'height': '100%', "padding-left": "0.5vh", "padding-right": "0.5vh"}, className="border rounded-3")], style={ 'width': '100%', 'height': '100%', "padding-left": "0vh", "padding-right": "0vh"}),
            # className="p-5 bg-light border rounded-3",
            style={ "height": "35vh", "width": "28vw",
                    # "padding-top": "5vh",
                    # "padding-left": "0.5vh",
                    "position": "absolute",
                    "top": "1vh",
                    "left": "72vw",
                    "z-index": "2",
                    "opacity": "0.8" },
            id="length-height"
        )

        cluster_data["colour"] = (cluster_data["trail_id"] == trail_id).apply(lambda x: shortened_trail_name if x else "cluster")
        scatter_l_h = px.scatter(x=cluster_data["length"], y=cluster_data["elevation"], color=cluster_data["colour"])
        scatter_l_h.update_layout(
            xaxis_title="length (Mi.)",
            yaxis_title="Elevation (ft.)",
            title="Length vs. Elevation for Associated Cluster",
            template = "plotly_dark",
            margin=go.layout.Margin(
                l=0,
                r=0,
                b=0,
                t=0)
        )
        similar_trails = html.Div(
            [
             #html.H4("Similar Trails"),
             #html.Div([html.H5(["To be implemented...", dbc.Badge("New", className="ms-1")])])
             dcc.Graph(figure=scatter_l_h,
                       style={ 'width': '100%', 'height': '100%', 'display': 'flex'},
                       config={
                           'displayModeBar': False,
                           'staticPlot': False
                       }
                       )
            ],
            # className="p-5 bg-light border rounded-3",
            style={"height": "50vh", "width": "29.5vw",
                    # "padding-top": "5vh",
                    # "padding-left": "0.5vh",
                    "position": "absolute",
                    "top": "37vh",
                    "left": "1vw",
                    "z-index": "2",
                    "opacity": "0.8"},
            id="similar-trails",
            className="border rounded-3"
        )

        fig = sentiment_scatter_plot(data, cluster_data["trail_id"].tolist(), trail_id, shortened_trail_name)

        fig.update_layout(title="Sentiment Analysis for Associated Cluster",
                          margin=go.layout.Margin(
                              l=0,
                              r=0,
                              b=0,
                              t=0)
                          )
        sentiment_scatter = dbc.Col( html.Div(
                children=dcc.Graph(figure=fig,style={ 'width': '100%', 'height': '100%', 'display': 'flex'
                                              },
                                   config={
                                       'displayModeBar': False,
                                       'staticPlot': False
                                   }
                                   ),
                # className="bg-light border rounded-3",

            style={"height": "50vh", "width": "30.5vw",
                # "padding-top": "5vh",
                # "padding-left": "0.5vh",
                 "position": "absolute",
                "top": "37vh",
                "left": "31vw",
                "z-index": "2",
                "opacity": "0.8" },
            id="height-length",
            className="border rounded-3"
        ),
        )

        full_wordcloud = get_sentiment_analysis(data, selected_trails=[trail_id], full_set=True)

        sentiment_analysis = dbc.Col(html.Div(
            children=dcc.Graph(figure=full_wordcloud, style={ 'width': '100%', 'height': '100%', 'display': 'flex'
                                                   },
                               config={
                                   'displayModeBar': False,
                                   'staticPlot': False
                               },
                               ),
            # className="bg-light border rounded-3",

            style={ "height": "50vh", "width": "36.25vw",
                    # "padding-top": "5vh",
                    # "padding-left": "0.5vh",
                    "position": "absolute",
                    "top": "37vh",
                    "left": "62vw",
                    "z-index": "2",
                    "opacity": "0.8" },
            id="trail-card-sentiment",
        className="border rounded-3") )


        # set header
        header = dbc.ModalTitle(trail_name, style = { "padding-right": "1vw" })

        modal_content = [mini_map,
                         trail_description,
                         sentiment_scatter,
                         sentiment_analysis,
                         length_height,
                         similar_trails]
        modal_content = [
            dbc.ModalHeader(children=[header, dbc.Button("Select", color="primary", id="select_trail",
                                                         n_clicks=0)], id="modal-header"),
            dbc.ModalBody([dbc.Row(
                                children=modal_content,
                                className="align-items-md-stretch", id="modal-content"
                            )]),
        ]
        return header, modal_content, modal_state, trail_id, no_update, no_update, \
               no_update, no_update, no_update, no_update, no_update, \
               no_update, no_update, no_update, no_update, no_update

    # @app.callback(
    #     [Output("selected_trails", 'data')],
    #     [Input("select_trail", 'n_clicks'),
    #      Input('current_trail_id', 'data'),
    #      Input("selected_trails", 'data')]
    # )
    # def add_to_trail_list(n_clicks, current_trail_id, selected_trails):
    #     print(selected_trails)
    #     if current_trail_id:
    #         selected_trails.append(current_trail_id)
    #         return selected_trails,
    #     else:
    #         return no_update,
    app.run_server(debug=False)



if __name__ == "__main__":
    run(csv_dir=r"C:\Users\NoahB\Desktop\School\first year MCSC (2021-2022)\CS6612\group_proj\GimmeAllTheTrails\data\csv")