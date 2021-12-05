import pandas as pd
import dash
import click
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from wordcloud import WordCloud
from dash import dcc, html, Input, Output, no_update, State
import plotly.express as px
import plotly.graph_objects as go
from GimmeAllTheTrails.dataset import AllTrails

# FIGURES #
def mapbox(map_data):
    """
    GENERAL MAP BOX PLOT -> USED FOR MAIN MAP
    :param map_data: df for map plot
    :return: fig
    """
    fig = go.Figure(
        go.Scattermapbox(
            lat=map_data.latitude.tolist(),
            lon=map_data.longitude.tolist(),
            mode="markers",
            marker=go.scattermapbox.Marker(size=9),
            text=map_data.trail_id.tolist(),
        )
    )
    fig.update_layout(
        autosize=True,
        hovermode="closest",
        mapbox_style="carto-darkmatter",
        mapbox=dict(
            bearing=0,
            center=go.layout.mapbox.Center(lat=45, lon=-63),
            pitch=0,
            zoom=6.5,
        ),
        margin=go.layout.Margin(l=0, r=0, b=0, t=0),
        clickmode="event+select",
    )

    if "is_selected" in map_data.columns:
        fig.update_traces(
            marker=go.scattermapbox.Marker(
                color=map_data["is_selected"].astype(int),
                size=map_data["is_selected"]
                .astype(int)
                .apply(lambda x: 15 if x == 1 else 9),
            )
        )

    return fig


def cluster_mapbox(map_data, color_col="clusters"):
    """
    map box for the clustered dataset
    :param map_data: data containing clusters
    :return: figure
    """
    fig = go.Figure(
        go.Scattermapbox(
            lat=map_data.latitude.tolist(),
            lon=map_data.longitude.tolist(),
            mode="markers",
            marker=go.scattermapbox.Marker(size=9, color=map_data[color_col].tolist()),
            text=map_data.trail_id.tolist(),
        )
    )
    fig.update_layout(
        autosize=True,
        hovermode="closest",
        mapbox_style="carto-darkmatter",
        mapbox=dict(
            bearing=0, center=go.layout.mapbox.Center(lat=45, lon=-63), pitch=0, zoom=5
        ),
        margin=go.layout.Margin(l=0, r=0, b=0, t=0),
        clickmode="event+select",
    )
    return fig


def sentiment_mapbox(map_data):
    """
    map box for the sentiment dataset
    :param map_data: data containing clusters
    :return: figure
    """
    fig = go.Figure(
        go.Scattermapbox(
            lat=map_data.latitude.tolist(),
            lon=map_data.longitude.tolist(),
            mode="markers",
            marker=go.scattermapbox.Marker(
                size=9,
                color=map_data.Polarity.tolist(),
                colorscale="RdBu_r",
                showscale=True,
            ),
            text=map_data.trail_id.tolist(),
        )
    )
    fig.update_layout(
        autosize=True,
        hovermode="closest",
        mapbox_style="carto-darkmatter",
        mapbox=dict(
            bearing=0, center=go.layout.mapbox.Center(lat=45, lon=-63), pitch=0, zoom=5
        ),
        margin=go.layout.Margin(l=0, r=0, b=0, t=0),
        clickmode="event+select",
    )
    return fig


def wordcloud(word_cloud):
    """
    make a wordcloud plot
    :param word_cloud: wordcloud generated im
    :return: fig
    """
    fig = px.imshow(word_cloud)
    fig.update_layout(
        autosize=True,
        margin=go.layout.Margin(l=0, r=0, b=0, t=0),
        template="plotly_dark",
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


def sentiment_scatter_plot(data, selected_trails, current_trail, trail_name):
    """
    plot a sentiment scatter plot
    :param data: data to be plotted
    :param selected_trails: trails that are currently selcted by the user
    :param current_trail: the current trail selected by user
    :param trail_name: name of trail selected
    :return: fig
    """
    selected_data = data.sentiment_analysis_data[
        data.sentiment_analysis_data["trail_id"].isin(selected_trails)
    ]
    ind = selected_data[selected_data["trail_id"] == current_trail].index.item()
    selected_data.loc[ind, "Analysis"] = trail_name
    sent_an_plot = px.scatter(
        selected_data,
        x="Polarity",
        y="Subjectivity",
        color="Analysis",
        size="Subjectivity",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.G10,
    )
    return sent_an_plot


def get_sentiment_analysis(data, selected_trails, full_set=False):
    """
    function to get both word cloudsw and sentiment scatter plot
    :param data: DF to be used for analysis
    :param selected_trails: trails selected by user
    :param full_set: wether to use the entire dataset or not
    :return: fig
    """
    selected_data = data.sentiment_analysis_data[
        data.sentiment_analysis_data["trail_id"].isin(selected_trails)
    ]
    if full_set:
        # generate using the entire dataset #
        words = selected_data["clean_reviews"].tolist()
        if not len(words):
            words = ["NA"]
        word_cloud_data = " ".join([str(w) for w in words])
        full_wordcloud = WordCloud(width=1000, height=600, margin=0).generate(
            word_cloud_data
        )
        return wordcloud(full_wordcloud)

    else:
        # generate using subset of trails #
        sent_an_plot = px.scatter(
            selected_data,
            x="Polarity",
            y="Subjectivity",
            color="Analysis",
            size="Subjectivity",
            template="plotly_dark",
        )
        sent_an_plot.update_layout(margin=go.layout.Margin(l=0, r=0, b=0, t=0))
        positive = selected_data[selected_data["Analysis"] == "Positive"][
            "clean_reviews"
        ].tolist()

        # if there is positive sentiment in analysis #
        if len(positive):
            pos_word_cloud_data = " ".join(
                selected_data[selected_data["Analysis"] == "Positive"][
                    "clean_reviews"
                ].tolist()
            )
            pos_wordcloud = WordCloud(width=620, height=480, margin=0).generate(
                pos_word_cloud_data
            )
            pos_wordcloud = wordcloud(pos_wordcloud)
        else:
            empty_fig = go.Figure()
            empty_fig.update_layout(template="plotly_dark")
            pos_wordcloud = empty_fig
        negative = selected_data[selected_data["Analysis"] == "Negative"][
            "clean_reviews"
        ].tolist()

        # if there is negative sentiment in analysis #
        if len(negative):
            neg_word_cloud_data = " ".join(negative)
            neg_wordcloud = WordCloud(width=620, height=480, margin=0).generate(
                neg_word_cloud_data
            )
            neg_wordcloud = wordcloud(neg_wordcloud)
        else:
            empty_fig = go.Figure()
            empty_fig.update_layout(template="plotly_dark")
            neg_wordcloud = empty_fig
        return sent_an_plot, pos_wordcloud, neg_wordcloud


def get_filtered_map_data(
    data, selected_trails, filter_type, filter_rating, filter_length, filter_elev
):
    """
    function to filter the dataset based on the user selected filters
    :param data: Base DF
    :param selected_trails: user sleceted trails
    :param filter_type: filter - type of trail
    :param filter_rating: filter - avg rating of trail
    :param filter_length: filter - length of trail
    :param filter_elev: filter - elevation of trail
    :return: filtered DF
    """
    # filter data based on results
    filtered_data = data.main_map_data.copy(deep=False)
    selected_trails = filtered_data[filtered_data["trail_id"].isin(selected_trails)]
    to_filter = filtered_data[~filtered_data["trail_id"].isin(selected_trails)]
    # type
    if filter_type != "all" and filter_type is not None:
        to_filter = to_filter[to_filter["type"] == filter_type]
    # # length
    to_filter = to_filter[to_filter["normalized_length"] >= filter_length[0]]
    to_filter = to_filter[to_filter["normalized_length"] < filter_length[1]]
    # # elevation
    to_filter = to_filter[to_filter["normalized_elevation"] >= filter_elev[0]]
    to_filter = to_filter[to_filter["normalized_elevation"] < filter_elev[1]]
    # # rating
    to_filter = to_filter[to_filter["avg_rating"] <= filter_rating]
    to_filter.index = to_filter["trail_id"]
    selected_trails.index = selected_trails["trail_id"]
    filtered_data = pd.concat([to_filter, selected_trails], axis=0)
    return filtered_data


def get_main_mini_map(data, selected_trails, main_mini_map_style):
    """
    helper function for main callback code,
    changes cluster map between sentiment plot and cluster_plot
    plots only the selected trails
    :param data: main dataset
    :param cluster_map_style: either cluster or sentiment
    :param filter_data: the filtered data for the cluster map
    :return:
    """
    if main_mini_map_style == "cluster":
        main_mini_data = data.cluster_map_data[
            data.cluster_map_data["trail_id"].isin(set(selected_trails))
        ]
        main_mini_map = cluster_mapbox(main_mini_data)
    elif main_mini_map_style == "sentiment":
        main_mini_data = data.sentiment_analysis_data[
            data.sentiment_analysis_data["trail_id"].isin(set(selected_trails))
        ]
        main_mini_map = sentiment_mapbox(main_mini_data)
    return main_mini_map


# APP #
@click.command()
@click.option("--csv_dir", type=str, default=None, help="directory of csv files")
def run(csv_dir):
    # init app #
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
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
                    dbc.ModalHeader(
                        children=[
                            dbc.ModalTitle(
                                children=[],
                                id="modal-header",
                                style={"padding-right": "1vw"},
                            ),
                            dbc.Button(
                                "Select", color="primary", id="select_trail", n_clicks=0
                            ),
                        ]
                    ),
                    dbc.ModalBody(
                        [
                            dbc.Row(
                                children=[],
                                className="align-items-md-stretch",
                                id="modal-content",
                            )
                        ]
                    ),
                ],
                id="modal",
                fullscreen=True,
                is_open=False,
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
                    {"label": t_type, "value": t_type}
                    for t_type in data.main_map_data["type"].unique()
                ]
                + [{"label": "all", "value": "all"}],
                value="all",
            ),
        ],
        className="mb-3",
    )

    filter_rating = html.Div(
        [
            dbc.Label("Star Rating", html_for="slider"),
            dcc.Slider(id="filter_rating", min=0, max=5, step=1, value=5),
        ],
        className="mb-3",
    )

    filter_length = html.Div(
        [
            dbc.Label("Trail Length", html_for="range-slider"),
            # set min and max to the length column
            dcc.RangeSlider(
                id="filter_length",
                min=data.main_map_data["normalized_length"].min(),
                max=0.3,
                value=[data.main_map_data["normalized_length"].min(), 0.3],
                step=0.05,
            ),
        ],
        className="mb-3",
    )
    filter_elev = html.Div(
        [
            dbc.Label("Trail Elevation", html_for="range-slider"),
            dcc.RangeSlider(
                id="filter_elev",
                min=data.main_map_data["normalized_elevation"].min(),
                max=0.3,
                value=[data.main_map_data["normalized_elevation"].min(), 0.3],
                step=0.05,
            ),
        ],
        className="mb-3",
    )
    # final filter form #
    filter_form = dbc.Form([filter_type, filter_rating, filter_length, filter_elev])
    filters = dbc.Card(
        dbc.CardBody(
            [
                html.H4("Filters", className="card-title"),
                filter_form,
                html.Div(
                    dbc.Col(
                        [
                            dbc.Button("Filter", color="primary", id="filter_submit"),
                            dbc.Button(
                                "Select All Filtered Trails",
                                color="primary",
                                id="select_all_filtered",
                            ),
                            dbc.Button(
                                "Select All Trails",
                                color="primary",
                                id="select_all",
                                n_clicks=0,
                            ),
                            dbc.Button(
                                "Unselect All",
                                color="primary",
                                id="unselect_all",
                                n_clicks=0,
                            ),
                        ],
                    ),
                    style={"width": "auto", "height": "auto", "display": "flex"},
                ),
            ]
        ),
        style={
            "height": "66vh",
            "width": "20vw",
            "position": "absolute",
            "top": "1vh",
            "left": "1vw",
            "z-index": "2",
            "opacity": "0.8",
        },
    )
    # Sentiment Analysis on right #
    empty_fig = go.Figure()
    empty_fig.update_layout(template="plotly_dark")
    sent_analysis = html.Div(
        [
            dcc.Graph(
                figure=empty_fig,
                config={"displayModeBar": False, "staticPlot": True},
                style={
                    "width": "100%",
                    "height": "100%",
                    "display": "flex",
                    "border-radius": "25px",
                },
                id="sentiment-analysis",
            )
        ],
        style={
            "height": "31vh",
            "width": "20vw",
            "position": "absolute",
            "top": "1vh",
            "right": "1vw",
            "z-index": "2",
            "opacity": "0.8",
            "border-radius": "25px",
        },
        className="border rounded-3",
    )

    # positive word cloud
    empty_fig = go.Figure()
    empty_fig.update_layout(template="plotly_dark")
    positive_sentiment = html.Div(
        [
            dbc.Label("Positive Sentiment"),
            dcc.Graph(
                figure=empty_fig,
                config={"displayModeBar": False, "staticPlot": True},
                style={"width": "100%", "height": "95%", "display": "flex"},
                id="positive-sentiment",
            ),
        ],
        style={
            "height": "31vh",
            "width": "20vw",
            "position": "absolute",
            "top": "33vh",
            "right": "1vw",
            "z-index": "2",
            "opacity": "0.8",
        },
        className="border rounded-3",
    )

    # negative word cloud
    empty_fig = go.Figure()
    empty_fig.update_layout(template="plotly_dark")
    negative_sentiment = html.Div(
        [
            dbc.Label("Negative Sentiment"),
            dcc.Graph(
                figure=empty_fig,
                config={"displayModeBar": False, "staticPlot": True},
                style={"width": "100%", "height": "90%", "display": "flex"},
                id="negative-sentiment",
            ),
        ],
        style={
            "height": "31vh",
            "width": "20vw",
            "position": "absolute",
            "top": "65vh",
            "right": "1vw",
            "z-index": "2",
            "opacity": "0.8",
        },
        className="border rounded-3",
    )
    # Cluster map on bottom left corner #
    main_mini_map = html.Div(
        [
            dcc.Dropdown(
                id="main_mini_map_style",
                # in meta reviews
                options=[
                    {"label": "Trail Clusters", "value": "cluster"},
                    {"label": "Trail Sentiment", "value": "sentiment"},
                ],
                value="cluster",
            ),
            dcc.Graph(
                figure=cluster_mapbox(data.cluster_map_data),
                id="main-mini-map",
                config={"displayModeBar": False},
                clear_on_unhover=True,
                style={"height": "100%", "width": "100%"},
            ),
        ],
        style={
            "height": "31vh",
            "width": "20vw",
            "position": "absolute",
            "top": "65.5vh",
            "left": "1vw",
            "z-index": "2",
            "opacity": "0.8",
        },
        className="border rounded-3",
    )

    # main map #
    main_map_graph = html.Div(
        [
            dcc.Graph(
                figure={},
                id="main-map",
                config={"displayModeBar": False},
                clear_on_unhover=True,
                style={
                    "width": "100vw",
                    "height": "100vh",
                    "display": "flex",
                    "position": "absolute",
                    "z-index": "1",
                },
            ),
            dcc.Tooltip(
                id="main-map-hover",
                children=[],
                show=False,
                style={"position": "absolute", "z-index": "2"},
            ),
        ]
    )
    main = html.Div(
        [
            # app components #
            trail_view,
            sent_analysis,
            positive_sentiment,
            negative_sentiment,
            main_map_graph,
            filters,
            main_mini_map,
            # variables for callback functionality #
            dcc.Store(id="current_trail_id", data=None),
            dcc.Store(id="selected_trails", data=[]),
            dcc.Store(id="selected_filter", data=None),
            dcc.Store(id="select_trail_prev_clicks", data=0),
            dcc.Store(id="filter_trail_prev_clicks", data=0),
            dcc.Store(id="has_been_init", data=False),
            dcc.Store(id="select_all_prev_clicks", data=0),
            dcc.Store(id="select_filter_prev_clicks", data=0),
            dcc.Store(id="unselect_all_prev_clicks", data=0),
            dcc.Store(id="prev_main_mini_map_style", data="cluster"),
        ],
        style={"width": "100%", "height": "100%"},
    )

    app.layout = main

    # FUNCTIONALITY

    @app.callback(
        [
            Output("has_been_init", "data"),
            Output("filter_trail_prev_clicks", "data"),
            Output("main-mini-map", "figure"),
            Output("prev_main_mini_map_style", "data"),
            Output("select_trail_prev_clicks", "data"),
            Output("select_all_prev_clicks", "data"),
            Output("select_filter_prev_clicks", "data"),
            Output("unselect_all_prev_clicks", "data"),
            Output("sentiment-analysis", "figure"),
            Output("positive-sentiment", "figure"),
            Output("negative-sentiment", "figure"),
            Output("modal", "is_open"),
            Output("modal-header", "children"),
            Output("modal-content", "children"),
            Output("current_trail_id", "data"),
            Output("selected_trails", "data"),
            Output("main-map", "figure"),
        ],
        [
            Input("main-map", "clickData"),
            Input("select_trail", "n_clicks"),
            Input("filter_submit", "n_clicks"),
            Input("select_all", "n_clicks"),
            Input("select_all_filtered", "n_clicks"),
            Input("unselect_all", "n_clicks"),
            Input("current_trail_id", "data"),
            Input("selected_trails", "data"),
            Input("filter_trail_prev_clicks", "data"),
            Input("select_trail_prev_clicks", "data"),
            Input("has_been_init", "data"),
            Input("select_all_prev_clicks", "data"),
            Input("select_filter_prev_clicks", "data"),
            Input("unselect_all_prev_clicks", "data"),
            Input("main_mini_map_style", "value"),
            Input("prev_main_mini_map_style", "data"),
        ],
        [
            State("modal", "is_open"),
            State("filter_type", "value"),
            State("filter_rating", "value"),
            State("filter_length", "value"),
            State("filter_elev", "value"),
        ],
    )
    def main_callback(
        clickData,
        select_trail_cur_clicks,
        filter_cur_clicks,
        select_all_cur_clicks,
        select_filter_cur_clicks,
        unselect_all_cur_clicks,
        current_trail_id,
        selected_trails,
        filter_trail_prev_clicks,
        select_trail_prev_clicks,
        has_been_init,
        select_all_prev_clicks,
        select_filter_prev_clicks,
        unselect_all_prev_clicks,
        main_mini_map_style,
        prev_main_mini_map_style,
        modal_is_open,
        filter_type,
        filter_rating,
        filter_length,
        filter_elev,
    ):

        # boolean logics #
        # done this way to set shared variables #
        is_not_init = not has_been_init
        is_filter = filter_cur_clicks and (filter_cur_clicks > filter_trail_prev_clicks)
        is_switch_mini_map_style = main_mini_map_style != prev_main_mini_map_style
        is_select_trail = select_trail_cur_clicks and (
            select_trail_cur_clicks > select_trail_prev_clicks
        )
        is_select_all = select_all_cur_clicks and (
            select_all_cur_clicks > select_all_prev_clicks
        )
        is_select_filter = select_filter_cur_clicks and (
            select_filter_cur_clicks > select_filter_prev_clicks
        )
        is_unselect_all = unselect_all_cur_clicks and (
            unselect_all_cur_clicks > unselect_all_prev_clicks
        )

        # handling the variables not being filled by statements below
        # handling no sentiment analysis being updated
        sentiment_analysis = no_update
        positive_sentiment = no_update
        negative_sentiment = no_update
        # handling no main_map being updated
        main_map = no_update
        # handling not calling show modal
        modal_header = no_update
        modal_content = no_update
        main_mini_map = no_update

        # FILTER BUTTON #
        if is_not_init:
            # first initialization of map, show filtered data, and initalize some variables
            filter_data = get_filtered_map_data(
                data,
                selected_trails,
                filter_type,
                filter_rating,
                filter_length,
                filter_elev,
            )
            main_map = mapbox(filter_data)
            has_been_init = True
            filter_trail_prev_clicks = 0
            # set current trail id to none, there are no trails selected
            current_trail_id = None

        elif is_filter:
            filter_data = get_filtered_map_data(
                data,
                selected_trails,
                filter_type,
                filter_rating,
                filter_length,
                filter_elev,
            )
            filter_data.loc[:, "is_selected"] = filter_data["trail_id"].isin(
                selected_trails
            )
            main_map = mapbox(filter_data)
            filter_trail_prev_clicks = filter_cur_clicks
            # set current trail id to none, there are no trails selected
            current_trail_id = None

        # MINI-MAP SWITCH VIEWS BETWEEN CLUSTER AND SENTIMENT ANALYSIS #
        elif is_switch_mini_map_style:
            filter_trails = get_filtered_map_data(
                data,
                selected_trails,
                filter_type,
                filter_rating,
                filter_length,
                filter_elev,
            )["trail_id"].to_list()
            # we must then create a new map
            main_mini_map = get_main_mini_map(data, filter_trails, main_mini_map_style)
            prev_main_mini_map_style = main_mini_map_style
            # set current trail id to none, there are no trails selected
            current_trail_id = None

        # SELECT TRAIL BUTTON #
        elif is_select_trail:
            # if the select trail button has been clicked #
            # current trail to selected trails
            selected_trails.append(current_trail_id)
            # get the filtered trails
            filter_data = get_filtered_map_data(
                data,
                selected_trails,
                filter_type,
                filter_rating,
                filter_length,
                filter_elev,
            )
            selected_trails = list(set(selected_trails))
            # update main map - plot shows all filtered trails + highlighted selected trails
            filter_data.loc[:, "is_selected"] = filter_data["trail_id"].isin(
                selected_trails
            )
            main_map = mapbox(filter_data)
            # update sentiment analysis plots
            (
                sentiment_analysis,
                positive_sentiment,
                negative_sentiment,
            ) = get_sentiment_analysis(data, selected_trails)
            # close modal
            # modal_is_open = False
            current_trail_id = None
            # update number of clicks to be stored #
            select_trail_prev_clicks = select_trail_cur_clicks

        # SELECT ALL BUTTON #
        elif is_select_all:
            selected_trails = data.main_map_data["trail_id"].tolist()
            filter_data = data.main_map_data.copy(deep=False)
            filter_data.loc[:, "is_selected"] = (
                filter_data["trail_id"].isin(selected_trails).astype(int)
            )
            main_map = mapbox(filter_data)
            # update sentiment analysis plots
            (
                sentiment_analysis,
                positive_sentiment,
                negative_sentiment,
            ) = get_sentiment_analysis(data, selected_trails)
            # set current trail id to none, there are no trails selected
            current_trail_id = None
            # update the selected all prev clicks
            select_all_prev_clicks = select_all_cur_clicks

        # SELECT FILTER BUTTON #
        elif is_select_filter:
            # if select_filter_cur_clicks > select_filter_prev_clicks:
            selected_trails += get_filtered_map_data(
                data,
                selected_trails,
                filter_type,
                filter_rating,
                filter_length,
                filter_elev,
            )["trail_id"].tolist()
            filter_data = get_filtered_map_data(
                data,
                selected_trails,
                filter_type,
                filter_rating,
                filter_length,
                filter_elev,
            )
            filter_data.loc[:, "is_selected"] = (
                filter_data["trail_id"].isin(selected_trails).astype(int)
            )

            # move selected to bottom of df so we can see it better #
            not_selected = filter_data[~filter_data["trail_id"].isin(selected_trails)]
            selected = filter_data[filter_data["trail_id"].isin(selected_trails)]
            filter_data = pd.concat([not_selected, selected])
            main_map = mapbox(filter_data)
            (
                sentiment_analysis,
                positive_sentiment,
                negative_sentiment,
            ) = get_sentiment_analysis(data, selected_trails)
            # update the selected filter prev clicks
            select_filter_prev_clicks = select_filter_cur_clicks
            # set current trail id to none, there are no trails selected
            current_trail_id = None

        # UNSELECT ALL BUTTON #
        elif is_unselect_all:
            selected_trails = []
            filter_data = get_filtered_map_data(
                data,
                selected_trails,
                filter_type,
                filter_rating,
                filter_length,
                filter_elev,
            )
            filter_data.loc[:, "is_selected"] = (
                filter_data["trail_id"].isin(selected_trails).astype(int)
            )
            main_map = mapbox(filter_data)
            (
                sentiment_analysis,
                positive_sentiment,
                negative_sentiment,
            ) = get_sentiment_analysis(data, selected_trails)
            unselect_all_prev_clicks = unselect_all_cur_clicks
            # set current trail id to none, there are no trails selected
            current_trail_id = None

        # ELSE OPEN UP TRAIL CARD #
        else:
            modal_is_open = True
            # get trail id
            current_trail_id = clickData["points"][0]["text"]
            # get trail name
            trail_name = " ".join(current_trail_id.split("/")[-1].split("-"))
            if len(trail_name) > 7:
                shortened_trail_name = trail_name[:8] + "..."
            else:
                shortened_trail_name = trail_name

            # update mini map
            # show only points that exist in same cluster
            trail_clustermap_data = data.cluster_map_data
            cluster = trail_clustermap_data[
                trail_clustermap_data["trail_id"] == current_trail_id
            ]["clusters"].iloc[0]
            trail_clustermap_data = trail_clustermap_data[
                trail_clustermap_data["clusters"] == cluster
            ]
            trail_mini_map = cluster_mapbox(trail_clustermap_data)
            trail_mini_map.update_layout(
                mapbox=dict(
                    bearing=0,
                    center=go.layout.mapbox.Center(
                        lat=float(clickData["points"][0]["lat"]),
                        lon=float(clickData["points"][0]["lon"]),
                    ),
                    pitch=0,
                    zoom=5.0,
                ),
            )

            trail_mini_map = html.Div(
                [
                    dcc.Graph(
                        figure=trail_mini_map,
                        config={"displayModeBar": False, "staticPlot": False},
                        style={"width": "100%", "height": "100%", "display": "flex"},
                    )
                ],
                style={
                    "height": "35vh",
                    "width": "30vw",
                    "position": "absolute",
                    "top": "1vh",
                    "left": "1vw",
                    "z-index": "2",
                    "opacity": "0.8",
                },
                className="border rounded-3",
            )

            # get trail description #
            trail_desc = data.trail_descriptions[
                data.trail_descriptions["trail_id"] == current_trail_id
            ]["description"].iloc[0]
            # trail card #
            avg_stars = int(
                data.main_map_data[data.main_map_data["trail_id"] == current_trail_id][
                    "avg_rating"
                ].iloc[0]
            )
            keywords = data.key_words[data.key_words["trail_id"] == current_trail_id][
                "set"
            ]
            trail_card = html.Div(
                [
                    html.H3(("★" * avg_stars) + ("☆" * (5 - avg_stars))),
                    html.P(trail_desc),
                ]
                + [
                    dbc.Badge(kw, color="dark", className="me-1")
                    for kw in keywords.iloc[0]
                ],
            )

            # description of trail #
            trail_description = html.Div(
                children=trail_card,
                className="p-3 bg-light border rounded-3",
                style={
                    "height": "35vh",
                    "width": "40vw",
                    "position": "absolute",
                    "top": "1vh",
                    "left": "31.5vw",
                    "z-index": "2",
                    "opacity": "0.8",
                },
                id="trail-card",
            )
            # basic stats about the trail
            length_height = data.main_map_data[
                data.main_map_data["trail_id"] == current_trail_id
            ]
            normalized_length = length_height["normalized_elevation"].iloc[0]
            normalized_elevation = length_height["normalized_length"].iloc[0]
            reviews = data.num_reviews[data.num_reviews["trail_id"] == current_trail_id]
            normalized_ratings = reviews["ratings_normalized"].iloc[0]
            normalized_written = reviews["written_normalized"].iloc[0]
            # length and elevation bar plot #
            fig1 = px.bar(
                x=["Length", "Elevation"], y=[normalized_length, normalized_elevation]
            )
            fig1.update_yaxes(visible=False)
            fig1.update_xaxes(visible=False)
            fig1.update_layout(
                title="Trail Stats",
                template="plotly_dark",
                margin=go.layout.Margin(l=0, r=0, b=0, t=100),
            )

            fig2 = px.bar(
                x=["num ratings", "num reviews"],
                y=[normalized_ratings, normalized_written],
            )
            fig2.update_yaxes(visible=False)
            fig2.update_xaxes(visible=False)
            fig2.update_layout(
                title="Review Stats",
                template="plotly_dark",
                margin=go.layout.Margin(l=0, r=0, b=0, t=100),
            )
            # length versus elev #
            length_height = html.Div(
                children=dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(
                                figure=fig1,
                                style={"width": "100%", "height": "100%"},
                                config={"displayModeBar": False, "staticPlot": False},
                            ),
                            style={
                                "width": "50%",
                                "height": "100%",
                                "padding-left": "0.5vh",
                                "padding-right": "0.5vh",
                            },
                            className="border rounded-3",
                        ),
                        dbc.Col(
                            dcc.Graph(
                                figure=fig2,
                                style={"width": "100%", "height": "100%"},
                                config={"displayModeBar": False, "staticPlot": False},
                            ),
                            style={
                                "width": "50%",
                                "height": "100%",
                                "padding-left": "0.5vh",
                                "padding-right": "0.5vh",
                            },
                            className="border rounded-3",
                        ),
                    ],
                    style={
                        "width": "100%",
                        "height": "100%",
                        "padding-left": "0vh",
                        "padding-right": "0vh",
                    },
                ),
                style={
                    "height": "35vh",
                    "width": "28vw",
                    "position": "absolute",
                    "top": "1vh",
                    "left": "72vw",
                    "z-index": "2",
                    "opacity": "0.8",
                },
                id="length-height",
            )

            # scatter plot on trail card showing length vs. elevation for similar clusters
            trail_clusterscat_data = data.cluster_map_data
            trail_clusterscat_data.loc[:, "colour"] = (
                trail_clusterscat_data["trail_id"] == current_trail_id
            ).apply(lambda x: shortened_trail_name if x else "cluster")
            scatter_l_h = px.scatter(
                x=trail_clusterscat_data["length"],
                y=trail_clusterscat_data["elevation"],
                color=trail_clusterscat_data["colour"],
            )
            scatter_l_h.update_layout(
                xaxis_title="length (Mi.)",
                yaxis_title="Elevation (ft.)",
                title="Length vs. Elevation for Associated Cluster",
                template="plotly_dark",
                margin=go.layout.Margin(l=0, r=0, b=0, t=0),
            )
            similar_trails = html.Div(
                [
                    dcc.Graph(
                        figure=scatter_l_h,
                        style={"width": "100%", "height": "100%", "display": "flex"},
                        config={"displayModeBar": False, "staticPlot": False},
                    )
                ],
                style={
                    "height": "50vh",
                    "width": "29.5vw",
                    "position": "absolute",
                    "top": "37vh",
                    "left": "1vw",
                    "z-index": "2",
                    "opacity": "0.8",
                },
                id="similar-trails",
                className="border rounded-3",
            )
            # individualized sentiment plot #
            fig = sentiment_scatter_plot(
                data,
                trail_clusterscat_data["trail_id"].tolist(),
                current_trail_id,
                shortened_trail_name,
            )
            fig.update_layout(
                title="Sentiment Analysis for Associated Cluster",
                margin=go.layout.Margin(l=0, r=0, b=0, t=0),
            )
            sentiment_scatter = dbc.Col(
                html.Div(
                    children=dcc.Graph(
                        figure=fig,
                        style={"width": "100%", "height": "100%", "display": "flex"},
                        config={"displayModeBar": False, "staticPlot": False},
                    ),
                    style={
                        "height": "50vh",
                        "width": "30.5vw",
                        "position": "absolute",
                        "top": "37vh",
                        "left": "31vw",
                        "z-index": "2",
                        "opacity": "0.8",
                    },
                    id="height-length",
                    className="border rounded-3",
                ),
            )
            # word cloud for the trail and related trails #
            full_wordcloud = get_sentiment_analysis(
                data, selected_trails=[current_trail_id], full_set=True
            )
            trail_sentiment_analysis = dbc.Col(
                html.Div(
                    children=dcc.Graph(
                        figure=full_wordcloud,
                        style={"width": "100%", "height": "100%", "display": "flex"},
                        config={"displayModeBar": False, "staticPlot": False},
                    ),
                    # className="bg-light border rounded-3",
                    style={
                        "height": "50vh",
                        "width": "36.25vw",
                        "position": "absolute",
                        "top": "37vh",
                        "left": "62vw",
                        "z-index": "2",
                        "opacity": "0.8",
                    },
                    id="trail-card-sentiment",
                    className="border rounded-3",
                )
            )

            # set header for trail card #
            modal_header = trail_name
            modal_content = [
                trail_mini_map,
                trail_description,
                sentiment_scatter,
                trail_sentiment_analysis,
                length_height,
                similar_trails,
            ]

        # MINI-MAP SWITCH VIEWS BETWEEN CLUSTER AND SENTIMENT ANALYSIS #
        if (
            is_filter
            or is_switch_mini_map_style
            or is_select_trail
            or is_select_all
            or is_select_filter
            or is_unselect_all
        ):
            filter_trails = get_filtered_map_data(
                data,
                selected_trails,
                filter_type,
                filter_rating,
                filter_length,
                filter_elev,
            )["trail_id"].to_list()
            # we must then create a new map
            main_mini_map = get_main_mini_map(data, filter_trails, main_mini_map_style)
            prev_main_mini_map_style = main_mini_map_style
            current_trail_id = None
        # toggle modal
        if not current_trail_id:
            modal_is_open = False

        return (
            has_been_init,  # passed as input
            filter_trail_prev_clicks,  # passed as input
            main_mini_map,  # not passed as input
            prev_main_mini_map_style,  # passed as input
            select_trail_prev_clicks,  # passed as input
            select_all_prev_clicks,  # passed as input
            select_filter_prev_clicks,  # passed as input
            unselect_all_prev_clicks,  # passed as input
            sentiment_analysis,  # not passed as input
            positive_sentiment,  # not passed as input
            negative_sentiment,  # not passed as input
            modal_is_open,  # passed as input
            modal_header,  # not passed as input
            modal_content,  # not passed as input
            current_trail_id,  # passed as input
            selected_trails,  # passed as input
            main_map,
        )  # not passed as input

    app.run_server(debug=False)
